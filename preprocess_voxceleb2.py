"""
Voxceleb Data preparation pipeline
1. Resample audio to 16kHz and video to 25 fps
2. Detect scene and segment the video when the scene changes
3. Do face detection and landmark detection and use the landmarks to crop and affine transform the face
4. Do another round of face detection to filter out bad faces
5. Do Audio video synchronization using syncnet
6. Filter out low quality videos using Hyper IQA
"""
from pipecraft.sources import DataSource, NO_DATA
from pipecraft.processors import DataProcessor
from pipecraft.sinks import DataSink
from pipecraft.pipelines.multi_proc import run_pipeline
from pipecraft.utils.media import (
    get_video_fps,
    read_video,
    write_video,
)
from pipecraft.utils.face_align import FaceAlignmentProcessor

from dataclasses import dataclass
import os
from typing import List, Tuple, Optional
import numpy as np
import subprocess
from tqdm import tqdm
import threading
import time
import uuid
import traceback
import gc

def gather_video_paths_iter(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        for file in sorted(files):
            if file.endswith(".mp4"):
                rel_path = os.path.relpath(root, input_dir)
                video_input = os.path.join(root, file)
                video_output_dir = os.path.join(output_dir, rel_path)
                video_output = os.path.join(video_output_dir, file)
                if not os.path.isfile(video_output):
                    yield video_input, video_output, rel_path

def gather_video_paths(input_dir, output_dir):
    video_paths = []
    for paths in gather_video_paths_iter(input_dir, output_dir):
        video_paths.append(paths)
        
    # Sort the video paths
    video_paths.sort()
    return video_paths

@dataclass
class VideoPaths:
    video_input: str
    video_resampled: str
    video_temp: str
    video_output: str

@dataclass
class VideoObjects(VideoPaths):
    video_frames: List[np.ndarray]

class AVPathGenerator(DataSource[VideoPaths]):
    def __init__(self, video_paths, output_dir, world_size=1, rank=0, verbose=False, assigned_cores: Optional[List[int]] = None): 
        super().__init__(
            buffer_size=None,
            num_workers=1,
            verbose=verbose,
            assigned_cores=assigned_cores
        )
        # self.video_paths_iter = gather_video_paths_iter(input_dir, output_dir)
        self.output_dir = output_dir
        print(f"Total videos for Rank {rank} is {len(video_paths)}")
        self.total_paths = len(video_paths)
        self.video_paths_iter = iter(video_paths)
        
    def fetch(self, **kwargs) -> VideoPaths:
        try:
            video_input, video_output, rel_path = next(self.video_paths_iter)
            video_resampled = os.path.join(f"{self.output_dir}_resampled", rel_path, os.path.basename(video_input))
            video_temp = os.path.join(f"{self.output_dir}_temp", rel_path, os.path.basename(video_input).split(".")[0])
            
            if self.verbose:
                print(f"Processing {video_input} -> {video_output}, {video_resampled}, {video_temp}")
            
            return VideoPaths(
                video_input=video_input, 
                video_output=video_output,
                video_resampled=video_resampled,
                video_temp=video_temp,
            )
        except StopIteration:
            return NO_DATA
        except Exception as e:
            print(f"Error in AVPathGenerator: {e}")
            traceback.print_exc()
            return None

    def close(self):
        if self.video_paths_iter is not None:
            self.video_paths_iter = None


# Stage 1: Resample audio and video
class AVResample(DataProcessor[VideoPaths, VideoPaths]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def process(self, video_paths: VideoPaths, **kwargs) -> VideoPaths:
        os.makedirs(os.path.dirname(video_paths.video_resampled), exist_ok=True)
        try:
            if get_video_fps(video_paths.video_input) == 25:
                command = f"ffmpeg -loglevel error -y -i {video_paths.video_input} -c:v copy -ar 16000 -q:a 0 {video_paths.video_resampled}"
            else:
                command = f"ffmpeg -loglevel error -y -i {video_paths.video_input} -r 25 -ar 16000 -q:a 0 {video_paths.video_resampled}"
            subprocess.run(command, shell=True)
            if self.verbose:
                print(f"Resampled {video_paths.video_input} to {video_paths.video_resampled}")
        except Exception as e:
            print(f"Error processing {video_paths.video_input}: {e}")
            traceback.print_exc()
        return video_paths
        
# Stage 2: Read video frames for further processing
class AVDataReader(DataProcessor[VideoPaths, VideoObjects]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def process(self, video_paths: VideoPaths, **kwargs) -> VideoObjects:
        video_frames = read_video(video_paths.video_resampled, change_fps=False, reader="decord")
        if video_frames is None:
            return None
        if len(video_frames) == 0:
            print(f"Error reading video frames for {video_paths.video_resampled}")
            return None
        # Delete resampled video to save space
        # os.remove(video_paths.video_resampled)
        # Emit the video frames and paths
        return VideoObjects(
            video_input=video_paths.video_input,
            video_output=video_paths.video_output,
            video_resampled=video_paths.video_resampled,
            video_temp=video_paths.video_temp,
            video_frames=video_frames
        )
    
    def close(self):
        return super().close()
    
    
# Stage 3: Affine Transformation using landmarks
class AVAffineTransform(DataProcessor[VideoObjects, VideoObjects]):
    def __init__(self, resolution=256, wandb_run=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aligners: List[FaceAlignmentProcessor] = []
        for worker in range(self._num_workers):
            self.aligners.append(FaceAlignmentProcessor(resolution=resolution))
        self.success_count = 0
        self.failure_count = 0
        self.wandb_run = wandb_run
        
    def process(self, video_objects: VideoObjects, threadId: int) -> VideoObjects:
        try:
            # Perform affine transformation on the frames
            aligner = self.aligners[threadId]
            aligned = aligner.process_frames(video_objects.video_frames, break_on_error=True)
            
            # Clear the original frames to free memory
            original_frames = video_objects.video_frames
            video_objects.video_frames = None
            
            if aligned is None or len(aligned) != len(original_frames):
                self.failure_count += 1
                print(f"Error in affine transformation for {video_objects.video_input}")
                # Important: explicitly delete large objects
                del original_frames
                return None
                
            if self.verbose:
                print(f"Aligned {video_objects.video_input} with {len(aligned)} frames")
                
            # Now do a face detection on the aligned frames
            detected_faces = aligner.detect_faces(aligned, break_on_error=True)
            if detected_faces is None or len(detected_faces) != len(aligned):
                self.failure_count += 1
                print(f"Error in face detection for {video_objects.video_input}")
                # Important: explicitly delete large objects
                del aligned
                del original_frames
                return None
                
            # Replace with aligned frames
            video_objects.video_frames = aligned
            
            # Explicitly delete original frames to free memory
            del original_frames
            
            self.success_count += 1
            success_rate = self.success_count / (1e-6 + self.success_count + self.failure_count)
            if self.wandb_run is not None:
                self.wandb_run.log({"success_rate": success_rate}, step=self.success_count + self.failure_count)
            return video_objects
        except Exception as e:
            print(f"Error in AVAffineTransform processing {video_objects.video_input}: {e}")
            traceback.print_exc()
            return None
    
    def close(self):
        # Clean up aligners
        for aligner in self.aligners:
            if hasattr(aligner, 'close'):
                aligner.close()
        self.aligners.clear()
        super().close()

        
# Stage 5: Write the processed video frames to disk
class AVWrite(DataSink[VideoObjects]):
    def __init__(self, process_temp_dir, total_paths=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_paths = total_paths
        self.lock = threading.Lock()
        if total_paths is not None:
            # Initialize pbar with total paths
            self.pbar = tqdm(total=total_paths, desc="Writing videos")
        else:
            self.pbar = None
        os.makedirs(process_temp_dir, exist_ok=True)
        self.process_temp_dir = process_temp_dir
        self.counter = 0
        
    def write(self, video_objects: VideoObjects, threadId: int) -> None:
        audio_temp = None
        video_temp = None
        
        try:
            os.makedirs(os.path.dirname(video_objects.video_output), exist_ok=True)
            
            # Generate a random name for the video to avoid conflicts
            video_name = os.path.splitext(os.path.basename(video_objects.video_resampled))[0]
            video_name = f"{video_name}_{hash(time.time())}_{uuid.uuid4()}"
            
            audio_temp = os.path.join(self.process_temp_dir, f"{video_name}_temp.wav")
            video_temp = os.path.join(self.process_temp_dir, f"{video_name}_temp.mp4")

            write_video(video_temp, video_objects.video_frames, fps=25)
            
            # Free video frames memory early
            del video_objects.video_frames
            video_objects.video_frames = None

            command = f"ffmpeg -y -loglevel error -i {video_objects.video_resampled} -q:a 0 -map a {audio_temp}"
            subprocess.run(command, shell=True)

            os.makedirs(os.path.dirname(video_objects.video_output), exist_ok=True)
            command = f"ffmpeg -y -loglevel error -i {video_temp} -i {audio_temp} -c:v libx264 -c:a aac -map 0:v -map 1:a -q:v 0 -q:a 0 {video_objects.video_output}"
            subprocess.run(command, shell=True)
            
            os.remove(video_objects.video_resampled)

            if self.verbose:
                print(f"Written {video_objects.video_output}")
                
            if self.pbar is not None:
                with self.lock:
                    self.pbar.update(1)
                    self.counter += 1
                    if self.counter % 20 == 0:
                        gc.collect()
            return True
        except Exception as e:
            print(f"Error combining video and audio: {str(e)}")
            traceback.print_exc()
            return False
        finally:
            # Always clean up temporary files
            try:
                if audio_temp and os.path.exists(audio_temp):
                    os.remove(audio_temp)
                if video_temp and os.path.exists(video_temp):
                    os.remove(video_temp)
            except Exception as e:
                print(f"Error cleaning up temp files: {e}")

    def close(self):
        # Clean up the progress bar
        if self.pbar:
            self.pbar.close()
            self.pbar = None
        super().close()

def pipeline(
    video_paths: List[Tuple[str, str, str]], # Changed: Expect list of tuples
    output_dir: str,
    process_temp_dir: str,
    
    world_size: int,
    rank: int,
    num_workers_per_process: int,
    wandb_run: Optional[any] = None,
    verbose: bool = False,
    assigned_cores: Optional[List[int]] = None, # Changed: Expect list of cores
):
    av_path_generator = AVPathGenerator(video_paths, output_dir, world_size, rank, verbose=verbose, assigned_cores=assigned_cores)
    total_paths = av_path_generator.total_paths
    av_resample = AVResample(sources=[av_path_generator], num_workers=num_workers_per_process, verbose=verbose, assigned_cores=assigned_cores)
    av_data_reader = AVDataReader(sources=[av_resample], num_workers=num_workers_per_process, verbose=verbose, assigned_cores=assigned_cores)
    av_affine_transform = AVAffineTransform(
        sources=[av_data_reader], 
        num_workers=num_workers_per_process, 
        verbose=verbose,
        wandb_run=wandb_run,
        assigned_cores=assigned_cores,
    )
    av_write = AVWrite(process_temp_dir, total_paths, sources=[av_affine_transform], num_workers=num_workers_per_process, verbose=verbose, assigned_cores=assigned_cores)

    # Run the pipeline
    av_path_generator.start()
    av_resample.start()
    av_data_reader.start()
    av_affine_transform.start()
    # av_face_detection.start()
    av_write.start()
        
    # Wait for the pipeline to finish
    av_write.join()
    # av_affine_transform.join()

    print(f"Rank {rank} (PID {os.getpid()}): Pipeline join complete.")

if __name__ == "__main__":
    # run_pipeline(
    #     input_dir='/home/mrwhite0racle/persist/data/vox2/test/',
    #     output_dir='/home/mrwhite0racle/persist/data/vox2/test_output2/',
    #     process_temp_dir='/home/mrwhite0racle/persist/data/vox2/temp/',
    #     num_processes=60,
    #     num_workers_per_process=4,
    #     use_wandb=True,
    # )
    # input_dir='/home/mrwhite0racle/persist/data/vox2/train/'
    # output_dir='/home/mrwhite0racle/persist/data/vox2/train_output/'
    # process_temp_dir='/home/mrwhite0racle/persist/data/vox2/temp/'
    input_dir='/home/mrwhite0racle/persist/data/vox2/test/'
    output_dir='/home/mrwhite0racle/persist/data/vox2/test_output2/'
    process_temp_dir='/home/mrwhite0racle/persist/data/vox2/temp/'

    output_dir = output_dir[:-1] if output_dir.endswith("/") else output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(process_temp_dir, exist_ok=True)
    video_paths = gather_video_paths(input_dir, output_dir)
    print(f"Total video paths to process: {len(video_paths)}")
    pipeline_opts = {
        'resolution': 256,
        'video_paths': video_paths,
        'output_dir': output_dir,
        'process_temp_dir': process_temp_dir,
    }

    run_pipeline(
        pipeline_opts=pipeline_opts,
        pipeline=pipeline,
        num_processes=60,
        num_workers_per_process=4,
        use_wandb=True,
    )