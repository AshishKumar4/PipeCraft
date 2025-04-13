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

from dataclasses import dataclass
import os
from typing import List, Tuple, Optional
import numpy as np
import subprocess
from tqdm import tqdm
import shutil
import threading
import time
import uuid
import traceback
import gc
import torch
from einops import rearrange
from pipecraft.utils.hyper_iqa import load_hyper_model, get_hyperiqa_score

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


# Global cache to optimize repeated file lookups
class FileExistenceCache:
    def __init__(self, capacity=100000):
        self.cache = {}
        self.capacity = capacity
    
    def file_exists(self, path):
        if path in self.cache:
            return self.cache[path]
        
        # Fast existence check
        try:
            os.stat(path)
            exists = True
        except FileNotFoundError:
            exists = False
        
        # Manage cache size
        if len(self.cache) >= self.capacity:
            # Simple eviction - clear half the cache when full
            keys = list(self.cache.keys())[:self.capacity//2]
            for k in keys:
                del self.cache[k]
        
        self.cache[path] = exists
        return exists

# Create a global cache instance
file_cache = FileExistenceCache()

def gather_video_paths_cached(input_dir, output_dir, workers=None):
    """Version with file existence caching for repeated calls"""
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    
    # Find all mp4 files using a more efficient approach
    video_input_paths = []
    for root, _, files in os.walk(input_dir, followlinks=False):
        for file in files:
            if file.endswith(".mp4"):
                video_input_paths.append(os.path.join(root, file))
    
    # Sort once
    video_input_paths.sort()
    
    # Process in larger batches with caching
    video_paths = []
    input_prefix_len = len(input_dir) + 1
    
    for input_path in video_input_paths:
        rel_path = input_path[input_prefix_len:]
        output_path = os.path.join(output_dir, rel_path)
        
        if not file_cache.file_exists(output_path):
            video_paths.append((input_path, output_path, rel_path))
    
    return video_paths

@dataclass
class VideoPaths:
    video_input: str
    video_temp: str
    video_output: str

@dataclass
class VideoObjects(VideoPaths):
    video_frames: List[np.ndarray]

class AVPathGenerator(DataSource[VideoPaths]):
    def __init__(self, video_paths, output_dir, **kwargs): 
        kwargs = dict(kwargs)
        del kwargs['num_workers']
        
        self.output_dir = output_dir
        print(f"Total videos for Rank {kwargs['rank']} is {len(video_paths)}")
        self.total_paths = len(video_paths)
        self.video_paths_iter = iter(video_paths)
        
        super().__init__(
            buffer_size=None,
            num_workers=1,
            **kwargs
        )
    
    def __len__(self):
        return self.total_paths
        
    def fetch(self, **kwargs) -> VideoPaths:
        try:
            video_input, video_output, rel_path = next(self.video_paths_iter)
            video_temp = os.path.join(f"{self.output_dir}_temp", rel_path, os.path.basename(video_input).split(".")[0])
            
            if self.verbose:
                print(f"Processing {video_input} -> {video_output}, {video_temp}")
            
            return VideoPaths(
                video_input=video_input, 
                video_output=video_output,
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
    def process(self, video_paths: VideoPaths, **kwargs) -> VideoPaths:
        os.makedirs(os.path.dirname(video_paths.video_temp), exist_ok=True)
        try:
            video_resampled = f"{video_paths.video_temp}_resampled.mp4"
            if get_video_fps(video_paths.video_input) == 25:
                command = f"ffmpeg -loglevel error -y -i {video_paths.video_input} -c:v copy -ar 16000 -q:a 0 {video_resampled}"
            else:
                command = f"ffmpeg -loglevel error -y -i {video_paths.video_input} -r 25 -ar 16000 -q:a 0 {video_resampled}"
            subprocess.run(command, shell=True)
            if self.verbose:
                print(f"Resampled {video_paths.video_input} to {video_resampled}")
                video_paths.video_input = video_resampled
        except Exception as e:
            print(f"Error processing {video_paths.video_input}: {e}")
            traceback.print_exc()
        return video_paths
        
# Stage 2: Read video frames for further processing
class AVDataReader(DataProcessor[VideoPaths, VideoObjects]):
    def process(self, video_paths: VideoPaths, **kwargs) -> VideoObjects:
        video_frames = read_video(video_paths.video_input, change_fps=False, reader="rsreader_fast")
        if video_frames is None:
            return None
        if len(video_frames) == 0:
            print(f"Error reading video frames for {video_paths.video_input}")
            return None
        # Delete resampled video to save space
        # os.remove(video_paths.video_resampled)
        # Emit the video frames and paths
        return VideoObjects(
            video_input=video_paths.video_input,
            video_output=video_paths.video_output,
            video_temp=video_paths.video_temp,
            video_frames=video_frames
        )
    
    def close(self):
        return super().close()
    
    
# Stage 3: Affine Transformation using landmarks
class AVAffineTransform(DataProcessor[VideoObjects, VideoObjects]):
    def __init__(self, resolution=256, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from pipecraft.utils.face_align import FaceAlignmentProcessor
        self.aligners: List[FaceAlignmentProcessor] = []
        for worker in range(self._num_workers):
            self.aligners.append(FaceAlignmentProcessor(resolution=resolution))
        self.success_count = 0
        
    def process(self, video_objects: VideoObjects, threadId: int) -> VideoObjects:
        try:
            # Perform affine transformation on the frames
            aligner = self.aligners[threadId]
            aligned = aligner.process_frames(video_objects.video_frames, break_on_error=True)
            
            # Clear the original frames to free memory
            original_frames = video_objects.video_frames
            video_objects.video_frames = None
            
            if aligned is None or len(aligned) != len(original_frames):
                print(f"Error in affine transformation for {video_objects.video_input}")
                # Important: explicitly delete large objects
                del original_frames
                return None
                
            if self.verbose:
                print(f"Aligned {video_objects.video_input} with {len(aligned)} frames")
                
            # Now do a face detection on the aligned frames
            detected_faces = aligner.detect_faces(aligned, break_on_error=True)
            if detected_faces is None or len(detected_faces) != len(aligned):
                print(f"Error in face detection for {video_objects.video_input}")
                # Important: explicitly delete large objects
                del aligned
                del original_frames
                return None
                
            # Replace with aligned frames
            video_objects.video_frames = aligned
            
            # Explicitly delete original frames to free memory
            del original_frames
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
        

# Stage 5: Filter videos based on quality
class AVHyperIQA(DataProcessor[VideoObjects, VideoObjects]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Try to get CUDA device
        
        self.device = None
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        # else:
        #     try:
        #         import torch_xla.core.xla_model as xm
        #         self.device = xm.xla_device()
        #     finally:
        #         pass
        if self.device is None:
            self.device = torch.device("cpu")
        self.hyper_model = load_hyper_model(self.device)
        
    def process(self, video_objects: VideoObjects, threadId: int) -> VideoObjects:
        try:
            frames = video_objects.video_frames
            first_frame = frames[0]
            middle_frame = frames[len(frames) // 2]
            last_frame = frames[-1]
            video_frames = np.stack([first_frame, middle_frame, last_frame], axis=0)
            video_frames = torch.from_numpy(video_frames)
            video_frames = video_frames.permute(0, 3, 1, 2)  # Convert to CxHxW
            video_frames = video_frames / 255.0
            # Get the hyperiqa score
            video_score = get_hyperiqa_score(video_frames, self.hyper_model, self.device)
            if video_score is None:
                print(f"Error getting hyperiqa score for {video_objects.video_input}")
                return None
            
            # Filter based on the score
            if video_score >= 40:
                # Save the video frames
                if video_score > 100:
                    print(f"Warning: HyperIQA score {video_score} is unusually high for {video_objects.video_input}")
                return video_objects
            else:
                # print(f"Filtered out {video_objects.video_input} with score {video_score}")
                # Explicitly delete frames to free memory
                del video_objects.video_frames
                return None
        except Exception as e:
            print(f"Error in AVHyperIQA processing {video_objects.video_input}: {e}")
            traceback.print_exc()
        return None
    
    def close(self):
        super().close()
        
# Stage 6: Write the processed video frames to disk
class AVWrite(DataSink[VideoObjects]):
    def __init__(self, process_temp_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock = threading.Lock()
        os.makedirs(process_temp_dir, exist_ok=True)
        self.process_temp_dir = process_temp_dir
        self.counter = 0
        
    def write(self, video_objects: VideoObjects, threadId: int) -> None:
        audio_temp = None
        video_temp = None
        
        try:
            os.makedirs(os.path.dirname(video_objects.video_output), exist_ok=True)
            
            # Generate a random name for the video to avoid conflicts
            video_name = os.path.splitext(os.path.basename(video_objects.video_input))[0]
            video_name = f"{video_name}_{hash(time.time())}_{uuid.uuid4()}"
            
            audio_temp = os.path.join(self.process_temp_dir, f"{video_name}_temp.wav")
            video_temp = os.path.join(self.process_temp_dir, f"{video_name}_temp.mp4")

            write_video(video_temp, video_objects.video_frames, fps=25)
            
            # Free video frames memory early
            del video_objects.video_frames
            video_objects.video_frames = None

            command = f"ffmpeg -y -loglevel error -i {video_objects.video_input} -q:a 0 -map a {audio_temp}"
            subprocess.run(command, shell=True)

            os.makedirs(os.path.dirname(video_objects.video_output), exist_ok=True)
            command = f"ffmpeg -y -loglevel error -i {video_temp} -i {audio_temp} -c:v libx264 -c:a aac -map 0:v -map 1:a -q:v 0 -q:a 0 {video_objects.video_output}"
            subprocess.run(command, shell=True)
            
            os.remove(video_objects.video_input)

            if self.verbose:
                print(f"Written {video_objects.video_output}")
                
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
        
class AVCopy(DataSink[VideoPaths]):
    def write(self, video_paths: VideoPaths, threadId: int) -> None:
        os.makedirs(os.path.dirname(video_paths.video_output), exist_ok=True)
        # Use shutil.copy to copy the video file from input to output
        try:
            shutil.copy(video_paths.video_input, video_paths.video_output)
            if self.verbose:
                print(f"Copied {video_paths.video_input} to {video_paths.video_output}")
        except Exception as e:
            print(f"Error copying {video_paths.video_input} to {video_paths.video_output}: {e}")
            traceback.print_exc()

def pipeline_main(
    video_paths: List[Tuple[str, str, str]], # Changed: Expect list of tuples
    output_dir: str,
    process_temp_dir: str,
    
    wandb_run=None,
    **kwargs
):
    def on_success(data, success_rate):
        if wandb_run:
            wandb_run.log({"success_rate": success_rate})
        print(f"Success rate: {success_rate}")
        
    av_path_generator = AVPathGenerator(video_paths, output_dir, **kwargs)
    av_resample = AVResample(sources=[av_path_generator], **kwargs)
    av_data_reader = AVDataReader(sources=[av_resample], **kwargs)
    av_affine_transform = AVAffineTransform(
        sources=[av_data_reader], 
        on_success=on_success,
        **kwargs
    )
    av_write = AVWrite(process_temp_dir, sources=[av_affine_transform], **kwargs)

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

    print(f"Rank {kwargs['rank']} (PID {os.getpid()}): Pipeline join complete.")
    
def pipeline_filter(
    video_paths: List[Tuple[str, str, str]], # Changed: Expect list of tuples
    output_dir: str,
    process_temp_dir: str,
        
    wandb_run=None,
    **kwargs,
):
    def on_success(data, success_rate):
        if wandb_run:
            wandb_run.log({"success_rate": success_rate})
        print(f"Success rate: {success_rate}")
    
    av_path_generator = AVPathGenerator(video_paths, output_dir, **kwargs)
    av_data_reader = AVDataReader(sources=[av_path_generator], **kwargs)
    av_hyper_iqa = AVHyperIQA(sources=[av_data_reader], on_success=on_success, **kwargs)
    av_copy = AVCopy(sources=[av_hyper_iqa], **kwargs)
    
    # Run the pipeline
    av_path_generator.start()
    av_data_reader.start()
    av_hyper_iqa.start()
    av_copy.start()
    
    # Wait for the pipeline to finish
    av_copy.join()
    av_hyper_iqa.join()
    print(f"Rank {kwargs['rank']} (PID {os.getpid()}): Pipeline join complete.")
    
def prepare_pipeline_opts(
    input_dir: str,
    output_dir: str,
    process_temp_dir: str,
):
    output_dir = output_dir[:-1] if output_dir.endswith("/") else output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(process_temp_dir, exist_ok=True)
    print(f"Gathering video paths from {input_dir} to {output_dir}")
    video_paths = gather_video_paths_cached(input_dir, output_dir)
    print(f"Total video paths to process: {len(video_paths)}")
    pipeline_opts = {
        # 'resolution': 256,
        'video_paths': video_paths,
        'output_dir': output_dir,
        'process_temp_dir': process_temp_dir,
    }
    return pipeline_opts

if __name__ == "__main__":
    process_temp_dir='/home/mrwhite0racle/persist/data/vox2/temp/'
    
    # input_dir='/home/mrwhite0racle/persist/data/vox2/train/'
    # output_dir='/home/mrwhite0racle/persist/data/vox2/train_output/'
    
    # input_dir='/home/mrwhite0racle/persist/data/vox2/test/'
    # output_dir='/home/mrwhite0racle/persist/data/vox2/test_output/'

    # run_pipeline(
    #     pipeline_opts=prepare_pipeline_opts(input_dir, output_dir, process_temp_dir),
    #     pipeline=pipeline_main,
    #     num_processes=60,
    #     num_workers_per_process=4,
    #     use_wandb=True,
    # )
    
    
    input_dir='/home/mrwhite0racle/persist/data/vox2/train_output/'
    output_dir='/home/mrwhite0racle/persist/data/vox2/train_filtered/'
    # input_dir='/home/mrwhite0racle/persist/data/vox2/test_output/'
    # output_dir='/home/mrwhite0racle/persist/data/vox2/test_filtered/'

    run_pipeline(
        pipeline_opts=prepare_pipeline_opts(input_dir, output_dir, process_temp_dir),
        pipeline=pipeline_filter,
        num_processes=60,
        num_workers_per_process=4,
        use_wandb=True,
    )