import cv2
import numpy as np
import subprocess
import shutil
from decord import VideoReader
from video_reader import PyVideoReader
import os

def get_video_fps(video_path: str):
    cam = cv2.VideoCapture(video_path)
    fps = cam.get(cv2.CAP_PROP_FPS)
    cam.release()
    return fps

def read_video(video_path: str, change_fps=True, reader="rsreader"):
    temp_dir = None
    try:
        if change_fps:
            print(f"Changing fps of {video_path} to 25")
            temp_dir = "temp"
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)
            command = (
                f"ffmpeg -loglevel error -y -nostdin -i {video_path} -r 25 -crf 18 {os.path.join(temp_dir, 'video.mp4')}"
            )
            subprocess.run(command, shell=True)
            target_video_path = os.path.join(temp_dir, "video.mp4")
        else:
            target_video_path = video_path

        if reader == "rsreader":
            return read_video_rsreader(target_video_path)
        elif reader == "rsreader_fast":
            return read_video_rsreader(target_video_path, fast=True)
        elif reader == "decord":
            return read_video_decord(target_video_path)
        elif reader == "opencv":
            return read_video_opencv(target_video_path)
        else:
            raise ValueError(f"Unknown reader: {reader}")
    finally:
        # Clean up temp directory when done
        if change_fps and temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def read_video_decord(video_path: str):
    vr = VideoReader(video_path)
    video_frames = vr[:].asnumpy()
    vr.seek(0)
    return video_frames

# Fixed OpenCV video reader - properly release resources
def read_video_opencv(video_path):
    cap = cv2.VideoCapture(video_path)
    try:
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        return np.array(frames)[:, :, :, ::-1]
    finally:
        cap.release()

def read_video_rsreader(video_path, fast=False):
    vr = PyVideoReader(video_path)
    return vr.decode_fast() if fast else vr.decode()

def write_video(video_output_path: str, video_frames: np.ndarray, fps: int):
    height, width = video_frames[0].shape[:2]
    out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    # out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*"vp09"), fps, (width, height))
    for frame in video_frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()