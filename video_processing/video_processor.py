"""Video processing utilities for frame extraction and analysis."""

import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import tempfile


class VideoProcessor:
    """Process videos: extract frames and run predictions."""
    
    def __init__(self, fps: float = 1.0):
        """
        Initialize video processor.
        
        Args:
            fps: Frames per second to extract (default: 1.0)
        """
        self.fps = fps
    
    def extract_frames(self, video_path: str, output_dir: Optional[str] = None) -> List[Tuple[Image.Image, float]]:
        """
        Extract frames from video at specified FPS.
        
        Args:
            video_path: Path to video file
            output_dir: Optional directory to save frames (if None, frames kept in memory)
        
        Returns:
            List of (PIL Image, timestamp) tuples
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps_video / self.fps) if fps_video > 0 else 1
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame at specified interval
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                timestamp = frame_count / fps_video if fps_video > 0 else frame_count
                frames.append((pil_image, timestamp))
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def process_video_file(self, video_file, fps: float = 1.0) -> Tuple[str, List[Tuple[Image.Image, float]]]:
        """
        Process uploaded video file (Streamlit UploadedFile).
        
        Args:
            video_file: Streamlit UploadedFile object
            fps: Frames per second to extract
        
        Returns:
            Tuple of (temp_file_path, list of (PIL Image, timestamp) tuples)
        """
        self.fps = fps
        
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(video_file.read())
        
        # Extract frames
        frames = self.extract_frames(tmp_path)
        
        return tmp_path, frames
    
    def cleanup_temp_file(self, file_path: str):
        """Remove temporary video file."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Warning: Could not delete temp file {file_path}: {e}")

