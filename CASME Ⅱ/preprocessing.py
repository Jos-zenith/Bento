# ============================================================================
# CASME II Preprocessing - Optical Flow & Data Augmentation
# ============================================================================

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from typing import Tuple, List, Optional
import random
from pathlib import Path


class OpticalFlowProcessor:
    """Computes optical flow between frames to capture micro-movements"""
    
    def __init__(self, method: str = "farneback", clip_value: float = 20.0):
        """
        Args:
            method: 'farneback' or 'lucas_kanade'
            clip_value: Clip optical flow magnitude to this value
        """
        self.method = method
        self.clip_value = clip_value
    
    def compute_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Compute optical flow between two frames.
        
        Args:
            frame1: First frame (BGR or grayscale)
            frame2: Second frame (BGR or grayscale)
            
        Returns:
            flow: Optical flow with shape (H, W, 2) - [u, v] components
        """
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray1, gray2 = frame1, frame2
        
        if self.method == "farneback":
            flow = np.zeros((gray1.shape[0], gray1.shape[1], 2), dtype=np.float32)
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, 
                flow,
                0.5,     # pyr_scale
                3,       # levels
                15,      # winsize
                3,       # iterations
                5,       # poly_n
                1.2,     # poly_sigma
                cv2.OPTFLOW_FARNEBACK_GAUSSIAN
            )
        elif self.method == "lucas_kanade":
            # Lucas-Kanade with feature detection
            prev_gray = gray1
            curr_gray = gray2
            
            p0 = cv2.goodFeaturesToTrack(
                prev_gray,
                maxCorners=200,
                qualityLevel=0.01,
                minDistance=7,
                blockSize=7
            )
            
            if p0 is None:
                flow = np.zeros((gray1.shape[0], gray1.shape[1], 2), dtype=np.float32)
            else:
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    prev_gray, curr_gray, p0, np.array([]),
                    criteria=criteria
                )
                
                flow = np.zeros((gray1.shape[0], gray1.shape[1], 2), dtype=np.float32)
                if p1 is not None:
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]
                    for new, old in zip(good_new, good_old):
                        a, b = new.ravel()
                        c, d = old.ravel()
                        flow[int(b), int(a)] = [a - c, b - d]
        else:
            raise ValueError(f"Unknown optical flow method: {self.method}")
        
        return flow
    
    def normalize_flow(self, flow: np.ndarray) -> np.ndarray:
        """Normalize optical flow to [-1, 1] range"""
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag = np.clip(mag, 0, self.clip_value)
        mag = mag / (self.clip_value + 1e-6)
        return mag
    
    def flow_to_hsv(self, flow: np.ndarray) -> np.ndarray:
        """Convert optical flow to HSV visualization (for debugging)"""
        h, w = flow.shape[:2]
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
        # Normalize magnitude to [0, 255]
        mag_min, mag_max = mag.min(), mag.max()
        if mag_max > mag_min:
            mag_normalized = ((mag - mag_min) / (mag_max - mag_min) * 255).astype(np.uint8)
        else:
            mag_normalized = np.zeros_like(mag, dtype=np.uint8)
        hsv[..., 1] = mag_normalized
        hsv[..., 2] = 255
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class FrameSequenceProcessor:
    """Process frame sequences: load, resize, normalize"""
    
    def __init__(self, frame_size: Tuple[int, int] = (224, 224)):
        self.frame_size = frame_size
    
    def load_frame(self, path: str) -> np.ndarray:
        """Load and preprocess a single frame"""
        image_path = Path(path)
        if not image_path.exists():
            raise FileNotFoundError(f"Cannot load image: {path}")

        # OpenCV can fail on Unicode Windows paths; decode from bytes instead.
        buffer = np.fromfile(str(image_path), dtype=np.uint8)
        img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.frame_size, interpolation=cv2.INTER_LINEAR)
        return img
    
    def load_sequence(self, frame_paths: List[str]) -> np.ndarray:
        """
        Load a sequence of frames.
        
        Args:
            frame_paths: List of frame paths in order
            
        Returns:
            frames: Array of shape (T, H, W, 3) in RGB format
        """
        frames = []
        for path in frame_paths:
            frames.append(self.load_frame(path))
        return np.array(frames, dtype=np.float32)

    def load_video_sequence(self, video_path: str) -> np.ndarray:
        """
        Load and preprocess all frames from a video file.

        Args:
            video_path: Path to a video file such as .avi

        Returns:
            frames: Array of shape (T, H, W, 3) in RGB format
        """
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Cannot load video: {video_path}")

        capture = cv2.VideoCapture(str(video_file))
        if not capture.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        frames = []
        try:
            while True:
                ret, frame = capture.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_LINEAR)
                frames.append(frame)
        finally:
            capture.release()

        if not frames:
            raise ValueError(f"No frames decoded from video: {video_path}")

        return np.array(frames, dtype=np.float32)
    
    def sample_frames(self, frames: np.ndarray, num_frames: int) -> np.ndarray:
        """
        Uniformly sample num_frames from the sequence.
        
        Args:
            frames: Array of shape (T, H, W, 3)
            num_frames: Number of frames to sample
            
        Returns:
            sampled_frames: Array of shape (num_frames, H, W, 3)
        """
        total_frames = len(frames)
        if total_frames <= num_frames:
            # Pad if necessary
            padding = np.zeros((num_frames - total_frames, *frames.shape[1:]), dtype=frames.dtype)
            return np.vstack([frames, padding])
        
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        return frames[indices]
    
    def normalize_frames(self, frames: np.ndarray, mean, std) -> np.ndarray:
        """
        Normalize frames using ImageNet stats.
        
        Args:
            frames: Array of shape (T, H, W, 3) with values in [0, 255]
            mean: List/array of mean values
            std: List/array of std values
            
        Returns:
            normalized_frames: Array with normalized values
        """
        frames = frames / 255.0  # Convert to [0, 1]
        mean = np.array(mean).reshape(1, 1, 1, 3)
        std = np.array(std).reshape(1, 1, 1, 3)
        return (frames - mean) / (std + 1e-6)


class AugmentationPipeline:
    """Data augmentation for video frames"""
    
    def __init__(self, params: Optional[dict] = None):
        """
        Args:
            params: Dictionary with augmentation parameters
                - random_flip: Probability of horizontal flip
                - random_rotation: Max rotation angle in degrees
                - random_brightness: Brightness delta (as fraction)
                - random_contrast: Contrast range as [1-x, 1+x]
                - gaussian_noise_std: Gaussian noise standard deviation
        """
        self.params = params or {}
    
    def random_flip(self, frames: np.ndarray, p: float = 0.5) -> np.ndarray:
        """Randomly flip frames horizontally"""
        if random.random() < p:
            return np.fliplr(frames)
        return frames
    
    def random_rotation(self, frames: np.ndarray, max_angle: int = 15) -> np.ndarray:
        """Apply random rotation to all frames"""
        angle = random.uniform(-max_angle, max_angle)
        h, w = frames.shape[1:3]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        rotated_frames = []
        for frame in frames:
            rotated = cv2.warpAffine(frame, M, (w, h))
            rotated_frames.append(rotated)
        
        return np.array(rotated_frames)
    
    def random_brightness(self, frames: np.ndarray, delta: float = 0.2) -> np.ndarray:
        """Randomly adjust brightness"""
        brightness_delta = random.uniform(-delta, delta)
        return np.clip(frames + brightness_delta * 255, 0, 255)
    
    def random_contrast(self, frames: np.ndarray, range_val: float = 0.2) -> np.ndarray:
        """Randomly adjust contrast"""
        contrast_factor = random.uniform(1 - range_val, 1 + range_val)
        mean = frames.mean(axis=(1, 2, 3), keepdims=True)
        return np.clip((frames - mean) * contrast_factor + mean, 0, 255)
    
    def gaussian_noise(self, frames: np.ndarray, std: float = 0.01) -> np.ndarray:
        """Add Gaussian noise"""
        noise = np.random.normal(0, std * 255, frames.shape)
        return np.clip(frames + noise, 0, 255)
    
    def apply(self, frames: np.ndarray) -> np.ndarray:
        """Apply augmentation pipeline"""
        if self.params.get('random_flip', 0) > 0:
            frames = self.random_flip(frames, self.params['random_flip'])
        
        if self.params.get('random_rotation', 0) > 0:
            frames = self.random_rotation(frames, self.params['random_rotation'])
        
        if self.params.get('random_brightness', 0) > 0:
            frames = self.random_brightness(frames, self.params['random_brightness'])
        
        if self.params.get('random_contrast', 0) > 0:
            frames = self.random_contrast(frames, self.params['random_contrast'])
        
        if self.params.get('gaussian_noise_std', 0) > 0:
            frames = self.gaussian_noise(frames, self.params['gaussian_noise_std'])
        
        return frames


# ============================================================================
# PyTorch Transforms
# ============================================================================

def get_torch_transforms(mean: List[float], std: List[float]):
    """Get PyTorch transform pipeline"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
