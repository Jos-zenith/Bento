"""
Micro-expression datasets: CASME II and SAMM.

Micro-expressions are brief, involuntary expressions lasting 1/3 to 1/25 second.
Contain the most genuine emotional information.
"""

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class MicroExpressionDataset(Dataset):
    """Base class for micro-expression datasets."""
    
    MICRO_EMOTION_TO_IDX = {
        "positive": 0,
        "negative": 1,
        "surprise": 2,
        "repression": 3,
        "others": 4,
    }
    
    MICRO_EMOTIONS = list(MICRO_EMOTION_TO_IDX.keys())
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize micro-expression dataset.
        
        Args:
            data_dir: Path to dataset directory
            transform: Image transforms to apply
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        self.video_sequences = []
        self.labels = []
        self.onset_frames = []  # When micro-expression starts
        self.apex_frames = []   # Peak of micro-expression
        self.offset_frames = []  # When micro-expression ends
    
    def __len__(self) -> int:
        return len(self.video_sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, dict]:
        """
        Get video sequence and label.
        
        Returns:
            Tuple of (video_frames, emotion_label, metadata)
        """
        raise NotImplementedError("Subclasses must implement __getitem__")
    
    def _load_frame_from_path(self, frame_path: Path) -> np.ndarray:
        """Load a single frame."""
        image = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load frame: {frame_path}")
        # Convert to RGB for EfficientNet
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    @staticmethod
    def get_emotion_name(idx: int) -> str:
        """Get emotion name from index."""
        return MicroExpressionDataset.MICRO_EMOTIONS[idx]


class CASMEIIDataset(MicroExpressionDataset):
    """
    CASME II (Chinese Affective Micro-Expression Database II) dataset.
    
    ~255 micro-expressions from 26 subjects.
    Capture spontaneous and genuine micro-expressions in conversations.
    
    Download from: http://fu.psych.ac.cn/CASME/casme2-en.html
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        use_apex_frame: bool = True,
        num_frames: int = 16,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize CASME II dataset.
        
        Args:
            data_dir: Path to dataset root directory
            split: "train" or "test"
            use_apex_frame: If True, sample around apex frame; if False, use full sequence
            num_frames: Number of frames to sample from sequence
            transform: Image transforms to apply
        """
        super().__init__(data_dir, transform)
        
        self.split = split
        self.use_apex_frame = use_apex_frame
        self.num_frames = num_frames
        
        self._load_casme_ii_dataset()
    
    def _load_casme_ii_dataset(self):
        """Load CASME II dataset structure."""
        # Expected structure: data_dir/splitname/subject*/video*
        split_dir = self.data_dir / self.split if self.split else self.data_dir
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {split_dir}")
        
        # Scan for video directories
        for subject_dir in split_dir.glob("*/"):
            if not subject_dir.is_dir():
                continue
            
            for video_dir in subject_dir.glob("*/"):
                if not video_dir.is_dir():
                    continue
                
                # Try to find emotion label from metadata file
                emotion_label = self._get_emotion_label(video_dir)
                if emotion_label is None:
                    continue
                
                # Get frame sequence
                frames = sorted(video_dir.glob("*.jpg"))
                if len(frames) < 3:  # Need at least 3 frames
                    continue
                
                self.video_sequences.append(frames)
                self.labels.append(emotion_label)
                
                # Store onset, apex, offset frame indices (if available)
                onset_idx = 0
                apex_idx = len(frames) // 2  # Approximate
                offset_idx = len(frames) - 1
                
                self.onset_frames.append(onset_idx)
                self.apex_frames.append(apex_idx)
                self.offset_frames.append(offset_idx)
    
    def _get_emotion_label(self, video_dir: Path) -> Optional[int]:
        """Extract emotion label from video directory or metadata."""
        # Try reading from annotation file
        anno_file = video_dir / "annotation.txt"
        if anno_file.exists():
            with open(anno_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if "Emotion" in line or "emotion" in line:
                        # Parse emotion from annotation
                        parts = line.split(":")
                        if len(parts) > 1:
                            emotion_str = parts[1].strip().lower()
                            return self.MICRO_EMOTION_TO_IDX.get(emotion_str)
        
        # Fallback: try to infer from directory name
        dir_name = video_dir.name.lower()
        for emotion_name, emotion_idx in self.MICRO_EMOTION_TO_IDX.items():
            if emotion_name in dir_name:
                return emotion_idx
        
        return None
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, dict]:
        """Get video sequence and label."""
        frames_paths = self.video_sequences[idx]
        emotion_label = self.labels[idx]
        
        onset_idx = self.onset_frames[idx]
        apex_idx = self.apex_frames[idx]
        offset_idx = self.offset_frames[idx]
        
        # Determine which frames to use
        if self.use_apex_frame:
            # Focus on onset -> apex -> offset
            total_frames = offset_idx - onset_idx + 1
            if total_frames < self.num_frames:
                # Repeat frames to reach desired length
                selected_indices = np.linspace(
                    onset_idx, offset_idx, self.num_frames, dtype=int
                )
            else:
                selected_indices = np.linspace(
                    onset_idx, offset_idx, self.num_frames, dtype=int
                )
        else:
            # Sample uniformly from entire sequence
            selected_indices = np.linspace(
                0, len(frames_paths) - 1, self.num_frames, dtype=int
            )
        
        # Load frames
        frames = []
        for frame_idx in selected_indices:
            frame = self._load_frame_from_path(frames_paths[frame_idx])
            if self.transform:
                frame = self.transform(frame)
            else:
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frames.append(frame)
        
        video_tensor = torch.stack(frames, dim=0)  # (T, 3, H, W)
        
        metadata = {
            "onset_frame": onset_idx,
            "apex_frame": apex_idx,
            "offset_frame": offset_idx,
            "num_frames_available": len(frames_paths),
        }
        
        return video_tensor, emotion_label, metadata


class SAMMDataset(MicroExpressionDataset):
    """
    SAMM (Spontaneous Micro-Facial Movement Dataset).
    
    ~159 micro-expressions from 32 participants.
    High-speed cameras (200 fps) capturing spontaneous micro-expressions.
    
    Download from: http://www2.docm.mmu.ac.uk/SAMM/
    """
    
    def __init__(
        self,
        data_dir: str,
        use_apex_frame: bool = True,
        num_frames: int = 16,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize SAMM dataset.
        
        Args:
            data_dir: Path to dataset root directory
            use_apex_frame: If True, sample around apex frame
            num_frames: Number of frames to sample
            transform: Image transforms to apply
        """
        super().__init__(data_dir, transform)
        
        self.use_apex_frame = use_apex_frame
        self.num_frames = num_frames
        
        self._load_samm_dataset()
    
    def _load_samm_dataset(self):
        """Load SAMM dataset structure."""
        # Expected structure: data_dir/*/
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")
        
        # SAMM has format: subject_id/video_id/frames
        for subject_dir in self.data_dir.glob("*/"):
            if not subject_dir.is_dir():
                continue
            
            for video_dir in subject_dir.glob("*/"):
                if not video_dir.is_dir():
                    continue
                
                # Get emotion from video directory name or file
                emotion_label = self._get_emotion_label(video_dir)
                if emotion_label is None:
                    continue
                
                # Get all frame files
                frames = sorted(video_dir.glob("*.jpg"))
                if len(frames) < 3:
                    continue
                
                self.video_sequences.append(frames)
                self.labels.append(emotion_label)
                
                # Get temporal information from filename if available
                onset_idx, apex_idx, offset_idx = self._get_temporal_info(frames)
                self.onset_frames.append(onset_idx)
                self.apex_frames.append(apex_idx)
                self.offset_frames.append(offset_idx)
    
    def _get_emotion_label(self, video_dir: Path) -> Optional[int]:
        """Extract emotion label from video directory or metadata."""
        # Try reading SAMM's annotation format
        anno_file = video_dir / "annotation.txt"
        if anno_file.exists():
            with open(anno_file, "r") as f:
                content = f.read().lower()
                for emotion_name, emotion_idx in self.MICRO_EMOTION_TO_IDX.items():
                    if emotion_name in content:
                        return emotion_idx
        
        # Fallback: infer from directory name
        dir_name = video_dir.name.lower()
        for emotion_name, emotion_idx in self.MICRO_EMOTION_TO_IDX.items():
            if emotion_name in dir_name:
                return emotion_idx
        
        return None
    
    def _get_temporal_info(
        self, frames: list
    ) -> Tuple[int, int, int]:
        """Extract onset, apex, offset frame indices."""
        # For SAMM, frames may have naming convention like 01_0001.jpg
        # Try to parse temporal information
        total_frames = len(frames)
        onset = 0
        apex = total_frames // 2
        offset = total_frames - 1
        
        return onset, apex, offset
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, dict]:
        """Get video sequence and label."""
        frames_paths = self.video_sequences[idx]
        emotion_label = self.labels[idx]
        
        onset_idx = self.onset_frames[idx]
        apex_idx = self.apex_frames[idx]
        offset_idx = self.offset_frames[idx]
        
        # Sample frames
        if self.use_apex_frame:
            selected_indices = np.linspace(
                onset_idx, offset_idx, self.num_frames, dtype=int
            )
        else:
            selected_indices = np.linspace(
                0, len(frames_paths) - 1, self.num_frames, dtype=int
            )
        
        # Load and preprocess frames
        frames = []
        for frame_idx in selected_indices:
            frame = self._load_frame_from_path(frames_paths[frame_idx])
            if self.transform:
                frame = self.transform(frame)
            else:
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frames.append(frame)
        
        video_tensor = torch.stack(frames, dim=0)  # (T, 3, H, W)
        
        metadata = {
            "onset_frame": onset_idx,
            "apex_frame": apex_idx,
            "offset_frame": offset_idx,
            "num_frames_available": len(frames_paths),
        }
        
        return video_tensor, emotion_label, metadata
