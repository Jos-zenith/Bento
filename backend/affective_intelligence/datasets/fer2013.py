"""
FER2013 Dataset - Macro-expressions dataset.

7 emotions: neutral, happy, sad, angry, fearful, disgusted, surprised.
"""

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class FER2013Dataset(Dataset):
    """
    FER2013 (Facial Expression Recognition) dataset.
    
    Contains ~35,000 images of size 48x48 with 7 emotion labels.
    Commonly used for macro-expression recognition.
    
    Download from: https://www.kaggle.com/datasets/murngl/facial-expression-recognition-dataset
    """
    
    EMOTIONS = [
        "angry",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise",
    ]
    
    EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        """
        Initialize FER2013 dataset.
        
        Args:
            data_dir: Path to dataset directory containing train/test folders
            split: "train" or "test"
            transform: Image transforms to apply
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load image paths and labels."""
        split_dir = self.data_dir / self.split
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {split_dir}")
        
        # Assuming directory structure: data_dir/train/emotion/*.jpg
        for emotion_idx, emotion in enumerate(self.EMOTIONS):
            emotion_dir = split_dir / emotion
            if emotion_dir.exists():
                for img_file in emotion_dir.glob("*.jpg"):
                    self.image_paths.append(img_file)
                    self.labels.append(emotion_idx)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get image and label."""
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Read image
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Convert grayscale to RGB for EfficientNet
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, label
    
    @staticmethod
    def get_emotion_name(idx: int) -> str:
        """Get emotion name from index."""
        return FER2013Dataset.EMOTIONS[idx]
