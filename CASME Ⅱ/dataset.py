# ============================================================================
# CASME II Dataset Loader
# ============================================================================

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional, cast
from pathlib import Path
import warnings

from preprocessing import (
    OpticalFlowProcessor, 
    FrameSequenceProcessor, 
    AugmentationPipeline
)


class CASME2Dataset(Dataset):
    """PyTorch Dataset for CASME II Micro-Expression Classification"""

    _dataset_cache: Dict[Tuple[str, str, str], List[Dict]] = {}
    
    def __init__(
        self,
        root_dir: str,
        labels_file: str,
        mode: str = "train",
        data_stage: str = "cropped",
        frame_size: Tuple[int, int] = (224, 224),
        temporal_length: int = 12,
        use_optical_flow: bool = True,
        augmentation_params: Optional[dict] = None,
        use_augmentation: bool = False,
        img_mean: Optional[List[float]] = None,
        img_std: Optional[List[float]] = None,
    ):
        """
        Args:
            root_dir: Path to cropped faces directory
            labels_file: Path to Excel file with labels
            mode: 'train', 'val', or 'test'
            data_stage: Dataset stage to load from: raw, selected, compressed_video, cropped
            frame_size: Target frame size (H, W)
            temporal_length: Number of frames to sample
            use_optical_flow: Include optical flow as additional channel
            augmentation_params: Dict of augmentation parameters
            use_augmentation: Whether to apply augmentation
            img_mean: Normalization mean
            img_std: Normalization std
        """
        self.root_dir = root_dir
        self.mode = mode
        self.data_stage = data_stage.lower()
        self.frame_size = frame_size
        self.temporal_length = temporal_length
        self.use_optical_flow = use_optical_flow
        self.use_augmentation = use_augmentation and mode == "train"
        self.img_mean = img_mean or [0.485, 0.456, 0.406]
        self.img_std = img_std or [0.229, 0.224, 0.225]
        
        # Initialize processors
        self.frame_processor = FrameSequenceProcessor(frame_size)
        self.flow_processor = OpticalFlowProcessor(method="farneback", clip_value=20.0)
        self.augmentor = AugmentationPipeline(augmentation_params) if use_augmentation else None
        
        # Load labels and build dataset
        cache_key = (str(Path(labels_file)), self.data_stage, str(Path(root_dir)))
        if cache_key in self._dataset_cache:
            self.samples = [sample.copy() for sample in self._dataset_cache[cache_key]]
        else:
            self.samples = self._load_dataset(labels_file)
            self._dataset_cache[cache_key] = [sample.copy() for sample in self.samples]
        self._split_dataset()

    def _resolve_episode_source(self, subject_idx: int, filename: str) -> Optional[Tuple[Path, str]]:
        """Resolve the episode source path for the configured dataset stage."""
        subject_root = Path(self.root_dir) / f"sub{subject_idx:02d}"
        directory_candidate = subject_root / filename
        video_candidates = [
            directory_candidate.with_suffix(".avi"),
            directory_candidate.with_suffix(".mp4"),
            directory_candidate.with_suffix(".mov"),
            directory_candidate.with_suffix(".mkv"),
        ]

        if self.data_stage == "compressed_video":
            normalized_name = filename.replace("_", "")
            normalized_candidates = []
            if normalized_name:
                normalized_candidates.append(subject_root / f"{normalized_name}.avi")
                if normalized_name[-1].isalpha():
                    normalized_candidates.append(subject_root / f"{normalized_name[:-1]}_{normalized_name[-1]}.avi")
            underscore_tail_candidate = subject_root / f"{filename}_f.avi" if not filename.endswith("f") else subject_root / f"{filename[:-1]}_{filename[-1]}.avi"
            candidates = [*video_candidates, underscore_tail_candidate, *normalized_candidates, directory_candidate]
        else:
            candidates = [directory_candidate, *video_candidates]

        for candidate in candidates:
            if candidate.exists():
                if candidate.is_dir():
                    return candidate, "frames"
                if candidate.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}:
                    return candidate, "video"

        return None
    
    def _load_dataset(self, labels_file: str) -> List[Dict]:
        """
        Load dataset from Excel labels file.
        Expected columns: Subject, Filename, OnsetFrame, ApexFrame, OffsetFrame, Estimated Emotion
        """
        df = pd.read_excel(labels_file)
        
        samples = []
        for idx, row in df.iterrows():
            try:
                # Get subject number (already an integer in Excel)
                subject_value = pd.to_numeric(row['Subject'], errors='coerce')
                if pd.isna(subject_value):
                    warnings.warn(f"Invalid Subject at row {idx}, skipping")
                    continue
                subject_idx = int(subject_value)
                
                # Get filename/episode
                filename = str(row['Filename']).strip()

                def _parse_frame_value(value, field_name: str) -> Optional[int]:
                    parsed_value = pd.to_numeric(value, errors='coerce')
                    if pd.isna(parsed_value):
                        warnings.warn(f"Invalid {field_name} at row {idx}, skipping")
                        return None
                    return int(parsed_value)
                
                # Get frame numbers - handle missing/invalid values
                onset_frame = _parse_frame_value(row['OnsetFrame'], 'OnsetFrame')
                if onset_frame is None:
                    continue
                
                apex_frame = _parse_frame_value(row['ApexFrame'], 'ApexFrame')
                if apex_frame is None:
                    continue
                
                offset_frame = _parse_frame_value(row['OffsetFrame'], 'OffsetFrame')
                if offset_frame is None:
                    continue
                
                # Convert emotion label
                emotion_raw = str(row['Estimated Emotion']).strip().lower()
                if emotion_raw == 'happiness':
                    emotion = 'Happiness'
                elif emotion_raw == 'surprise':
                    emotion = 'Surprise'
                elif emotion_raw == 'disgust':
                    emotion = 'Disgust'
                elif emotion_raw == 'repression':
                    emotion = 'Repression'
                else:
                    emotion = 'Others'
                
                source = self._resolve_episode_source(subject_idx, filename)
                if source is None:
                    expected_root = Path(self.root_dir) / f"sub{subject_idx:02d}" / filename
                    warnings.warn(f"Episode source not found: {expected_root}")
                    continue

                source_path, source_type = source
                
                if source_type == "frames":
                    # List all frames
                    frames = sorted([f for f in os.listdir(source_path) if f.endswith('.jpg')])

                    if len(frames) < 3:
                        warnings.warn(f"Insufficient frames in {source_path}")
                        continue

                    frame_paths = [os.path.join(source_path, f) for f in frames]
                else:
                    frame_paths = []
                
                samples.append({
                    'subject_idx': subject_idx,
                    'episode': filename,
                    'emotion': emotion,
                    'source_type': source_type,
                    'source_path': str(source_path),
                    'frame_paths': frame_paths,
                    'onset_frame': onset_frame,
                    'apex_frame': apex_frame,
                    'offset_frame': offset_frame,
                })
            except Exception as e:
                warnings.warn(f"Error processing row {idx}: {e}")
                continue
        
        return samples
    
    def _split_dataset(self):
        """Split dataset into train/val/test"""
        # Group by subject to avoid subject leakage
        subject_ids = sorted(set(s['subject_idx'] for s in self.samples))
        
        n_subjects = len(subject_ids)
        n_train = int(n_subjects * 0.75)
        n_val = int(n_subjects * 0.1)
        
        train_subjects = set(subject_ids[:n_train])
        val_subjects = set(subject_ids[n_train:n_train + n_val])
        test_subjects = set(subject_ids[n_train + n_val:])
        
        if self.mode == "train":
            self.samples = [s for s in self.samples if s['subject_idx'] in train_subjects]
        elif self.mode == "val":
            self.samples = [s for s in self.samples if s['subject_idx'] in val_subjects]
        else:  # test
            self.samples = [s for s in self.samples if s['subject_idx'] in test_subjects]
    
    def _extract_temporal_region(self, frame_indices: np.ndarray) -> np.ndarray:
        """Extract frames from onset to apex region (most expressive)"""
        # Focus on onset-to-apex frames (most likely to show micro-expression)
        if len(frame_indices) > 0:
            return frame_indices[:len(frame_indices) // 2]  # First half
        return frame_indices
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            frames: Tensor of shape (T, C, H, W) where C=3 (RGB) or C=4 (RGB+Flow)
            label: Emotion class index
        """
        sample = self.samples[idx]
        frame_paths = sample['frame_paths']
        source_type = sample.get('source_type', 'frames')
        source_path = sample.get('source_path')
        emotion = sample['emotion']
        
        # Load frame sequence
        try:
            if source_type == 'video' and source_path:
                frames = self.frame_processor.load_video_sequence(source_path)
            else:
                frames = self.frame_processor.load_sequence(frame_paths)
        except Exception as e:
            warnings.warn(f"Error loading frames: {e}")
            # Return dummy data
            frames = np.zeros((self.temporal_length, *self.frame_size, 3), dtype=np.float32)
        
        # Sample temporal frames
        frames = self.frame_processor.sample_frames(frames, self.temporal_length)
        
        # Apply augmentation
        if self.use_augmentation and self.augmentor is not None:
            frames = self.augmentor.apply(frames)
        
        # Normalize frames
        frames = self.frame_processor.normalize_frames(frames, self.img_mean, self.img_std)
        
        # Compute optical flow
        if self.use_optical_flow:
            flows = []
            for i in range(len(frames) - 1):
                flow = self.flow_processor.compute_flow(
                    (frames[i] * 255).astype(np.uint8),  # Convert back for flow computation
                    (frames[i + 1] * 255).astype(np.uint8)
                )
                flow_mag = self.flow_processor.normalize_flow(flow)
                # Ensure flow_mag has shape (H, W, 1)
                if len(flow_mag.shape) == 2:
                    flow_mag = np.expand_dims(flow_mag, axis=-1)
                flows.append(flow_mag)
            
            # Pad last frame
            if flows:
                last_flow = flows[-1]
            else:
                last_flow = np.zeros((*self.frame_size, 1), dtype=np.float32)
            flows.append(last_flow)
            flows = np.array(flows)  # Shape: (T, H, W, 1)
            
            # Stack RGB + Flow: (T, H, W, 4)
            frames = np.concatenate([frames, flows], axis=-1)
        
        # Convert to PyTorch format: (T, C, H, W)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        
        # Get label
        from config import Config
        label = Config.CLASS_MAPPING.get(emotion, 4)
        
        return frames, label


def create_dataloaders(
    root_dir: str,
    labels_file: str,
    batch_size: int = 8,
    num_workers: int = 4,
    data_stage: str = "cropped",
    frame_size: Tuple[int, int] = (224, 224),
    temporal_length: int = 12,
    augmentation_params: Optional[dict] = None,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders.
    
    Returns:
        train_loader, val_loader, test_loader
    """
    effective_pin_memory = pin_memory and torch.cuda.is_available()
    
    # Train set with augmentation
    train_dataset = CASME2Dataset(
        root_dir=root_dir,
        labels_file=labels_file,
        mode="train",
        data_stage=data_stage,
        frame_size=frame_size,
        temporal_length=temporal_length,
        use_optical_flow=True,
        augmentation_params=augmentation_params,
        use_augmentation=True,
    )
    
    # Val set without augmentation
    val_dataset = CASME2Dataset(
        root_dir=root_dir,
        labels_file=labels_file,
        mode="val",
        data_stage=data_stage,
        frame_size=frame_size,
        temporal_length=temporal_length,
        use_optical_flow=True,
        use_augmentation=False,
    )
    
    # Test set without augmentation
    test_dataset = CASME2Dataset(
        root_dir=root_dir,
        labels_file=labels_file,
        mode="test",
        data_stage=data_stage,
        frame_size=frame_size,
        temporal_length=temporal_length,
        use_optical_flow=True,
        use_augmentation=False,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=effective_pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=effective_pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=effective_pin_memory,
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick test
    from config import Config
    
    print("Loading dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=Config.CROPPED_DATA_PATH,
        labels_file=Config.LABELS_FILE,
        batch_size=Config.BATCH_SIZE,
        num_workers=2,
        data_stage=Config.DATA_STAGE,
        augmentation_params=Config.AUGMENTATION_PARAMS,
    )
    
    print(f"Train samples: {len(cast(CASME2Dataset, train_loader.dataset))}")
    print(f"Val samples: {len(cast(CASME2Dataset, val_loader.dataset))}")
    print(f"Test samples: {len(cast(CASME2Dataset, test_loader.dataset))}")
    
    # Inspect a batch
    frames, labels = next(iter(train_loader))
    print(f"\nBatch shape: {frames.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels: {torch.unique(labels).tolist()}")
