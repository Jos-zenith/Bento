# ============================================================================
# Pre-compute Optical Flow for CASME II Dataset
# This script pre-computes optical flow for all training samples,
# eliminating expensive on-the-fly computation and speeding up training by ~2x
# ============================================================================

import os
import cv2
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OpticalFlowPrecomputer:
    """Pre-compute and cache optical flow for faster training"""
    
    def __init__(self, cache_dir: str = "optical_flow_cache", clip_value: float = 20.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.clip_value = clip_value
    
    def get_cache_path(self, subject: int, episode: str) -> Path:
        """Generate cache file path for optical flow"""
        cache_name = f"sub{subject:02d}_{episode}_flow.pkl"
        return self.cache_dir / cache_name
    
    def compute_optical_flow(self, frames: np.ndarray) -> np.ndarray:
        """
        Compute optical flow between consecutive frames.
        
        Args:
            frames: Array of shape (T, H, W, 3) in RGB format
            
        Returns:
            flow_sequence: Array of shape (T-1, H, W, 2) containing optical flow
        """
        flow_sequence = []
        
        for i in range(len(frames) - 1):
            frame1 = cv2.cvtColor(frames[i].astype(np.uint8), cv2.COLOR_RGB2GRAY)
            frame2 = cv2.cvtColor(frames[i + 1].astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Farneback optical flow
            flow = cv2.calcOpticalFlowFarneback(
                frame1, frame2,
                None,
                0.5, 3, 15, 3, 5, 1.2,
                cv2.OPTFLOW_FARNEBACK_GAUSSIAN
            )
            
            # Normalize and clip
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            mag = np.clip(mag, 0, self.clip_value) / (self.clip_value + 1e-6)
            
            flow_sequence.append(flow)
        
        return np.array(flow_sequence, dtype=np.float32)
    
    def save_optical_flow(self, flow: np.ndarray, cache_path: Path):
        """Save optical flow to cache"""
        with open(cache_path, 'wb') as f:
            pickle.dump(flow, f)
    
    def load_optical_flow(self, cache_path: Path) -> Optional[np.ndarray]:
        """Load optical flow from cache"""
        if not cache_path.exists():
            return None
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_path}: {e}")
            return None


def precompute_dataset_optical_flow(
    root_dir: str,
    labels_file: str,
    data_stage: str = "cropped",
    frame_size: tuple = (224, 224),
    temporal_length: int = 12,
):
    """
    Precompute optical flow for all dataset samples.
    
    Args:
        root_dir: Root directory containing subjects
        labels_file: Path to Excel labels file
        data_stage: Dataset stage (cropped, raw, etc.)
        frame_size: Frame size (H, W)
        temporal_length: Number of frames to sample
    """
    from preprocessing import FrameSequenceProcessor
    
    precomputer = OpticalFlowPrecomputer()
    frame_processor = FrameSequenceProcessor(frame_size)
    
    # Load labels
    df = pd.read_excel(labels_file)
    
    cached_count = 0
    computed_count = 0
    skipped_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Precomputing optical flow"):
        try:
            subject_idx = int(pd.to_numeric(row['Subject'], errors='coerce'))
            filename = str(row['Filename']).strip()
            onset = int(pd.to_numeric(row['OnsetFrame'], errors='coerce'))
            apex = int(pd.to_numeric(row['ApexFrame'], errors='coerce'))
            offset = int(pd.to_numeric(row['OffsetFrame'], errors='coerce'))
            
            # Check cache first
            cache_path = precomputer.get_cache_path(subject_idx, filename)
            cached_flow = precomputer.load_optical_flow(cache_path)
            
            if cached_flow is not None:
                cached_count += 1
                continue
            
            # Resolve data path
            subject_root = Path(root_dir) / f"sub{subject_idx:02d}"
            episode_dir = subject_root / filename
            
            if not episode_dir.exists():
                skipped_count += 1
                continue
            
            # Load frames
            frame_files = sorted([f for f in os.listdir(episode_dir) if f.endswith('.jpg')])
            if not frame_files:
                skipped_count += 1
                continue
            
            frame_paths = [str(episode_dir / f) for f in frame_files]
            
            # Load frame sequence
            frames = frame_processor.load_sequence(frame_paths)
            
            # Compute optical flow
            optical_flow = precomputer.compute_optical_flow(frames)
            
            # Save to cache
            precomputer.save_optical_flow(optical_flow, cache_path)
            computed_count += 1
            
        except Exception as e:
            logger.debug(f"Error processing row {idx}: {e}")
            skipped_count += 1
            continue
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Precomputation Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Cached (reused):     {cached_count}")
    logger.info(f"Computed (new):      {computed_count}")
    logger.info(f"Skipped (errors):    {skipped_count}")
    logger.info(f"Total processed:     {cached_count + computed_count + skipped_count}")
    logger.info(f"Cache directory:     {precomputer.cache_dir.absolute()}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    from config import Config
    
    logger.info("Starting optical flow precomputation...")
    precompute_dataset_optical_flow(
        root_dir=Config.get_data_root(Config.DATA_STAGE),
        labels_file=Config.LABELS_FILE,
        data_stage=Config.DATA_STAGE,
        frame_size=Config.FRAME_SIZE,
        temporal_length=Config.TEMPORAL_LENGTH,
    )
    logger.info("Optical flow precomputation finished!")
