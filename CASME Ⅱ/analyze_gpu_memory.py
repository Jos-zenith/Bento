#!/usr/bin/env python3
# ============================================================================
# GPU Memory Analyzer & Batch Size Optimizer
# Determines optimal batch size for your GPU
# ============================================================================

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_gpu_info():
    """Get detailed GPU information"""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Using CPU mode.")
        return None
    
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    
    info = {
        'name': torch.cuda.get_device_name(device),
        'total_memory_gb': props.total_memory / 1e9,
        'device_id': device.index,
        'compute_capability': f"{props.major}.{props.minor}",
    }
    
    logger.info(f"GPU Device: {info['name']}")
    logger.info(f"Total Memory: {info['total_memory_gb']:.2f} GB")
    logger.info(f"Compute Capability: {info['compute_capability']}")
    
    return info


def estimate_batch_memory(
    batch_size: int,
    temporal_length: int = 12,
    frame_size: tuple = (224, 224),
    input_channels: int = 7,  # RGB + Combined Optical Flow
    model_params: int = 7_200_000,  # ~7.2M after unidirectional LSTM
) -> float:
    """
    Estimate GPU memory usage for a batch.
    
    Returns:
        memory_gb: Estimated GPU memory in GB
    """
    # Input tensor size
    input_size = batch_size * temporal_length * input_channels * frame_size[0] * frame_size[1]
    input_memory = input_size * 4 / 1e9  # float32 = 4 bytes
    
    # Model weights (assume mixed precision doesn't help much here)
    model_memory = model_params * 4 / 1e9
    
    # Activations and gradients (roughly 2x model size during backprop)
    activation_memory = input_memory + model_memory * 2
    
    # Optimizer state (Adam has 2 states per param)
    optimizer_memory = model_params * 8 / 1e9
    
    # Safety margin (20%)
    total = (input_memory + model_memory + activation_memory + optimizer_memory) * 1.2
    
    return total


def recommend_batch_size(
    total_memory_gb: float,
    temporal_length: int = 12,
    frame_size: tuple = (224, 224),
    target_usage: float = 0.8,  # Use 80% of GPU memory
) -> int:
    """
    Recommend batch size based on available GPU memory.
    
    Args:
        total_memory_gb: Total GPU memory in GB
        temporal_length: Number of frames per sample
        frame_size: Input frame size (H, W)
        target_usage: Target GPU memory utilization (0.0-1.0)
    
    Returns:
        optimal_batch_size: Recommended batch size
    """
    max_memory_available = total_memory_gb * target_usage
    
    # Binary search for optimal batch size
    batch_sizes = [4, 8, 12, 16, 20, 24, 32, 48, 64, 96, 128]
    optimal_batch_size = 4
    
    for bs in batch_sizes:
        estimated_memory = estimate_batch_memory(
            batch_size=bs,
            temporal_length=temporal_length,
            frame_size=frame_size,
        )
        logger.info(f"Batch size {bs:3d}: {estimated_memory:.2f} GB")
        
        if estimated_memory <= max_memory_available:
            optimal_batch_size = bs
        else:
            break
    
    return optimal_batch_size


def run_memory_benchmark():
    """Run actual GPU memory benchmark by loading model and batch"""
    try:
        from config import Config
        from models import build_model
        
        logger.info("\n" + "="*60)
        logger.info("Running Actual Memory Benchmark")
        logger.info("="*60)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if device.type != "cuda":
            logger.warning("GPU not available. Skipping actual benchmark.")
            return
        
        # Build model
        model = build_model(
            model_type=Config.MODEL_TYPE,
            num_classes=Config.NUM_CLASSES,
            temporal_length=Config.TEMPORAL_LENGTH,
            input_channels=Config.MODEL_INPUT_CHANNELS,
            pretrained_backbone=False,  # Don't load pretrained weights
            freeze_backbone=False,
            device=device,
        )
        
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test different batch sizes
        torch.cuda.reset_peak_memory_stats()
        test_batch_sizes = [4, 8, 16, 24, 32]
        
        for bs in test_batch_sizes:
            torch.cuda.reset_peak_memory_stats()
            
            # Create dummy batch
            batch = torch.randn(
                bs,
                Config.TEMPORAL_LENGTH,
                Config.MODEL_INPUT_CHANNELS,
                Config.FRAME_SIZE[0],
                Config.FRAME_SIZE[1],
                device=device,
            )
            
            # Forward pass
            with torch.no_grad():
                logits = model(batch)
            
            # Measure peak memory
            peak_memory_mb = torch.cuda.max_memory_allocated() / 1e6
            peak_memory_gb = peak_memory_mb / 1024
            
            logger.info(f"Batch size {bs:2d}: {peak_memory_gb:6.2f} GB")
            
            if peak_memory_gb > 7.5:  # Safety threshold
                logger.warning(f"⚠️  Batch size {bs} exceeds safety threshold!")
                break
    
    except ImportError as e:
        logger.warning(f"Could not run benchmark: {e}")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")


def main():
    logger.info("\n" + "="*60)
    logger.info("GPU Memory & Batch Size Optimizer")
    logger.info("="*60 + "\n")
    
    # Get GPU info
    gpu_info = get_gpu_info()
    
    if gpu_info is None:
        return
    
    logger.info("\n" + "="*60)
    logger.info("Batch Size Recommendations")
    logger.info("="*60)
    
    optimal_bs = recommend_batch_size(
        total_memory_gb=gpu_info['total_memory_gb'],
        target_usage=0.85,  # Conservative 85% usage
    )
    
    logger.info(f"\n✅ Recommended batch size: {optimal_bs}")
    logger.info(f"   Expected speedup vs batch=8: {optimal_bs / 8:.1f}x\n")
    
    # Suggest configuration
    config_suggestions = [
        f"BATCH_SIZE = {optimal_bs}",
        f"NUM_WORKERS = 2",
        f"PIN_MEMORY = True",
        f"MODEL_TYPE = 'LSTM' (unidirectional)",
    ]
    
    logger.info("Suggested config.py settings:")
    for suggestion in config_suggestions:
        logger.info(f"  • {suggestion}")
    
    # Run actual benchmark if possible
    logger.info()
    run_memory_benchmark()
    
    logger.info("\n" + "="*60)
    logger.info("For more details, see OPTIMIZATION_GUIDE.md")
    logger.info("="*60)


if __name__ == "__main__":
    main()
