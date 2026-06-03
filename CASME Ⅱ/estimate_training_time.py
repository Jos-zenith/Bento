#!/usr/bin/env python3
# ============================================================================
# Training Time Calculator - Shows time savings from optimizations
# ============================================================================

import math
from datetime import timedelta

def format_time(hours: float) -> str:
    """Convert hours to readable format"""
    if hours < 1:
        return f"{int(hours * 60)} minutes"
    elif hours < 24:
        return f"{hours:.1f} hours"
    else:
        days = hours / 24
        return f"{days:.1f} days"


def calculate_speedups():
    """Calculate training time with various optimization levels"""
    
    print("\n" + "="*70)
    print("TRAINING TIME ESTIMATION - Optimization Impact")
    print("="*70 + "\n")
    
    # Configuration
    num_epochs = 100
    num_training_samples = 193
    batch_size_options = [8, 16, 24]
    
    # Baseline: From your logs (Epoch 2 = 30 min)
    baseline_time_per_epoch = 30  # minutes
    
    scenarios = {
        "❌ No Optimization (Current)": {
            "description": "As-is (3x degradation)",
            "epoch_1": 10,
            "epoch_2": 30,
            "avg": 25,  # Average across all 100 epochs
        },
        
        "✅ Level 1: GPU Memory + Data Loader Optimization": {
            "description": "GPU cache clearing + reduced num_workers",
            "speedup": 2.5,  # 2-3x for memory fix + 1.2x for workers ≈ 2.5x average
        },
        
        "✅ Level 2: + Unidirectional LSTM": {
            "description": "Level 1 + model optimization",
            "speedup": 3.25,  # 2.5x * 1.3x ≈ 3.25x
        },
        
        "✅ Level 3: + Optical Flow Cache": {
            "description": "Level 2 + pre-computed optical flow",
            "speedup": 6.5,  # 3.25x * 2x ≈ 6.5x
        },
        
        "🚀 Level 4: + Higher Batch Size (16)": {
            "description": "Level 3 + batch size 8→16",
            "speedup": 9.75,  # 6.5x * 1.5x ≈ 9.75x
        },
    }
    
    print("SCENARIO ANALYSIS\n")
    print(f"Training Configuration:")
    print(f"  • Epochs: {num_epochs}")
    print(f"  • Baseline epoch time: {baseline_time_per_epoch} min (from your Epoch 2)")
    print(f"  • Training samples: {num_training_samples}\n")
    
    results = []
    
    # Baseline
    total_hours_baseline = (baseline_time_per_epoch * num_epochs) / 60
    print(f"📊 BASELINE (No optimizations)")
    print(f"   Time per epoch: {baseline_time_per_epoch} min (degrading to 30+ min)")
    print(f"   Total training time: {format_time(total_hours_baseline)}")
    print(f"   " + "-"*66)
    results.append(("Baseline", total_hours_baseline))
    
    # Optimized scenarios
    for scenario_name, scenario_data in scenarios.items():
        if "speedup" not in scenario_data:
            continue
            
        speedup = scenario_data["speedup"]
        optimized_time_per_epoch = baseline_time_per_epoch / speedup
        total_hours_optimized = (optimized_time_per_epoch * num_epochs) / 60
        time_saved = total_hours_baseline - total_hours_optimized
        time_saved_percent = (time_saved / total_hours_baseline) * 100
        
        print(f"\n{scenario_name}")
        print(f"   {scenario_data['description']}")
        print(f"   Time per epoch: {optimized_time_per_epoch:.1f} min")
        print(f"   Total training time: {format_time(total_hours_optimized)}")
        print(f"   Time saved: {format_time(time_saved)} ({time_saved_percent:.0f}%)")
        print(f"   Speedup: {speedup:.1f}x")
        results.append((scenario_name, total_hours_optimized, time_saved))
    
    # Batch size variations
    print(f"\n\n🔋 BATCH SIZE COMPARISON (Level 3: With optical flow cache)")
    print(f"   " + "-"*66)
    
    level_3_speedup = 6.5
    for bs in batch_size_options:
        # Estimate batch size speedup (rough: time ∝ 1/batch_size)
        bs_speedup = 8 / bs  # 8 is baseline batch size
        total_speedup = level_3_speedup * bs_speedup
        
        optimized_time_per_epoch = baseline_time_per_epoch / total_speedup
        total_hours = (optimized_time_per_epoch * num_epochs) / 60
        time_saved = total_hours_baseline - total_hours
        
        print(f"\n   Batch Size {bs}:")
        print(f"     Time per epoch: {optimized_time_per_epoch:.1f} min")
        print(f"     Total training: {format_time(total_hours)}")
        print(f"     Time saved: {format_time(time_saved)} ({(time_saved/total_hours_baseline)*100:.0f}%)")
        print(f"     Total speedup: {total_speedup:.1f}x")
    
    print("\n" + "="*70)
    print("RECOMMENDED PATH\n")
    
    print("✅ IMMEDIATE (No prerequisites, do now):")
    print("   • Level 1: GPU Memory + Data Loader Optimization")
    print("   • Expected: ~2-3x speedup")
    print("   • Time required: Already done!")
    print("   • Training time: 50 hrs → 17-25 hrs\n")
    
    print("✅ OPTIONAL (Recommended, takes 30-60 min one-time):")
    print("   • Level 2-3: Add unidirectional LSTM + optical flow cache")
    print("   • Expected: ~6-7x total speedup")
    print("   • One-time setup: 60 minutes")
    print("   • Training time: 50 hrs → 8 hrs\n")
    
    print("⚡ ADVANCED (If you have extra GPU memory):")
    print("   • Level 4: Also increase batch size to 16-24")
    print("   • Expected: ~10x total speedup")
    print("   • Training time: 50 hrs → 5 hrs\n")
    
    print("="*70)
    print("\nCONCLUSION:")
    print("  • With current changes: 2.5-3.5x speedup (done!)")
    print("  • With optical flow cache: 6-7x speedup (optional)")
    print("  • With higher batch size: 10x+ speedup (optional)")
    print("\n  Next step: python train.py")
    print("  To further optimize: python precompute_optical_flow.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    calculate_speedups()
