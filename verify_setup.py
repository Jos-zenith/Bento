#!/usr/bin/env python3
"""
Verify Panjaayathu Emotion Recognition Module Setup

Run this script to verify:
1. Dependencies are installed
2. Module structure is correct
3. Models can be imported
4. API is working
"""

import sys
import subprocess
from pathlib import Path

def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def check_python_version():
    """Check Python version."""
    print_header("Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 11:
        print("✅ Python version OK")
        return True
    else:
        print("❌ Python 3.11+ required")
        return False

def check_dependencies():
    """Check required packages."""
    print_header("Dependencies")
    
    required = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'fastapi': 'FastAPI',
        'uvicorn': 'Uvicorn',
        'tqdm': 'tqdm',
    }
    
    all_ok = True
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - not installed")
            all_ok = False
    
    return all_ok

def check_module_structure():
    """Check module structure."""
    print_header("Module Structure")
    
    base = Path(__file__).parent / "backend" / "affective_intelligence"
    
    required_files = {
        "__init__.py": "Package init",
        "train.py": "Training script",
        "README.md": "Documentation",
        "models/__init__.py": "Models package",
        "models/emotion_net.py": "EmotionNet model",
        "datasets/__init__.py": "Datasets package",
        "datasets/fer2013.py": "FER2013 loader",
        "datasets/micro_expressions.py": "Micro-expression loaders",
        "datasets/transforms.py": "Data transforms",
        "losses/__init__.py": "Losses package",
        "losses/emotion_losses.py": "Custom losses",
        "inference/__init__.py": "Inference package",
        "inference/emotion_predictor.py": "Emotion predictor",
    }
    
    all_ok = True
    for file_path, description in required_files.items():
        full_path = base / file_path
        if full_path.exists():
            size = full_path.stat().st_size
            print(f"✅ {file_path} ({size:,} bytes)")
        else:
            print(f"❌ {file_path} - missing")
            all_ok = False
    
    return all_ok

def check_api_integration():
    """Check FastAPI integration."""
    print_header("API Integration")
    
    app_file = Path(__file__).parent / "backend" / "app" / "main.py"
    
    if app_file.exists():
        with open(app_file, 'r') as f:
            content = f.read()
            
        checks = {
            'from api.emotion import': 'Emotion API import',
            'emotion_router': 'Emotion router',
            'init_emotion_predictor': 'Predictor initialization',
        }
        
        all_ok = True
        for check_str, description in checks.items():
            if check_str in content:
                print(f"✅ {description}")
            else:
                print(f"❌ {description} - not found")
                all_ok = False
        
        return all_ok
    else:
        print(f"❌ {app_file} - not found")
        return False

def check_model_imports():
    """Test model imports."""
    print_header("Model Imports")
    
    try:
        from affective_intelligence.models import EmotionNet, EmotionNetConfig
        print(f"✅ EmotionNet imported successfully")
        
        config = EmotionNetConfig()
        model = EmotionNet(config)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model initialized ({total_params:,} parameters)")
        return True
    except Exception as e:
        print(f"❌ Model import failed: {e}")
        return False

def check_dataset_imports():
    """Test dataset imports."""
    print_header("Dataset Imports")
    
    try:
        from affective_intelligence.datasets import (
            FER2013Dataset,
            CASMEIIDataset,
            SAMMDataset,
            get_train_transforms,
            get_val_transforms,
        )
        print(f"✅ FER2013Dataset imported")
        print(f"✅ CASMEIIDataset imported")
        print(f"✅ SAMMDataset imported")
        print(f"✅ Data transforms imported")
        return True
    except Exception as e:
        print(f"❌ Dataset import failed: {e}")
        return False

def check_loss_imports():
    """Test loss function imports."""
    print_header("Loss Functions")
    
    try:
        from affective_intelligence.losses import (
            CombinedEmotionLoss,
            CenterLoss,
            FocalLoss,
        )
        print(f"✅ FocalLoss imported")
        print(f"✅ CenterLoss imported")
        print(f"✅ CombinedEmotionLoss imported")
        return True
    except Exception as e:
        print(f"❌ Loss import failed: {e}")
        return False

def check_inference_imports():
    """Test inference imports."""
    print_header("Inference")
    
    try:
        from affective_intelligence.inference import EmotionPredictor
        print(f"✅ EmotionPredictor imported")
        return True
    except Exception as e:
        print(f"❌ Inference import failed: {e}")
        return False

def check_documentation():
    """Check documentation files."""
    print_header("Documentation")
    
    docs = {
        "EMOTION_RECOGNITION_SETUP.md": "Setup guide",
        "MICRO_EXPRESSION_DETECTION_GUIDE.md": "Technical guide",
        "EMOTION_API_INTEGRATION.md": "API integration",
        "IMPLEMENTATION_SUMMARY.md": "Implementation summary",
        "backend/affective_intelligence/README.md": "Module README",
    }
    
    root = Path(__file__).parent
    all_ok = True
    
    for doc_path, description in docs.items():
        full_path = root / doc_path
        if full_path.exists():
            size = full_path.stat().st_size
            print(f"✅ {description} ({size:,} bytes)")
        else:
            print(f"❌ {description} - missing")
            all_ok = False
    
    return all_ok

def print_summary(checks):
    """Print summary."""
    print_header("Summary")
    
    total = len(checks)
    passed = sum(1 for c in checks if c)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ All checks passed! Module is ready to use.")
        print("\nNext steps:")
        print("1. Download dataset: https://www.kaggle.com/datasets/murngl/facial-expression-recognition-dataset")
        print("2. Train model: python affective_intelligence/train.py --fer2013_path ./data/FER2013")
        print("3. Start API: python -m uvicorn app.main:app --reload")
        return True
    else:
        print(f"\n⚠️  {total - passed} checks failed. Review errors above.")
        return False

def main():
    """Run all checks."""
    print("\n" + "="*60)
    print("  Panjaayathu Emotion Recognition Module - Verification")
    print("="*60)
    
    checks = [
        check_python_version(),
        check_dependencies(),
        check_module_structure(),
        check_api_integration(),
        check_model_imports(),
        check_dataset_imports(),
        check_loss_imports(),
        check_inference_imports(),
        check_documentation(),
    ]
    
    success = print_summary(checks)
    
    print("\n" + "="*60 + "\n")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
