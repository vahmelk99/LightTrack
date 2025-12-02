#!/usr/bin/env python
"""
Simple test to verify all imports work correctly for the multi-object tracker
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports for multi-object tracker...")
    
    try:
        import _init_paths
        print("✓ _init_paths")
    except ImportError as e:
        print(f"✗ _init_paths: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ cv2 (OpenCV version: {cv2.__version__})")
    except ImportError as e:
        print(f"✗ cv2: {e}")
        return False
    
    try:
        import torch
        print(f"✓ torch (PyTorch version: {torch.__version__})")
    except ImportError as e:
        print(f"✗ torch: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ numpy (version: {np.__version__})")
    except ImportError as e:
        print(f"✗ numpy: {e}")
        return False
    
    try:
        from easydict import EasyDict as edict
        print("✓ easydict")
    except ImportError as e:
        print(f"✗ easydict: {e}")
        return False
    
    try:
        import lib.models.models as models
        print("✓ lib.models.models")
    except ImportError as e:
        print(f"✗ lib.models.models: {e}")
        return False
    
    try:
        from lib.utils.utils import load_pretrain, cxy_wh_2_rect
        print("✓ lib.utils.utils")
    except ImportError as e:
        print(f"✗ lib.utils.utils: {e}")
        return False
    
    try:
        from lib.tracker.lighttrack_multi import LighttrackMulti
        print("✓ lib.tracker.lighttrack_multi")
    except ImportError as e:
        print(f"✗ lib.tracker.lighttrack_multi: {e}")
        return False
    
    print("\n✓ All imports successful!")
    return True


def test_model_file():
    """Test that the model file exists"""
    print("\nChecking model file...")
    model_path = "../snapshot/LightTrackM/LightTrackM.pth"
    
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✓ Model file found: {model_path} ({size_mb:.2f} MB)")
        return True
    else:
        print(f"✗ Model file not found: {model_path}")
        print("  Download from: https://drive.google.com/drive/folders/1HXhdJO3yhQYw3O7nGUOXHu2S20Bs8CfI")
        return False


def main():
    print("="*60)
    print("Multi-Object Tracker - Import Test")
    print("="*60)
    print()
    
    imports_ok = test_imports()
    model_ok = test_model_file()
    
    print()
    print("="*60)
    if imports_ok and model_ok:
        print("✓ All tests passed! Ready to run multi-object tracker.")
        print()
        print("Run the tracker with:")
        print("  ./run_multi_tracker.sh")
        print("  or")
        print("  python multi_object_camera_tracker_cpu.py --resume ../snapshot/LightTrackM/LightTrackM.pth")
        return 0
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
