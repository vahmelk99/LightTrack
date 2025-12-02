#!/usr/bin/env python
"""
Test that the model can be loaded in CPU mode without CUDA errors
"""

import _init_paths
import sys
import torch
import numpy as np
from easydict import EasyDict as edict
import lib.models.models as models
from lib.utils.utils import load_pretrain

def test_model_load_cpu():
    print("="*60)
    print("Testing Model Load in CPU Mode")
    print("="*60)
    print()
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()
    
    try:
        print("Step 1: Creating model architecture...")
        # Use the LightTrackM architecture path
        siam_net = models.__dict__['LightTrackM_Subnet'](
            path_name='back_04502514044521042540+cls_211000022+reg_100000111_ops_32', 
            stride=16
        )
        print("✓ Model architecture created")
        print()
        
        print("Step 2: Loading pretrained weights...")
        siam_net = load_pretrain(siam_net, '../snapshot/LightTrackM/LightTrackM.pth')
        print("✓ Pretrained weights loaded")
        print()
        
        print("Step 3: Setting model to eval mode...")
        siam_net.eval()
        print("✓ Model set to eval mode")
        print()
        
        print("Step 4: Ensuring model is on CPU...")
        siam_net = siam_net.cpu()
        print("✓ Model moved to CPU")
        print()
        
        print("Step 5: Checking model parameters device...")
        first_param = next(siam_net.parameters())
        print(f"✓ First parameter device: {first_param.device}")
        print()
        
        if first_param.device.type == 'cpu':
            print("="*60)
            print("✓ SUCCESS: Model loaded on CPU without CUDA errors!")
            print("="*60)
            return True
        else:
            print("="*60)
            print(f"✗ FAILED: Model is on {first_param.device}, expected CPU")
            print("="*60)
            return False
            
    except RuntimeError as e:
        if "CUDA" in str(e) or "cuda" in str(e):
            print("="*60)
            print(f"✗ CUDA ERROR: {e}")
            print("="*60)
            print()
            print("This means the model is trying to use CUDA even in CPU mode.")
            print("Check for hardcoded .cuda() calls in the model code.")
            return False
        else:
            print(f"✗ Runtime error: {e}")
            raise
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        raise

if __name__ == '__main__':
    success = test_model_load_cpu()
    sys.exit(0 if success else 1)
