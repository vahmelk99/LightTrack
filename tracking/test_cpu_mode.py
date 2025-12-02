#!/usr/bin/env python
"""
Test that CPU mode works without accessing CUDA
"""

import _init_paths
import torch
import numpy as np
from easydict import EasyDict as edict
import lib.models.models as models
from lib.utils.utils import load_pretrain

def test_cpu_mode():
    print("Testing CPU mode (should not access CUDA)...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()
    
    # Setup model info
    siam_info = edict()
    siam_info.arch = 'LightTrackM_Subnet'
    siam_info.dataset = 'VOT2019'
    siam_info.epoch_test = False
    siam_info.stride = 16
    
    print("Loading model...")
    siam_net = models.__dict__[siam_info.arch]('LightTrackM', stride=siam_info.stride)
    
    print("Loading pretrained weights...")
    siam_net = load_pretrain(siam_net, '../snapshot/LightTrackM/LightTrackM.pth')
    siam_net.eval()
    
    print("Setting model to CPU...")
    siam_net = siam_net.cpu()
    
    print("Creating dummy input...")
    # Create a small dummy input
    dummy_template = torch.randn(1, 3, 128, 128)
    dummy_search = torch.randn(1, 3, 256, 256)
    
    print("Running forward pass on CPU...")
    with torch.no_grad():
        try:
            # This should work without accessing CUDA
            siam_net.template(dummy_template)
            output = siam_net.track(dummy_search)
            print("✓ Forward pass successful on CPU!")
            print(f"  Output keys: {output.keys()}")
            return True
        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"✗ CUDA error on CPU mode: {e}")
                return False
            else:
                raise

if __name__ == '__main__':
    success = test_cpu_mode()
    if success:
        print("\n✓ CPU mode test passed!")
        exit(0)
    else:
        print("\n✗ CPU mode test failed!")
        exit(1)
