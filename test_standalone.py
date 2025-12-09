#!/usr/bin/env python3
"""
Test script for standalone MOT tracker
"""

import cv2
import numpy as np

# Test import
try:
    from mot_standalone.mot_wrapper import MOT
    print("✓ Successfully imported MOT from mot_standalone")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    exit(1)

# Test initialization
try:
    tracker = MOT(
        model_path='mot_standalone/LightTrackM.pth',
        device='cpu'  # Use CPU for testing
    )
    print("✓ Successfully initialized tracker")
except Exception as e:
    print(f"✗ Failed to initialize tracker: {e}")
    exit(1)

# Create a dummy frame
frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
print("✓ Created test frame")

# Test adding an object
try:
    obj_id = tracker.add_box(frame, [100, 100, 200, 200])
    print(f"✓ Added object with ID: {obj_id}")
except Exception as e:
    print(f"✗ Failed to add object: {e}")
    exit(1)

# Test tracking
try:
    results = tracker.track(frame)
    print(f"✓ Tracking results: {results}")
except Exception as e:
    print(f"✗ Failed to track: {e}")
    exit(1)

# Test removing object
try:
    success = tracker.remove_box(obj_id)
    print(f"✓ Removed object: {success}")
except Exception as e:
    print(f"✗ Failed to remove object: {e}")
    exit(1)

print("\n✓ All tests passed! Standalone tracker is working correctly.")
