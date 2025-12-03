#!/usr/bin/env python3
"""
Simple test script to verify desktop capture is working
"""
import cv2
import numpy as np

try:
    import mss
    print("✓ mss library is installed")
except ImportError:
    print("✗ mss library is NOT installed")
    print("Install with: pip install mss")
    exit(1)

print("\nTesting desktop capture...")

try:
    with mss.mss() as sct:
        # Get monitor info
        monitors = sct.monitors
        print(f"\nFound {len(monitors)} monitors:")
        for i, monitor in enumerate(monitors):
            print(f"  Monitor {i}: {monitor}")
        
        # Capture primary monitor
        monitor = monitors[1]
        print(f"\nCapturing from primary monitor: {monitor['width']}x{monitor['height']}")
        
        # Grab a screenshot
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        
        print(f"Screenshot captured: shape={frame.shape}, dtype={frame.dtype}")
        
        # Convert BGRA to BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        print(f"Converted to BGR: shape={frame_bgr.shape}, dtype={frame_bgr.dtype}")
        
        # Display the frame
        cv2.namedWindow('Desktop Capture Test', cv2.WINDOW_NORMAL)
        cv2.imshow('Desktop Capture Test', frame_bgr)
        
        print("\n✓ Desktop capture is working!")
        print("Press any key to close the window...")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print("\n✓ Test completed successfully!")

except Exception as e:
    print(f"\n✗ Error during test: {e}")
    import traceback
    traceback.print_exc()
