# Standalone Multi-Object Tracker

This folder contains a standalone version of the multi-object tracker with minimal dependencies.

## Files

1. **LightTrackM.pth** - Pre-trained model weights
2. **mot_tracker.py** - Core tracking implementation with all utilities embedded
3. **mot_wrapper.py** - Model architecture and wrapper class (complete standalone version)

## Usage

```python
from mot_standalone.mot_wrapper import MOT

# Initialize tracker
tracker = MOT(
    model_path='mot_standalone/LightTrackM.pth',
    device='cuda'  # or 'cpu'
)

# Add objects to track
frame = cv2.imread('frame.jpg')
obj_id = tracker.add_box(frame, [x1, y1, x2, y2])

# Track in subsequent frames
results = tracker.track(frame)
# returns: [(id, [x1, y1, x2, y2]), ...]

# Remove object
tracker.remove_box(obj_id)
```

## Dependencies

- torch
- opencv-python (cv2)
- numpy
- yaml
- easydict
