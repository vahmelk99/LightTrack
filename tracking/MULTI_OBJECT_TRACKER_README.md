# Multi-Object Camera Tracker

A real-time multi-object tracking system using LightTrack that allows you to select and track multiple objects from a camera feed or video file.

## Features

- **Interactive Object Selection**: Draw bounding boxes around objects you want to track
- **Multi-Object Tracking**: Track multiple objects simultaneously with unique IDs and colors
- **Real-time Performance**: Optimized for both CPU and GPU
- **Flexible Input**: Works with camera feeds or video files
- **User-Friendly Controls**: Simple keyboard controls for selection, tracking, and reset

## Quick Start

### Using the Shell Script (Recommended)

```bash
# Track objects from default camera (CPU)
./tracking/run_multi_tracker.sh

# Track objects using GPU
./tracking/run_multi_tracker.sh -d cuda

# Track objects from a video file
./tracking/run_multi_tracker.sh -v path/to/video.mp4

# Use a different camera device
./tracking/run_multi_tracker.sh -c 1
```

### Using Python Directly

```bash
# CPU version
python tracking/multi_object_camera_tracker_cpu.py \
    --arch LightTrackM_Subnet \
    --resume snapshot/LightTrackM/LightTrackM.pth \
    --path_name LightTrackM \
    --device cpu

# GPU version
python tracking/multi_object_camera_tracker_cpu.py \
    --arch LightTrackM_Subnet \
    --resume snapshot/LightTrackM/LightTrackM.pth \
    --path_name LightTrackM \
    --device cuda
```

## How to Use

### Selection Mode

1. When the application starts, you'll be in **SELECTION MODE**
2. Click and drag on the video frame to draw bounding boxes around objects you want to track
3. You can select multiple objects
4. Press **SPACE** to start tracking the selected objects
5. Press **C** to clear all selections and start over

### Tracking Mode

Once you press SPACE, the tracker will start tracking all selected objects:

- Each object gets a unique ID and color
- The tracker displays FPS and object count
- Press **P** to pause/resume tracking
- Press **R** to reset and return to selection mode
- Press **Q** to quit the application

## Keyboard Controls

| Key | Action |
|-----|--------|
| **SPACE** | Start tracking (in selection mode) |
| **C** | Clear all selections (in selection mode) |
| **P** | Pause/Resume tracking (in tracking mode) |
| **R** | Reset to selection mode (in tracking mode) |
| **Q** | Quit application |

## Requirements

- Python 3.x
- PyTorch
- OpenCV (cv2)
- NumPy
- easydict
- LightTrack model checkpoint

## Installation

1. Install required packages:
```bash
pip install torch opencv-python numpy easydict
```

2. Download the LightTrack model:
   - Visit: https://drive.google.com/drive/folders/1HXhdJO3yhQYw3O7nGUOXHu2S20Bs8CfI
   - Download `LightTrackM.pth`
   - Place it in: `snapshot/LightTrackM/`

## Troubleshooting

### Camera Not Opening
- Check if your camera is connected and not being used by another application
- Try a different camera ID: `./tracking/run_multi_tracker.sh -c 1`

### Model Not Found
- Ensure the model file exists at `snapshot/LightTrackM/LightTrackM.pth`
- Download it from the link above if missing

### Low FPS
- Try using GPU: `./tracking/run_multi_tracker.sh -d cuda`
- Reduce the number of tracked objects
- Use a lower resolution camera/video

### Import Errors
- Make sure all dependencies are installed
- The `torch._six` compatibility issue has been fixed in `lib/models/backbone/models/utils.py`

## Architecture

The multi-object tracker consists of:

1. **MultiObjectTracker**: Manages multiple tracker instances
2. **LighttrackMulti**: Device-agnostic tracker for individual objects
3. **BBoxSelector**: Interactive bounding box selection interface
4. **Shell Script**: Convenient launcher with argument parsing

## Performance

- CPU: ~10-20 FPS (depending on number of objects and hardware)
- GPU: ~30-60 FPS (depending on GPU and number of objects)

## Notes

- The tracker works best with distinct, well-separated objects
- Avoid selecting overlapping objects for better tracking accuracy
- The tracker uses the LightTrack algorithm optimized for speed and accuracy
