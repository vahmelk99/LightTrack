# Dynamic Multi-Object Tracking Guide

## New Features

### 1. Skip Initial Selection
Use `--no_initial_select` flag to start tracking immediately without waiting for initial object selection.

```bash
# Start tracking immediately (add objects on-the-fly)
python tracking/multi_object_camera_tracker_cpu.py \
    --resume snapshot/LightTrackM/LightTrackM.pth \
    --no_initial_select

# Traditional mode (select objects first)
python tracking/multi_object_camera_tracker_cpu.py \
    --resume snapshot/LightTrackM/LightTrackM.pth
```

### 2. Desktop Capture Mode
Use `--desktop` flag to capture your desktop screen instead of camera/video.

```bash
# Track objects on your desktop screen
python tracking/multi_object_camera_tracker_cpu.py \
    --resume snapshot/LightTrackM/LightTrackM.pth \
    --desktop

# Desktop capture with no initial selection
python tracking/multi_object_camera_tracker_cpu.py \
    --resume snapshot/LightTrackM/LightTrackM.pth \
    --desktop \
    --no_initial_select
```

**Note:** Desktop capture requires the `mss` library:
```bash
pip install mss
```

### 3. Dynamic Object Management

#### Add Objects During Tracking
- Simply draw bounding boxes while the video is running
- New objects are automatically assigned IDs

#### Remove Objects by ID
- Press number keys `0-9` to remove objects
- Example: Press `3` to remove object with ID 3

#### Smart ID Reuse
- When you remove object ID 2, the next object you add will get ID 2
- IDs stay sequential and reuse the lowest available number

## Usage Examples

### Example 1: Desktop Tracking with Dynamic Objects
```bash
python tracking/multi_object_camera_tracker_cpu.py \
    --resume snapshot/LightTrackM/LightTrackM.pth \
    --desktop \
    --no_initial_select \
    --device cpu
```
- Starts immediately tracking your desktop
- Draw boxes to add objects as they appear
- Press number keys to remove objects

### Example 2: Video File with Initial Selection
```bash
python tracking/multi_object_camera_tracker_cpu.py \
    --resume snapshot/LightTrackM/LightTrackM.pth \
    --video_file path/to/video.mp4
```
- Pauses on first frame for object selection
- Draw boxes around objects to track
- Press SPACE to start tracking

### Example 3: Camera with No Initial Selection
```bash
python tracking/multi_object_camera_tracker_cpu.py \
    --resume snapshot/LightTrackM/LightTrackM.pth \
    --camera 0 \
    --no_initial_select
```
- Starts tracking immediately from camera
- Add objects on-the-fly as needed

## Controls

### During Tracking
- **Mouse:** Draw boxes to add new objects
- **0-9 Keys:** Remove object by ID
- **P:** Pause/Resume
- **R:** Reset (go back to selection mode)
- **Q:** Quit

### During Selection Mode (if not using --no_initial_select)
- **Mouse:** Draw boxes around objects
- **SPACE:** Start tracking
- **C:** Clear all selections
- **Q:** Quit

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--resume` | Path to model weights (required) | - |
| `--arch` | Model architecture | LightTrackM_Subnet |
| `--camera` | Camera device ID | 0 |
| `--video_file` | Path to video file | None |
| `--desktop` | Enable desktop capture mode | False |
| `--no_initial_select` | Skip initial selection, start immediately | False |
| `--device` | Device to use (cpu/cuda) | cpu |
| `--stride` | Network stride | 16 |

## Tips

1. **Desktop Capture Performance:** Desktop capture may be slower than camera/video due to screen resolution. Consider reducing your screen resolution for better FPS.

2. **ID Management:** Objects are numbered 0-9. If you need to track more than 10 objects, you'll need to remove some first.

3. **Combining Flags:** You can combine `--desktop` and `--no_initial_select` for a seamless desktop tracking experience.

4. **GPU Acceleration:** Use `--device cuda` for better performance if you have a CUDA-capable GPU.
