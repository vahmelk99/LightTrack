# Multi-Object Camera Tracking

This extension adds multi-object tracking capability to LightTrack, allowing you to track multiple objects simultaneously from a camera or video file.

## Features

- **Real-time camera input**: Track objects from webcam or any video capture device
- **Video file support**: Process pre-recorded videos
- **Interactive object selection**: Draw bounding boxes around objects you want to track
- **Multiple object tracking**: Track unlimited number of objects simultaneously
- **Visual feedback**: Each tracked object has a unique color and ID
- **Pause/Resume**: Control tracking flow
- **Reset capability**: Clear all tracked objects and start fresh
- **FPS display**: Monitor tracking performance
- **CPU/GPU support**: Works on both CPU and CUDA-enabled GPUs

## Installation

Make sure you have the base LightTrack environment set up:

```bash
cd lighttrack
conda activate lighttrack
```

Additional requirements (if not already installed):
```bash
pip install opencv-python
```

## Usage

### Basic Usage (GPU)

```bash
python tracking/multi_object_camera_tracker.py \
    --arch LightTrackM_Subnet \
    --resume snapshot/LightTrackM/LightTrackM.pth \
    --path_name LightTrackM
```

### CPU Version

For systems without CUDA or for testing:

```bash
python tracking/multi_object_camera_tracker_cpu.py \
    --arch LightTrackM_Subnet \
    --resume snapshot/LightTrackM/LightTrackM.pth \
    --path_name LightTrackM \
    --device cpu
```

### Using a Video File

Instead of camera input:

```bash
python tracking/multi_object_camera_tracker_cpu.py \
    --arch LightTrackM_Subnet \
    --resume snapshot/LightTrackM/LightTrackM.pth \
    --path_name LightTrackM \
    --video_file path/to/your/video.mp4 \
    --device cpu
```

### Using Different Camera

To use a different camera device (default is 0):

```bash
python tracking/multi_object_camera_tracker_cpu.py \
    --arch LightTrackM_Subnet \
    --resume snapshot/LightTrackM/LightTrackM.pth \
    --path_name LightTrackM \
    --camera 1 \
    --device cpu
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--arch` | str | LightTrackM_Subnet | Backbone architecture |
| `--resume` | str | Required | Path to pretrained model |
| `--dataset` | str | VOT2019 | Dataset name for config |
| `--stride` | int | 16 | Network stride |
| `--even` | int | 0 | Even flag for model |
| `--path_name` | str | LightTrackM | Model path name |
| `--camera` | int | 0 | Camera device ID |
| `--video_file` | str | None | Video file path (optional) |
| `--device` | str | cpu | Device to use (cpu/cuda) |

## Interactive Controls

### Selection Mode

When you first start the application, you'll be in **Selection Mode**:

1. **Draw Bounding Boxes**: Click and drag on the frame to draw rectangles around objects you want to track
2. **Multiple Objects**: Draw as many boxes as you need
3. **Start Tracking**: Press `SPACE` to begin tracking all selected objects
4. **Clear Selections**: Press `c` to remove all drawn boxes and start over

### Tracking Mode

Once tracking starts:

- **Pause/Resume**: Press `p` to pause or resume tracking
- **Reset**: Press `r` to stop tracking and return to selection mode
- **Quit**: Press `q` to exit the application

## How It Works

### Architecture

The multi-object tracker creates independent tracker instances for each selected object:

1. **Initialization**: Each object gets its own LightTrack instance initialized with the selected bounding box
2. **Template Extraction**: The model extracts a template feature for each object
3. **Parallel Tracking**: All objects are tracked independently in each frame
4. **Visual Output**: Results are displayed with unique colors and IDs

### Performance Considerations

- **GPU vs CPU**: GPU provides significantly better performance (10-30 FPS vs 1-5 FPS)
- **Number of Objects**: More objects = slower tracking (roughly linear relationship)
- **Image Resolution**: Lower resolution = faster tracking
- **Model Size**: LightTrackM is optimized for mobile/edge devices

### Tips for Best Results

1. **Object Selection**:
   - Draw tight bounding boxes around objects
   - Ensure the entire object is visible in the first frame
   - Avoid selecting partially occluded objects

2. **Tracking Quality**:
   - Good lighting conditions improve tracking
   - Avoid rapid camera movements
   - Objects with distinct features track better
   - Minimize occlusions between tracked objects

3. **Performance**:
   - Use GPU for real-time tracking of multiple objects
   - Reduce camera resolution if needed
   - Track fewer objects for higher FPS

## Code Structure

### Main Components

1. **MultiObjectTracker**: Manages multiple tracker instances
   - `add_object()`: Initialize tracking for a new object
   - `track_all()`: Update all trackers with new frame
   - `remove_object()`: Remove a specific tracker
   - `clear_all()`: Reset all trackers

2. **BBoxSelector**: Handles interactive bounding box selection
   - Mouse callback for drawing boxes
   - Box validation and storage

3. **LighttrackMulti**: Device-agnostic tracker wrapper
   - Supports both CPU and CUDA
   - Compatible with original LightTrack models

### File Organization

```
tracking/
├── multi_object_camera_tracker.py      # GPU version
├── multi_object_camera_tracker_cpu.py  # CPU/GPU version with device selection
└── MULTI_OBJECT_TRACKING.md           # This file

lib/tracker/
├── lighttrack.py                       # Original single-object tracker
└── lighttrack_multi.py                 # Device-agnostic multi-object tracker
```

## Troubleshooting

### Camera Not Opening

```
Error: Cannot open camera/video
```

**Solutions**:
- Check camera is connected and not used by another application
- Try different camera ID: `--camera 1` or `--camera 2`
- Verify camera permissions on your system
- Test with a video file instead: `--video_file test.mp4`

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solutions**:
- Use CPU mode: `--device cpu`
- Track fewer objects
- Reduce camera resolution
- Close other GPU-intensive applications

### Slow Performance

**Solutions**:
- Use GPU if available: `--device cuda`
- Reduce number of tracked objects
- Lower camera resolution
- Use a smaller model if available

### Poor Tracking Quality

**Solutions**:
- Improve lighting conditions
- Draw tighter bounding boxes
- Avoid tracking very small objects
- Minimize occlusions
- Reduce camera motion

## Examples

### Example 1: Track People in a Room

```bash
python tracking/multi_object_camera_tracker_cpu.py \
    --arch LightTrackM_Subnet \
    --resume snapshot/LightTrackM/LightTrackM.pth \
    --path_name LightTrackM \
    --device cuda
```

1. Position camera to see multiple people
2. Draw boxes around each person
3. Press SPACE to start tracking
4. Watch as each person is tracked with a unique color

### Example 2: Track Objects in a Video

```bash
python tracking/multi_object_camera_tracker_cpu.py \
    --arch LightTrackM_Subnet \
    --resume snapshot/LightTrackM/LightTrackM.pth \
    --path_name LightTrackM \
    --video_file demo_video.mp4 \
    --device cpu
```

1. First frame appears
2. Draw boxes around objects of interest
3. Press SPACE to process the video
4. Press 'p' to pause and examine results

## Future Enhancements

Possible improvements for this implementation:

- [ ] Object re-identification after occlusion
- [ ] Automatic object detection (no manual selection)
- [ ] Save tracking results to file
- [ ] Trajectory visualization
- [ ] Object counting and statistics
- [ ] Multi-camera support
- [ ] Online object addition during tracking
- [ ] Confidence score display
- [ ] Export tracked video with annotations

## References

- Original LightTrack Paper: [arXiv:2104.14545](https://arxiv.org/abs/2104.14545)
- LightTrack Repository: [GitHub](https://github.com/researchmm/LightTrack)

## License

This extension follows the same license as the original LightTrack project.
