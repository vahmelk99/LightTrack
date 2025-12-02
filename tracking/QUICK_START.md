# Multi-Object Tracker - Quick Start Guide

## 🚀 Run the Tracker

```bash
./tracking/run_multi_tracker.sh
```

## 🎮 Controls

### Selection Mode (Draw boxes around objects)
```
🖱️  Click & Drag  → Draw bounding box
⎵   SPACE        → Start tracking
C   Clear        → Clear all selections
```

### Tracking Mode (Objects being tracked)
```
P   Pause/Resume → Pause or resume tracking
R   Reset        → Return to selection mode
Q   Quit         → Exit application
```

## 📋 Common Commands

```bash
# Default: Camera 0, CPU
./tracking/run_multi_tracker.sh

# Use GPU for faster tracking
./tracking/run_multi_tracker.sh -d cuda

# Track from video file
./tracking/run_multi_tracker.sh -v video.mp4

# Use different camera
./tracking/run_multi_tracker.sh -c 1

# Show help
./tracking/run_multi_tracker.sh --help
```

## ✅ Verify Setup

```bash
cd tracking
python test_multi_tracker_import.py
```

## 💡 Tips

- Select distinct, well-separated objects for best results
- Avoid overlapping objects
- GPU mode provides 2-3x better FPS
- You can track multiple objects simultaneously
- Each object gets a unique ID and color

## 🐛 Troubleshooting

**Camera not opening?**
- Check if camera is connected
- Try different camera ID: `-c 1`

**Low FPS?**
- Use GPU: `-d cuda`
- Reduce number of tracked objects

**Import errors?**
- Run: `pip install torch opencv-python numpy easydict`

## 📚 Full Documentation

See `MULTI_OBJECT_TRACKER_README.md` for complete documentation.
