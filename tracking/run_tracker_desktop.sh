#!/bin/bash
# Wrapper script to run multi-object tracker with desktop capture
# This sets proper environment variables to avoid Qt/display issues

# Force X11 backend (fixes Wayland issues)
export QT_QPA_PLATFORM=xcb
export QT_QPA_PLATFORM_PLUGIN_PATH=""

# Disable OpenCV's Qt backend if it causes issues (fallback to GTK)
# Uncomment the line below if you still have issues:
# export OPENCV_VIDEOIO_PRIORITY_QT=0

echo "Starting multi-object tracker with desktop capture..."
echo "Environment: QT_QPA_PLATFORM=$QT_QPA_PLATFORM"

# Default arguments
RESUME_PATH="${1:-snapshot/LightTrackM/LightTrackM.pth}"
DEVICE="${2:-cpu}"

python tracking/multi_object_camera_tracker_cpu.py \
    --resume "$RESUME_PATH" \
    --desktop \
    --no_initial_select \
    --device "$DEVICE"
