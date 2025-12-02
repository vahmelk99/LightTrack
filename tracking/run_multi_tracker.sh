#!/bin/bash

# Multi-Object Camera Tracker Launch Script
# Usage: ./run_multi_tracker.sh [options]

# Default values
ARCH="LightTrackM_Subnet"
RESUME="snapshot/LightTrackM/LightTrackM.pth"
PATH_NAME="back_04502514044521042540+cls_211000022+reg_100000111_ops_32"
DEVICE="cpu"
CAMERA=0
VIDEO_FILE=""

# Help message
show_help() {
    echo "Multi-Object Camera Tracker"
    echo ""
    echo "Usage: ./run_multi_tracker.sh [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -d, --device DEVICE     Device to use: cpu or cuda (default: cpu)"
    echo "  -c, --camera ID         Camera device ID (default: 0)"
    echo "  -v, --video FILE        Video file path (optional)"
    echo "  -m, --model PATH        Model checkpoint path (default: snapshot/LightTrackM/LightTrackM.pth)"
    echo ""
    echo "Examples:"
    echo "  ./run_multi_tracker.sh                          # Use default camera with CPU"
    echo "  ./run_multi_tracker.sh -d cuda                  # Use GPU"
    echo "  ./run_multi_tracker.sh -c 1                     # Use camera device 1"
    echo "  ./run_multi_tracker.sh -v video.mp4            # Process video file"
    echo "  ./run_multi_tracker.sh -d cuda -v video.mp4    # Process video with GPU"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -c|--camera)
            CAMERA="$2"
            shift 2
            ;;
        -v|--video)
            VIDEO_FILE="$2"
            shift 2
            ;;
        -m|--model)
            RESUME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if model file exists
if [ ! -f "$RESUME" ]; then
    echo "Error: Model file not found: $RESUME"
    echo ""
    echo "Please download the model first:"
    echo "  1. Visit: https://drive.google.com/drive/folders/1HXhdJO3yhQYw3O7nGUOXHu2S20Bs8CfI"
    echo "  2. Download LightTrackM.pth"
    echo "  3. Place it in: snapshot/LightTrackM/"
    echo ""
    exit 1
fi

# Build command
CMD="python tracking/multi_object_camera_tracker_cpu.py \
    --arch $ARCH \
    --resume $RESUME \
    --path_name $PATH_NAME \
    --device $DEVICE \
    --camera $CAMERA"

# Add video file if specified
if [ -n "$VIDEO_FILE" ]; then
    if [ ! -f "$VIDEO_FILE" ]; then
        echo "Error: Video file not found: $VIDEO_FILE"
        exit 1
    fi
    CMD="$CMD --video_file $VIDEO_FILE"
fi

# Print configuration
echo "================================"
echo "Multi-Object Tracker Configuration"
echo "================================"
echo "Device: $DEVICE"
echo "Model: $RESUME"
if [ -n "$VIDEO_FILE" ]; then
    echo "Input: Video file ($VIDEO_FILE)"
else
    echo "Input: Camera device $CAMERA"
fi
echo "================================"
echo ""

# Run the tracker
echo "Starting tracker..."
echo ""
eval $CMD
