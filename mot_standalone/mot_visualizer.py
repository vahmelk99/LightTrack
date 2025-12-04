"""
Multi-Object Tracker Visualizer
Test and visualize the MOT wrapper with camera/video input
"""

import cv2
import argparse
import numpy as np
from mot_wrapper import MOT


class BBoxSelector:
    """Interactive bounding box selector"""
    def __init__(self):
        self.boxes = []
        self.current_box = None
        self.drawing = False
        self.start_point = None
        self.enabled = True
    
    def mouse_callback(self, event, x, y, flags, param):
        if not self.enabled:
            return
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.current_box = None
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_box = (self.start_point[0], self.start_point[1],
                                   x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.start_point is not None:
                x1, y1 = self.start_point
                x2, y2 = x, y
                
                # Ensure x1 < x2 and y1 < y2
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                
                w = x2 - x1
                h = y2 - y1
                
                if w > 5 and h > 5:  # Minimum box size
                    self.boxes.append([x1, y1, x2, y2])
                
                self.current_box = None
                self.start_point = None
    
    def get_boxes(self):
        return self.boxes.copy()
    
    def clear(self):
        self.boxes.clear()
        self.current_box = None
        self.drawing = False
        self.start_point = None


def draw_boxes(frame, tracked_boxes, colors=None):
    """
    Draw bounding boxes on frame
    
    Args:
        frame: Input frame
        tracked_boxes: List of tuples [(id, box), ...]
        colors: Optional dict of colors {id: (B, G, R)}
    
    Returns:
        Frame with boxes drawn
    """
    display_frame = frame.copy()
    
    if colors is None:
        colors = {}
    
    for obj_id, bbox in tracked_boxes:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # Get or generate color for this ID
        if obj_id not in colors:
            np.random.seed(obj_id * 123)
            colors[obj_id] = tuple(np.random.randint(50, 255, 3).tolist())
        
        color = colors[obj_id]
        
        # Draw rectangle
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"ID: {obj_id}"
        cv2.putText(display_frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return display_frame


def main():
    parser = argparse.ArgumentParser(description='Multi-Object Tracker Visualizer')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to pretrained model weights')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--video', type=str, default=None,
                       help='Video file path (optional, instead of camera)')
    parser.add_argument('--arch', type=str, default='LightTrackM_Subnet',
                       help='Model architecture')
    parser.add_argument('--path_name', type=str,
                       default='back_04502514044521042540+cls_211000022+reg_100000111_ops_32',
                       help='Model architecture path')
    parser.add_argument('--stride', type=int, default=16,
                       help='Network stride')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to run on')
    args = parser.parse_args()
    
    # Initialize tracker
    print("Initializing Multi-Object Tracker...")
    mot = MOT(
        model_path=args.model,
        arch=args.arch,
        path_name=args.path_name,
        stride=args.stride,
        device=args.device
    )
    print("Tracker initialized!")
    
    # Open video source
    if args.video:
        cap = cv2.VideoCapture(args.video)
        print(f'Opened video file: {args.video}')
    else:
        cap = cv2.VideoCapture(args.camera)
        print(f'Opened camera device: {args.camera}')
    
    if not cap.isOpened():
        print("Error: Cannot open video source")
        return
    
    # Setup window and mouse callback
    window_name = 'Multi-Object Tracker'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    bbox_selector = BBoxSelector()
    cv2.setMouseCallback(window_name, bbox_selector.mouse_callback)
    
    # State variables
    mode = 'track'  # Start in tracking mode (no initial selection required)
    paused = False
    colors = {}
    
    print("\n=== Multi-Object Tracking Instructions ===")
    print("TRACKING MODE:")
    print("  - Draw boxes to add objects to track")
    print("  - Press number keys (0-9) to remove object by ID")
    print("  - Press 'p' to pause/resume")
    print("  - Press 'r' to reset all objects")
    print("  - Press 'q' to quit")
    print("==========================================\n")
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame")
        return
    
    display_frame = frame.copy()
    
    while True:
        # Always in tracking mode now
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video/camera stream")
                break
            
            # Check if new objects were drawn
            new_boxes = bbox_selector.get_boxes()
            if new_boxes:
                for box in new_boxes:
                    obj_id = mot.add_box(frame, box)
                    print(f"Added new object with ID: {obj_id}")
                bbox_selector.clear()
            
            # Track all objects
            tracked_boxes = mot.track(frame)
            
            # Draw results
            display_frame = draw_boxes(frame, tracked_boxes, colors)
            
            # Draw current box being drawn
            if bbox_selector.current_box:
                x1, y1, x2, y2 = bbox_selector.current_box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            status = "TRACKING"
            cv2.putText(display_frame, f"{status} - Draw to add, 0-9 to remove, 'p' pause, 'r' reset",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Objects: {mot.get_num_objects()}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if mot.get_num_objects() == 0:
                cv2.putText(display_frame, "Draw boxes to add objects",
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow(window_name, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset all objects
            mot.clear_all()
            bbox_selector.clear()
            colors.clear()
            print("Reset - all objects cleared")
        elif key == ord('p'):
            # Pause/resume
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key >= ord('0') and key <= ord('9'):
            # Remove object by ID
            obj_id = key - ord('0')
            if mot.remove_box(obj_id):
                print(f"Removed object ID: {obj_id}")
                if obj_id in colors:
                    del colors[obj_id]
            else:
                print(f"No object with ID: {obj_id}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Tracking ended")


if __name__ == '__main__':
    main()
