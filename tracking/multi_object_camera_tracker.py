import _init_paths
import os
import cv2
import torch
import argparse
import numpy as np

import lib.models.models as models

from easydict import EasyDict as edict
from lib.utils.utils import load_pretrain, cxy_wh_2_rect
from lib.tracker.lighttrack import Lighttrack


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-Object Camera Tracking with LightTrack')
    parser.add_argument('--arch', dest='arch', default='LightTrackM_Subnet', help='backbone architecture')
    parser.add_argument('--resume', type=str, required=True, help='pretrained model path')
    parser.add_argument('--dataset', default='VOT2019', help='dataset name for config')
    parser.add_argument('--stride', type=int, default=16, help='network stride')
    parser.add_argument('--even', type=int, default=0)
    parser.add_argument('--path_name', type=str, default='back_04502514044521042540+cls_211000022+reg_100000111_ops_32',
                        help='model architecture path')
    parser.add_argument('--camera', type=int, default=0, help='camera device id')
    parser.add_argument('--video_file', type=str, default=None, help='video file path (optional, instead of camera)')
    args = parser.parse_args()
    return args


class MultiObjectTracker:
    def __init__(self, siam_info, model, args):
        self.trackers = []  # List of tracker instances
        self.states = []    # List of tracker states
        self.colors = []    # List of colors for each tracked object
        self.model = model
        self.siam_info = siam_info
        self.args = args
        self.next_id = 0
        
    def add_object(self, frame, bbox):
        """Add a new object to track
        Args:
            frame: RGB image
            bbox: (x, y, w, h) bounding box
        """
        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2
        
        target_pos = np.array([cx, cy])
        target_sz = np.array([w, h])
        
        # Create new tracker instance
        tracker = Lighttrack(self.siam_info, even=self.args.even)
        
        # Initialize tracker
        state = tracker.init(frame, target_pos, target_sz, self.model)
        
        self.trackers.append(tracker)
        self.states.append(state)
        
        # Generate random color for this object
        color = tuple(np.random.randint(0, 255, 3).tolist())
        self.colors.append(color)
        
        obj_id = self.next_id
        self.next_id += 1
        
        return obj_id
    
    def track_all(self, frame):
        """Track all objects in the current frame
        Args:
            frame: RGB image
        Returns:
            List of bounding boxes [(x, y, w, h), ...]
        """
        results = []
        
        for i, (tracker, state) in enumerate(zip(self.trackers, self.states)):
            # Update tracker
            state = tracker.track(state, frame)
            self.states[i] = state
            
            # Get bounding box
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            results.append(location)
        
        return results
    
    def remove_object(self, obj_id):
        """Remove an object from tracking"""
        if 0 <= obj_id < len(self.trackers):
            del self.trackers[obj_id]
            del self.states[obj_id]
            del self.colors[obj_id]
            return True
        return False
    
    def clear_all(self):
        """Clear all tracked objects"""
        self.trackers.clear()
        self.states.clear()
        self.colors.clear()
        self.next_id = 0


class BBoxSelector:
    def __init__(self):
        self.boxes = []
        self.current_box = None
        self.drawing = False
        self.start_point = None
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.current_box = None
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_box = (self.start_point[0], self.start_point[1], 
                                   x - self.start_point[0], y - self.start_point[1])
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.start_point is not None:
                x1, y1 = self.start_point
                w = x - x1
                h = y - y1
                
                # Ensure positive width and height
                if w < 0:
                    x1 = x
                    w = -w
                if h < 0:
                    y1 = y
                    h = -h
                    
                if w > 5 and h > 5:  # Minimum box size
                    self.boxes.append((x1, y1, w, h))
                    
                self.current_box = None
                self.start_point = None
    
    def get_boxes(self):
        return self.boxes.copy()
    
    def clear(self):
        self.boxes.clear()
        self.current_box = None
        self.drawing = False
        self.start_point = None


def draw_boxes(frame, boxes, colors, labels=None):
    """Draw bounding boxes on frame"""
    display_frame = frame.copy()
    
    for i, (bbox, color) in enumerate(zip(boxes, colors)):
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
        
        label = f"ID: {i}" if labels is None else labels[i]
        cv2.putText(display_frame, label, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return display_frame


def main():
    args = parse_args()
    
    # Setup model info
    siam_info = edict()
    siam_info.arch = args.arch
    siam_info.dataset = args.dataset
    siam_info.epoch_test = False
    siam_info.stride = args.stride
    
    # Build siamese network
    print('Loading model...')
    if args.path_name != 'NULL':
        siam_net = models.__dict__[args.arch](args.path_name, stride=siam_info.stride)
    else:
        siam_net = models.__dict__[args.arch](stride=siam_info.stride)
    
    siam_net = load_pretrain(siam_net, args.resume)
    siam_net.eval()
    siam_net = siam_net.cuda()
    print('Model loaded successfully!')
    
    # Initialize multi-object tracker
    multi_tracker = MultiObjectTracker(siam_info, siam_net, args)
    
    # Open camera or video file
    if args.video_file:
        cap = cv2.VideoCapture(args.video_file)
        print(f'Opened video file: {args.video_file}')
    else:
        cap = cv2.VideoCapture(args.camera)
        print(f'Opened camera device: {args.camera}')
    
    if not cap.isOpened():
        print("Error: Cannot open camera/video")
        return
    
    # Set up window
    window_name = 'Multi-Object Tracker'
    cv2.namedWindow(window_name)
    
    bbox_selector = BBoxSelector()
    cv2.setMouseCallback(window_name, bbox_selector.mouse_callback)
    
    mode = 'select'  # 'select' or 'track'
    paused = False
    
    print("\n=== Multi-Object Tracking Instructions ===")
    print("SELECTION MODE:")
    print("  - Draw boxes around objects to track")
    print("  - Press SPACE to start tracking")
    print("  - Press 'c' to clear all selections")
    print("\nTRACKING MODE:")
    print("  - Press 'p' to pause/resume")
    print("  - Press 'r' to reset and select new objects")
    print("  - Press 'q' to quit")
    print("==========================================\n")
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame")
        return
    
    selection_frame = frame.copy()
    
    while True:
        if mode == 'select':
            display_frame = selection_frame.copy()
            
            # Draw existing boxes
            boxes = bbox_selector.get_boxes()
            for bbox in boxes:
                x, y, w, h = bbox
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw current box being drawn
            if bbox_selector.current_box:
                x, y, w, h = bbox_selector.current_box
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            
            cv2.putText(display_frame, "SELECTION MODE - Draw boxes, SPACE to track, 'c' to clear", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(window_name, display_frame)
            
        else:  # tracking mode
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video/camera stream")
                    break
                
                # Convert to RGB for tracker
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Track all objects
                tracked_boxes = multi_tracker.track_all(rgb_frame)
                
                # Draw results
                display_frame = draw_boxes(frame, tracked_boxes, multi_tracker.colors)
                
                status = "PAUSED" if paused else "TRACKING"
                cv2.putText(display_frame, f"{status} - 'p' pause, 'r' reset, 'q' quit", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Objects: {len(multi_tracker.trackers)}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow(window_name, display_frame)
            else:
                cv2.waitKey(10)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' ') and mode == 'select':
            # Start tracking
            boxes = bbox_selector.get_boxes()
            if len(boxes) > 0:
                print(f"Starting tracking with {len(boxes)} objects...")
                rgb_frame = cv2.cvtColor(selection_frame, cv2.COLOR_BGR2RGB)
                
                with torch.no_grad():
                    for bbox in boxes:
                        multi_tracker.add_object(rgb_frame, bbox)
                
                mode = 'track'
                print("Tracking started!")
            else:
                print("No objects selected!")
        elif key == ord('c') and mode == 'select':
            # Clear selections
            bbox_selector.clear()
            print("Selections cleared")
        elif key == ord('r') and mode == 'track':
            # Reset to selection mode
            multi_tracker.clear_all()
            bbox_selector.clear()
            mode = 'select'
            ret, frame = cap.read()
            if ret:
                selection_frame = frame.copy()
            print("Reset to selection mode")
        elif key == ord('p') and mode == 'track':
            # Pause/resume
            paused = not paused
            print("Paused" if paused else "Resumed")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Tracking ended")


if __name__ == '__main__':
    main()
