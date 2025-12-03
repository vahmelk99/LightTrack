import _init_paths
import os
import sys

# Fix Qt platform plugin issues on Linux
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Force X11 backend
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''

import cv2
import torch
import argparse
import numpy as np

import lib.models.models as models

from easydict import EasyDict as edict
from lib.utils.utils import load_pretrain, cxy_wh_2_rect
from lib.tracker.lighttrack_multi import LighttrackMulti

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-Object Camera Tracking with LightTrack (CPU Version)')
    parser.add_argument('--arch', dest='arch', default='LightTrackM_Subnet', help='backbone architecture')
    parser.add_argument('--resume', type=str, required=True, help='pretrained model path')
    parser.add_argument('--dataset', default='VOT2019', help='dataset name for config')
    parser.add_argument('--stride', type=int, default=16, help='network stride')
    parser.add_argument('--even', type=int, default=0)
    parser.add_argument('--path_name', type=str, default='back_04502514044521042540+cls_211000022+reg_100000111_ops_32', 
                        help='model architecture path')
    parser.add_argument('--camera', type=int, default=0, help='camera device id')
    parser.add_argument('--video_file', type=str, default=None, help='video file path (optional, instead of camera)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='device to use')
    parser.add_argument('--no_initial_select', action='store_true', help='skip initial selection, start tracking immediately')
    parser.add_argument('--desktop', action='store_true', help='capture desktop screen instead of camera/video')
    args = parser.parse_args()
    return args


class MultiObjectTracker:
    def __init__(self, siam_info, model, args):
        self.trackers = []
        self.states = []
        self.colors = []
        self.model = model
        self.siam_info = siam_info
        self.args = args
        self.next_id = 0
        self.device = args.device
        self.available_ids = []  # Track available IDs from removed objects
        
    def add_object(self, frame, bbox):
        """Add a new object to track"""
        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2
        
        target_pos = np.array([cx, cy])
        target_sz = np.array([w, h])
        
        tracker = LighttrackMulti(self.siam_info, device=self.device, even=self.args.even)
        state = tracker.init(frame, target_pos, target_sz, self.model)
        
        color = tuple(np.random.randint(50, 255, 3).tolist())
        
        # Reuse available ID if any, otherwise use next_id
        if self.available_ids:
            obj_id = min(self.available_ids)  # Use the smallest available ID
            self.available_ids.remove(obj_id)
            
            # Insert at the correct position
            self.trackers.insert(obj_id, tracker)
            self.states.insert(obj_id, state)
            self.colors.insert(obj_id, color)
        else:
            obj_id = self.next_id
            self.next_id += 1
            
            self.trackers.append(tracker)
            self.states.append(state)
            self.colors.append(color)
        
        return obj_id
    
    def track_all(self, frame):
        """Track all objects in the current frame"""
        results = []
        
        for i, (tracker, state) in enumerate(zip(self.trackers, self.states)):
            state = tracker.track(state, frame)
            self.states[i] = state
            
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            results.append(location)
        
        return results
    
    def remove_object(self, obj_id):
        """Remove an object from tracking"""
        if 0 <= obj_id < len(self.trackers):
            del self.trackers[obj_id]
            del self.states[obj_id]
            del self.colors[obj_id]
            
            # Add this ID to available IDs for reuse
            if obj_id not in self.available_ids:
                self.available_ids.append(obj_id)
                self.available_ids.sort()
            
            return True
        return False
    
    def clear_all(self):
        """Clear all tracked objects"""
        self.trackers.clear()
        self.states.clear()
        self.colors.clear()
        self.next_id = 0
        self.available_ids.clear()


class BBoxSelector:
    def __init__(self):
        self.boxes = []
        self.current_box = None
        self.drawing = False
        self.start_point = None
        self.enabled = True  # Can be disabled during tracking
        
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
                                   x - self.start_point[0], y - self.start_point[1])
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.start_point is not None:
                x1, y1 = self.start_point
                w = x - x1
                h = y - y1
                
                if w < 0:
                    x1 = x
                    w = -w
                if h < 0:
                    y1 = y
                    h = -h
                    
                if w > 10 and h > 10:
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
        
        # Ensure coordinates are within frame
        x = max(0, min(x, frame.shape[1] - 1))
        y = max(0, min(y, frame.shape[0] - 1))
        w = max(1, min(w, frame.shape[1] - x))
        h = max(1, min(h, frame.shape[0] - y))
        
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
        
        label = f"ID: {i}" if labels is None else labels[i]
        
        # Draw label background
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(display_frame, (x, y - label_h - 10), (x + label_w, y), color, -1)
        cv2.putText(display_frame, label, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return display_frame


def main():
    args = parse_args()
    
    siam_info = edict()
    siam_info.arch = args.arch
    siam_info.dataset = args.dataset
    siam_info.epoch_test = False
    siam_info.stride = args.stride
    
    print('Loading model...')
    if args.path_name != 'NULL':
        siam_net = models.__dict__[args.arch](args.path_name, stride=siam_info.stride)
    else:
        siam_net = models.__dict__[args.arch](stride=siam_info.stride)
    
    siam_net = load_pretrain(siam_net, args.resume)
    siam_net.eval()
    
    if args.device == 'cuda' and torch.cuda.is_available():
        siam_net = siam_net.cuda()
        print('Model loaded on CUDA')
    else:
        siam_net = siam_net.cpu()
        print('Model loaded on CPU')
    
    multi_tracker = MultiObjectTracker(siam_info, siam_net, args)
    
    # Open camera, video file, or desktop capture
    cap = None
    sct = None
    monitor = None
    
    if args.desktop:
        if not MSS_AVAILABLE:
            print("Error: mss library not installed. Install with: pip install mss")
            return
        sct = mss.mss()
        monitor = sct.monitors[1]  # Primary monitor
        print(f'Desktop capture mode: {monitor["width"]}x{monitor["height"]}')
    elif args.video_file:
        cap = cv2.VideoCapture(args.video_file)
        print(f'Opened video file: {args.video_file}')
        if not cap.isOpened():
            print("Error: Cannot open video file")
            return
    else:
        cap = cv2.VideoCapture(args.camera)
        print(f'Opened camera device: {args.camera}')
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return
    
    bbox_selector = BBoxSelector()
    
    window_name = 'Multi-Object Tracker'
    print(f"Creating display window...")
    
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.setMouseCallback(window_name, bbox_selector.mouse_callback)
        print("✓ Window created successfully")
    except Exception as e:
        print(f"✗ Error creating window: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have X11 or Wayland display server running")
        print("2. Try: export DISPLAY=:0")
        print("3. Check if you can run: xclock (to test X11)")
        print("4. Install opencv-python-headless if running on a server without display")
        return
    
    mode = 'track' if args.no_initial_select else 'select'
    paused = False
    
    print(f"\nOpenCV version: {cv2.__version__}")
    
    print("\n" + "="*50)
    print("Multi-Object Tracking Instructions")
    print("="*50)
    if not args.no_initial_select:
        print("\nSELECTION MODE:")
        print("  • Click and drag to draw boxes around objects")
        print("  • Press SPACE to start tracking")
        print("  • Press 'c' to clear all selections")
    print("\nTRACKING MODE:")
    print("  • Draw new boxes to add objects during tracking")
    print("  • Press number keys (0-9) to remove object by ID")
    print("  • Press 'p' to pause/resume")
    print("  • Press 'r' to reset and select new objects")
    print("  • Press 'q' to quit")
    print("="*50 + "\n")
    
    # Read first frame
    try:
        if sct:
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            print(f"Desktop capture: {frame.shape}, dtype={frame.dtype}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            print(f"After conversion: {frame.shape}, dtype={frame.dtype}")
        else:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame")
                return
            print(f"Frame captured: {frame.shape}, dtype={frame.dtype}")
    except Exception as e:
        print(f"Error capturing first frame: {e}")
        import traceback
        traceback.print_exc()
        return
    
    selection_frame = frame.copy()
    display_frame = frame.copy()  # Initialize for paused state
    fps_counter = 0
    fps_start_time = cv2.getTickCount()
    current_fps = 0
    
    while True:
        if mode == 'select':
            display_frame = selection_frame.copy()
            
            boxes = bbox_selector.get_boxes()
            for bbox in boxes:
                x, y, w, h = bbox
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            if bbox_selector.current_box:
                x, y, w, h = bbox_selector.current_box
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            
            cv2.putText(display_frame, "SELECTION MODE", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Objects selected: {len(boxes)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, "SPACE: Start | C: Clear", 
                       (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow(window_name, display_frame)
            
        else:
            # Capture frame
            if sct:
                screenshot = sct.grab(monitor)
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            else:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("End of video/camera stream")
                        break
            
            if not paused:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Check if new objects were drawn
                new_boxes = bbox_selector.get_boxes()
                if new_boxes:
                    with torch.no_grad():
                        for bbox in new_boxes:
                            obj_id = multi_tracker.add_object(rgb_frame, bbox)
                            print(f"Added new object with ID: {obj_id}")
                    bbox_selector.clear()
                
                # Track all objects (only if there are objects to track)
                if len(multi_tracker.trackers) > 0:
                    with torch.no_grad():
                        tracked_boxes = multi_tracker.track_all(rgb_frame)
                    display_frame = draw_boxes(frame, tracked_boxes, multi_tracker.colors)
                else:
                    display_frame = frame.copy()
                
                # Draw current box being drawn
                if bbox_selector.current_box:
                    x, y, w, h = bbox_selector.current_box
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                
                # Calculate FPS
                fps_counter += 1
                if fps_counter >= 10:
                    fps_end_time = cv2.getTickCount()
                    time_diff = (fps_end_time - fps_start_time) / cv2.getTickFrequency()
                    current_fps = fps_counter / time_diff
                    fps_counter = 0
                    fps_start_time = cv2.getTickCount()
                
                status = "PAUSED" if paused else "TRACKING"
                cv2.putText(display_frame, f"{status} | FPS: {current_fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Objects: {len(multi_tracker.trackers)}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                if len(multi_tracker.trackers) == 0:
                    cv2.putText(display_frame, "Draw boxes to add objects", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                cv2.putText(display_frame, "Draw: Add | 0-9: Remove | P: Pause | R: Reset | Q: Quit", 
                           (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow(window_name, display_frame)
            else:
                # When paused, just show the last frame
                cv2.imshow(window_name, display_frame)
                cv2.waitKey(10)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' ') and mode == 'select':
            boxes = bbox_selector.get_boxes()
            if len(boxes) > 0:
                print(f"\nStarting tracking with {len(boxes)} objects...")
                rgb_frame = cv2.cvtColor(selection_frame, cv2.COLOR_BGR2RGB)
                
                with torch.no_grad():
                    for bbox in boxes:
                        multi_tracker.add_object(rgb_frame, bbox)
                
                mode = 'track'
                bbox_selector.clear()
                fps_counter = 0
                fps_start_time = cv2.getTickCount()
                print("Tracking started!\n")
            else:
                print("No objects selected!")
        elif key == ord('c') and mode == 'select':
            bbox_selector.clear()
            print("Selections cleared")
        elif key == ord('r') and mode == 'track':
            multi_tracker.clear_all()
            bbox_selector.clear()
            mode = 'select'
            if sct:
                screenshot = sct.grab(monitor)
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                selection_frame = frame.copy()
            else:
                ret, frame = cap.read()
                if ret:
                    selection_frame = frame.copy()
            print("\nReset to selection mode\n")
        elif key == ord('p') and mode == 'track':
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif mode == 'track' and key >= ord('0') and key <= ord('9'):
            # Remove object by ID
            obj_id = key - ord('0')
            if multi_tracker.remove_object(obj_id):
                print(f"Removed object ID: {obj_id}")
            else:
                print(f"No object with ID: {obj_id}")
    
    if cap:
        cap.release()
    if sct:
        sct.close()
    cv2.destroyAllWindows()
    print("\nTracking ended")


if __name__ == '__main__':
    main()
