import _init_paths
import os
import cv2
import torch
import argparse
import numpy as np

import lib.models.models as models

from easydict import EasyDict as edict
from lib.utils.utils import load_pretrain, cxy_wh_2_rect
from lib.tracker.lighttrack_multi import LighttrackMulti


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
        
    def add_object(self, frame, bbox):
        """Add a new object to track"""
        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2
        
        target_pos = np.array([cx, cy])
        target_sz = np.array([w, h])
        
        tracker = LighttrackMulti(self.siam_info, device=self.device, even=self.args.even)
        state = tracker.init(frame, target_pos, target_sz, self.model)
        
        self.trackers.append(tracker)
        self.states.append(state)
        
        color = tuple(np.random.randint(50, 255, 3).tolist())
        self.colors.append(color)
        
        obj_id = self.next_id
        self.next_id += 1
        
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
    
    if args.video_file:
        cap = cv2.VideoCapture(args.video_file)
        print(f'Opened video file: {args.video_file}')
    else:
        cap = cv2.VideoCapture(args.camera)
        print(f'Opened camera device: {args.camera}')
    
    if not cap.isOpened():
        print("Error: Cannot open camera/video")
        return
    
    window_name = 'Multi-Object Tracker'
    cv2.namedWindow(window_name)
    
    bbox_selector = BBoxSelector()
    cv2.setMouseCallback(window_name, bbox_selector.mouse_callback)
    
    mode = 'select'
    paused = False
    
    print("\n" + "="*50)
    print("Multi-Object Tracking Instructions")
    print("="*50)
    print("\nSELECTION MODE:")
    print("  • Click and drag to draw boxes around objects")
    print("  • Press SPACE to start tracking")
    print("  • Press 'c' to clear all selections")
    print("\nTRACKING MODE:")
    print("  • Press 'p' to pause/resume")
    print("  • Press 'r' to reset and select new objects")
    print("  • Press 'q' to quit")
    print("="*50 + "\n")
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame")
        return
    
    selection_frame = frame.copy()
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
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video/camera stream")
                    break
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                with torch.no_grad():
                    tracked_boxes = multi_tracker.track_all(rgb_frame)
                
                display_frame = draw_boxes(frame, tracked_boxes, multi_tracker.colors)
                
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
                cv2.putText(display_frame, "P: Pause | R: Reset | Q: Quit", 
                           (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow(window_name, display_frame)
            else:
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
            ret, frame = cap.read()
            if ret:
                selection_frame = frame.copy()
            print("\nReset to selection mode\n")
        elif key == ord('p') and mode == 'track':
            paused = not paused
            print("Paused" if paused else "Resumed")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nTracking ended")


if __name__ == '__main__':
    main()
