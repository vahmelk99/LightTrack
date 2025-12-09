"""
Multi-Object Tracker Wrapper
Simple API for multi-object tracking using LightTrack
"""

import sys
import os
import cv2
import torch
import numpy as np
from easydict import EasyDict as edict

# Add parent directory to path to import lib modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import lib.models.models as models
from lib.utils.utils import load_pretrain

from mot_tracker import MultiObjectTracker


class MOT:
    """
    Multi-Object Tracker Wrapper Class
    
    Simple API for tracking multiple objects in video frames.
    """
    
    def __init__(self, model_path, arch='LightTrackM_Subnet', 
                 path_name='back_04502514044521042540+cls_211000022+reg_100000111_ops_32',
                 stride=16, dataset='VOT2019', device='cuda'):
        """
        Initialize the Multi-Object Tracker
        
        Args:
            model_path: Path to the pretrained model weights
            arch: Model architecture name (default: 'LightTrackM_Subnet')
            path_name: Model architecture path configuration
            stride: Network stride (default: 16)
            dataset: Dataset name for configuration (default: 'VOT2019')
            device: Device to run on ('cuda' or 'cpu', default: 'cuda')
        """
        self.device = device
        self.model_path = model_path
        
        # Setup model info
        self.siam_info = edict()
        self.siam_info.arch = arch
        self.siam_info.dataset = dataset
        self.siam_info.epoch_test = False
        self.siam_info.stride = stride
        
        # Setup args
        self.args = edict()
        self.args.even = 0
        self.args.arch = arch
        self.args.path_name = path_name
        self.args.stride = stride
        
        # Build and load model
        print(f'Loading model from {model_path}...')
        if path_name != 'NULL':
            self.siam_net = models.__dict__[arch](path_name, stride=stride)
        else:
            self.siam_net = models.__dict__[arch](stride=stride)
        
        self.siam_net = load_pretrain(self.siam_net, model_path)
        self.siam_net.eval()
        
        if device == 'cuda':
            if not torch.cuda.is_available():
                print("Warning: CUDA not available, falling back to CPU")
                self.device = 'cpu'
            else:
                self.siam_net = self.siam_net.cuda()
        
        print('Model loaded successfully!')
        
        # Initialize multi-object tracker
        self.multi_tracker = MultiObjectTracker(self.siam_info, self.siam_net, self.args, device=self.device)
    
    def track(self, frame):
        """
        Track all objects in the current frame
        
        Args:
            frame: Input frame (BGR format from OpenCV or RGB numpy array)
                   Shape: (H, W, 3), dtype: uint8
        
        Returns:
            List of tuples: [(id, box), (id, box), ...]
            where box is [x1, y1, x2, y2] in pixel coordinates
        """
        # Convert BGR to RGB if needed (assume BGR from OpenCV)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame
        
        # Track all objects
        with torch.no_grad():
            results = self.multi_tracker.track_all(rgb_frame)
        
        return results
    
    def add_box(self, frame, box):
        """
        Add a new object to track
        
        Args:
            frame: Input frame (BGR format from OpenCV or RGB numpy array)
                   Shape: (H, W, 3), dtype: uint8
            box: Bounding box in format [x1, y1, x2, y2]
        
        Returns:
            int: ID assigned to the new object
        """
        # Convert BGR to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame
        
        # Add object
        with torch.no_grad():
            obj_id = self.multi_tracker.add_object(rgb_frame, box)
        
        return obj_id
    
    def remove_box(self, obj_id):
        """
        Remove an object from tracking
        
        Args:
            obj_id: ID of the object to remove
        
        Returns:
            bool: True if successful, False if ID not found
        """
        return self.multi_tracker.remove_object(obj_id)
    
    def clear_all(self):
        """Remove all tracked objects"""
        self.multi_tracker.clear_all()
    
    def get_num_objects(self):
        """
        Get the number of currently tracked objects
        
        Returns:
            int: Number of tracked objects
        """
        return self.multi_tracker.get_num_objects()
    
    def get_available_ids(self):
        """
        Get list of available (reusable) IDs
        
        Returns:
            list: List of available IDs
        """
        return self.multi_tracker.available_ids.copy()
