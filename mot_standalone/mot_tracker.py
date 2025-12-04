"""
Multi-Object Tracker - Main Implementation
Self-contained LightTrack-based tracker with all dependencies included.
"""

import os
import cv2
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from collections import namedtuple


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_yaml(path, subset=True):
    """Load YAML configuration file"""
    file = open(path, 'r')
    yaml_obj = yaml.load(file.read(), Loader=yaml.FullLoader)
    if subset:
        hp = yaml_obj['TEST']
    else:
        hp = yaml_obj
    return hp


def to_torch(ndarray):
    """Convert numpy array to torch tensor"""
    return torch.from_numpy(ndarray)


def im_to_torch(img):
    """Convert image to torch tensor (C, H, W)"""
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    return img


def python2round(f):
    """Use python2 round function in python3"""
    if round(f + 1) - round(f) != 1:
        return f + abs(f) / f * 0.5
    return round(f)


def cxy_wh_2_rect(pos, sz):
    """Convert center position and size to rectangle [x, y, w, h]"""
    return [float(max(float(0), pos[0] - sz[0] / 2)), 
            float(max(float(0), pos[1] - sz[1] / 2)), 
            float(sz[0]), float(sz[1])]


def get_axis_aligned_bbox(region):
    """Get axis-aligned bounding box from region"""
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x + w / 2
        cy = y + h / 2
    return cx, cy, w, h


def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans, out_mode='torch'):
    """
    SiamFC type cropping
    Extract a subwindow from the image for tracking
    """
    crop_info = dict()
    
    if isinstance(pos, float):
        pos = [pos, pos]
    
    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    context_xmin = round(pos[0] - c)
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))
    
    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad
    
    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
        tete_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad))
        
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), 
                                   int(context_xmin):int(context_xmax + 1), :]
    else:
        tete_im = np.zeros(im.shape[0:2])
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), 
                                int(context_xmin):int(context_xmax + 1), :]
    
    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    else:
        im_patch = im_patch_original
    
    crop_info['crop_cords'] = [context_xmin, context_xmax, context_ymin, context_ymax]
    crop_info['empty_mask'] = tete_im
    crop_info['pad_info'] = [top_pad, left_pad, r, c]
    
    if out_mode == "torch":
        return im_to_torch(im_patch.copy()), crop_info
    else:
        return im_patch, crop_info


# ============================================================================
# TRACKER CONFIGURATION
# ============================================================================

class Config(object):
    """Tracker configuration parameters"""
    def __init__(self, stride=8, even=0):
        self.penalty_k = 0.062
        self.window_influence = 0.38
        self.lr = 0.765
        self.windowing = 'cosine'
        if even:
            self.exemplar_size = 128
            self.instance_size = 256
        else:
            self.exemplar_size = 127
            self.instance_size = 255
        self.total_stride = stride
        self.score_size = int(round(self.instance_size / self.total_stride))
        self.context_amount = 0.5
        self.ratio = 0.94
    
    def update(self, newparam=None):
        if newparam:
            for key, value in newparam.items():
                setattr(self, key, value)
            self.renew()
    
    def renew(self):
        self.score_size = int(round(self.instance_size / self.total_stride))


# ============================================================================
# LIGHTTRACK SINGLE OBJECT TRACKER
# ============================================================================

class Lighttrack(object):
    """Single object tracker using LightTrack"""
    def __init__(self, info, even=0, device='cuda'):
        super(Lighttrack, self).__init__()
        self.info = info
        self.stride = info.stride
        self.even = even
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    def normalize(self, x):
        """Normalize input tensor (C, H, W)"""
        x /= 255
        x -= self.mean
        x /= self.std
        return x
    
    def init(self, im, target_pos, target_sz, model):
        """Initialize tracker with first frame"""
        state = dict()
        
        p = Config(stride=self.stride, even=self.even)
        
        state['im_h'] = im.shape[0]
        state['im_w'] = im.shape[1]
        
        # Load hyper-parameters from the yaml file
        prefix = [x for x in ['OTB', 'VOT'] if x in self.info.dataset]
        if len(prefix) == 0:
            prefix = [self.info.dataset]
        yaml_path = os.path.join(os.path.dirname(__file__), 
                                 '../experiments/test/%s/' % prefix[0], 'LightTrack.yaml')
        cfg = load_yaml(yaml_path)
        cfg_benchmark = cfg[self.info.dataset]
        p.update(cfg_benchmark)
        p.renew()
        
        if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
            p.instance_size = cfg_benchmark['big_sz']
            p.renew()
        else:
            p.instance_size = cfg_benchmark['small_sz']
            p.renew()
        
        self.grids(p)
        
        net = model
        
        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))
        
        avg_chans = np.mean(im, axis=(0, 1))
        z_crop, _ = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)
        z_crop = self.normalize(z_crop)
        z = z_crop.unsqueeze(0)
        if self.device == 'cuda':
            z = z.cuda()
        net.template(z)
        
        if p.windowing == 'cosine':
            window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
        elif p.windowing == 'uniform':
            window = np.ones(int(p.score_size), int(p.score_size))
        else:
            raise ValueError("Unsupported window type")
        
        state['p'] = p
        state['net'] = net
        state['avg_chans'] = avg_chans
        state['window'] = window
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        
        return state
    
    def update(self, net, x_crops, target_pos, target_sz, window, scale_z, p, debug=False):
        """Update tracker with new frame"""
        cls_score, bbox_pred = net.track(x_crops)
        cls_score = F.sigmoid(cls_score).squeeze().cpu().data.numpy()
        
        # bbox to real predict
        bbox_pred = bbox_pred.squeeze().cpu().data.numpy()
        
        pred_x1 = self.grid_to_search_x - bbox_pred[0, ...]
        pred_y1 = self.grid_to_search_y - bbox_pred[1, ...]
        pred_x2 = self.grid_to_search_x + bbox_pred[2, ...]
        pred_y2 = self.grid_to_search_y + bbox_pred[3, ...]
        
        # size penalty
        s_c = self.change(self.sz(pred_x2 - pred_x1, pred_y2 - pred_y1) / (self.sz_wh(target_sz)))
        r_c = self.change((target_sz[0] / target_sz[1]) / ((pred_x2 - pred_x1) / (pred_y2 - pred_y1)))
        
        penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)
        pscore = penalty * cls_score
        
        # window penalty
        pscore = pscore * (1 - p.window_influence) + window * p.window_influence
        
        # get max
        r_max, c_max = np.unravel_index(pscore.argmax(), pscore.shape)
        
        # to real size
        pred_x1 = pred_x1[r_max, c_max]
        pred_y1 = pred_y1[r_max, c_max]
        pred_x2 = pred_x2[r_max, c_max]
        pred_y2 = pred_y2[r_max, c_max]
        
        pred_xs = (pred_x1 + pred_x2) / 2
        pred_ys = (pred_y1 + pred_y2) / 2
        pred_w = pred_x2 - pred_x1
        pred_h = pred_y2 - pred_y1
        
        diff_xs = pred_xs - p.instance_size // 2
        diff_ys = pred_ys - p.instance_size // 2
        
        diff_xs, diff_ys, pred_w, pred_h = diff_xs / scale_z, diff_ys / scale_z, pred_w / scale_z, pred_h / scale_z
        
        target_sz = target_sz / scale_z
        
        # size learning rate
        lr = penalty[r_max, c_max] * cls_score[r_max, c_max] * p.lr
        
        # size rate
        res_xs = target_pos[0] + diff_xs
        res_ys = target_pos[1] + diff_ys
        res_w = pred_w * lr + (1 - lr) * target_sz[0]
        res_h = pred_h * lr + (1 - lr) * target_sz[1]
        
        target_pos = np.array([res_xs, res_ys])
        target_sz = target_sz * (1 - lr) + lr * np.array([res_w, res_h])
        
        if debug:
            return target_pos, target_sz, cls_score[r_max, c_max], cls_score
        else:
            return target_pos, target_sz, cls_score[r_max, c_max]
    
    def track(self, state, im):
        """Track object in new frame"""
        p = state['p']
        net = state['net']
        avg_chans = state['avg_chans']
        window = state['window']
        target_pos = state['target_pos']
        target_sz = state['target_sz']
        
        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = p.exemplar_size / s_z
        d_search = (p.instance_size - p.exemplar_size) / 2
        pad = d_search / scale_z
        s_x = s_z + 2 * pad
        
        x_crop, _ = get_subwindow_tracking(im, target_pos, p.instance_size, 
                                           python2round(s_x), avg_chans)
        state['x_crop'] = x_crop.clone()
        x_crop = self.normalize(x_crop)
        x_crop = x_crop.unsqueeze(0)
        if self.device == 'cuda':
            x_crop = x_crop.cuda()
        
        debug = True
        if debug:
            target_pos, target_sz, _, cls_score = self.update(net, x_crop, target_pos, 
                                                              target_sz * scale_z, window, scale_z, p, debug=debug)
            state['cls_score'] = cls_score
        else:
            target_pos, target_sz, _ = self.update(net, x_crop, target_pos, 
                                                   target_sz * scale_z, window, scale_z, p, debug=debug)
        
        target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
        target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
        target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
        target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        state['p'] = p
        
        return state
    
    def grids(self, p):
        """Generate grid for feature map"""
        sz = p.score_size
        sz_x = sz // 2
        sz_y = sz // 2
        
        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))
        
        self.grid_to_search_x = x * p.total_stride + p.instance_size // 2
        self.grid_to_search_y = y * p.total_stride + p.instance_size // 2
    
    def change(self, r):
        return np.maximum(r, 1. / r)
    
    def sz(self, w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)
    
    def sz_wh(self, wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)


# ============================================================================
# MULTI-OBJECT TRACKER
# ============================================================================

class MultiObjectTracker:
    """Multi-object tracker managing multiple LightTrack instances"""
    def __init__(self, siam_info, model, args, device='cuda'):
        self.trackers = []
        self.states = []
        self.model = model
        self.siam_info = siam_info
        self.args = args
        self.device = device
        self.next_id = 0
        self.available_ids = []
    
    def add_object(self, frame, bbox):
        """
        Add a new object to track
        Args:
            frame: RGB image
            bbox: (x1, y1, x2, y2) bounding box
        Returns:
            obj_id: assigned object ID
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        
        target_pos = np.array([cx, cy])
        target_sz = np.array([w, h])
        
        # Create new tracker instance
        tracker = Lighttrack(self.siam_info, even=self.args.even, device=self.device)
        
        # Initialize tracker
        state = tracker.init(frame, target_pos, target_sz, self.model)
        
        # Reuse available ID if any, otherwise use next_id
        if self.available_ids:
            obj_id = min(self.available_ids)
            self.available_ids.remove(obj_id)
        else:
            obj_id = self.next_id
            self.next_id += 1
        
        # Store the ID in the state (array position doesn't matter)
        state["id"] = obj_id
        
        # Always append to the end (position in array is independent of ID)
        self.trackers.append(tracker)
        self.states.append(state)
        
        return obj_id
    
    def track_all(self, frame):
        """
        Track all objects in the current frame
        Args:
            frame: RGB image
        Returns:
            List of tuples [(id, bbox), ...] where bbox is [x1, y1, x2, y2]
        """
        results = []
        
        for i, (tracker, state) in enumerate(zip(self.trackers, self.states)):
            # Update tracker
            state = tracker.track(state, frame)
            self.states[i] = state
            
            # Get bounding box in [x, y, w, h] format
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            
            # Convert to [x1, y1, x2, y2] format
            x, y, w, h = location
            bbox = [x, y, x + w, y + h]
            
            results.append((state["id"], bbox))
        
        return results
    
    def remove_object(self, obj_id):
        """
        Remove an object from tracking by its ID (not array position)
        Args:
            obj_id: ID of object to remove
        Returns:
            bool: True if successful, False otherwise
        """
        # Find the object with this ID
        for i, state in enumerate(self.states):
            if state.get("id") == obj_id:
                # Found it - remove from both lists
                del self.trackers[i]
                del self.states[i]
                
                # Add this ID to available IDs for reuse
                if obj_id not in self.available_ids:
                    self.available_ids.append(obj_id)
                    self.available_ids.sort()
                
                return True
        
        # ID not found
        return False
    
    def clear_all(self):
        """Clear all tracked objects"""
        self.trackers.clear()
        self.states.clear()
        self.next_id = 0
        self.available_ids.clear()
    
    def get_num_objects(self):
        """Get number of currently tracked objects"""
        return len(self.trackers)
