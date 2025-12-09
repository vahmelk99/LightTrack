"""
Multi-Object Tracker Wrapper - Standalone Version
All dependencies embedded, no external imports from lib.
"""

import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import warnings
import logging
from collections import OrderedDict, namedtuple
from easydict import EasyDict as edict

from mot_standalone.mot_tracker import MultiObjectTracker

warnings.filterwarnings('ignore')


# ============================================================================
# EMBEDDED MODEL ARCHITECTURE CODE
# All code below is from lib/models and lib/utils to make this standalone
# ============================================================================



# ===== FROM lib/models/backbone/models/utils.py =====

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Tuple, Optional, List
try:
    from torch._six import container_abcs
except ImportError:
    import collections.abc as container_abcs
from itertools import repeat

def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return

tup_pair = _ntuple(2)

def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)

def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1), value: float = 0):
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return x

def conv2d_same(
        x, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0), dilation: Tuple[int, int] = (1, 1), groups: int = 1):
    x = pad_same(x, weight.shape[-2:], stride, dilation)
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)

def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

def is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0

def get_padding_value(padding, kernel_size, **kwargs) -> Tuple[Tuple, bool]:
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == 'same':
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = get_padding(kernel_size, **kwargs)
            else:
                # dynamic 'SAME' padding, has runtime/GPU memory overhead
                padding = 0
                dynamic = True
        elif padding == 'valid':
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic

def get_condconv_initializer(initializer, num_experts, expert_shape):
    def condconv_initializer(weight):
        """CondConv initializer function."""
        num_params = np.prod(expert_shape)
        if (len(weight.shape) != 2 or weight.shape[0] != num_experts or
                weight.shape[1] != num_params):
            raise (ValueError(
                'CondConv variables must have shape [num_experts, num_params]'))
        for i in range(num_experts):
            initializer(weight[i].view(expert_shape))

    return condconv_initializer

def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def _parse_ksize(ss):
    if ss.isdigit():
        return int(ss)
    else:
        return [int(k) for k in ss.split('.')]

def resolve_bn_args(kwargs):
    bn_args = {}
    bn_momentum = kwargs.pop('bn_momentum', None)
    if bn_momentum is not None:
        bn_args['momentum'] = bn_momentum
    bn_eps = kwargs.pop('bn_eps', None)
    if bn_eps is not None:
        bn_args['eps'] = bn_eps
    return bn_args

def round_channels(channels, multiplier=1.0, divisor=8, channel_min=None):
    """Round number of filters based on depth multiplier."""
    if not multiplier:
        return channels
    channels *= multiplier
    return make_divisible(channels, divisor, channel_min)

def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)

def create_conv2d(in_chs, out_chs, kernel_size, **kwargs):
    """ Select a 2d convolution implementation based on arguments
    Creates and returns one of torch.nn.Conv2d, Conv2dSame, MixedConv2d, or CondConv2d.
    Used extensively by EfficientNet, MobileNetv3 and related networks.
    """
    assert 'groups' not in kwargs  # only use 'depthwise' bool arg
    depthwise = kwargs.pop('depthwise', False)
    groups = out_chs if depthwise else 1
    m = create_conv2d_pad(in_chs, out_chs, kernel_size, groups=groups, **kwargs)
    return m


# ===== FROM lib/models/backbone/models/units.py =====

import torch.nn as nn

from functools import partial
# from .utils import * - already embedded above

def swish(x, inplace: bool = False):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    """
    return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())

class Swish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return swish(x, self.inplace)

def sigmoid(x, inplace: bool = False):
    return x.sigmoid_() if inplace else x.sigmoid()

_SE_ARGS_DEFAULT = dict(
    gate_fn=sigmoid,
    act_layer=None,
    reduce_mid=False,
    divisor=1)

def resolve_se_args(kwargs, in_chs, act_layer=None):
    se_kwargs = kwargs.copy() if kwargs is not None else {}
    # fill in args that aren't specified with the defaults
    for k, v in _SE_ARGS_DEFAULT.items():
        se_kwargs.setdefault(k, v)
    # some models, like MobilNetV3, calculate SE reduction chs from the containing block's mid_ch instead of in_ch
    if not se_kwargs.pop('reduce_mid'):
        se_kwargs['reduced_base_chs'] = in_chs
    # act_layer override, if it remains None, the containing block's act_layer will be used
    if se_kwargs['act_layer'] is None:
        assert act_layer is not None
        se_kwargs['act_layer'] = act_layer
    return se_kwargs

class Sigmoid(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.sigmoid_() if self.inplace else x.sigmoid()

class DepthwiseSeparableConv(nn.Module):
    """ DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
    """

    def __init__(self, in_chs, out_chs, dw_kernel_size=3,
                 stride=1, dilation=1, pad_type='', act_layer=nn.ReLU, noskip=False,
                 pw_kernel_size=1, pw_act=False, se_ratio=0., se_kwargs=None,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, drop_path_rate=0.):
        super(DepthwiseSeparableConv, self).__init__()
        norm_kwargs = norm_kwargs or {}
        has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act  # activation after point-wise conv
        self.drop_path_rate = drop_path_rate

        self.conv_dw = create_conv2d(
            in_chs, in_chs, dw_kernel_size, stride=stride, dilation=dilation, padding=pad_type, depthwise=True)
        self.bn1 = norm_layer(in_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

        # Squeeze-and-excitation
        if has_se:
            se_kwargs = resolve_se_args(se_kwargs, in_chs, act_layer)
            self.se = SqueezeExcite(in_chs, se_ratio=se_ratio, **se_kwargs)
        else:
            self.se = None

        self.conv_pw = create_conv2d(in_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = norm_layer(out_chs, **norm_kwargs)
        self.act2 = act_layer(inplace=True) if self.has_pw_act else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':
            # no expansion in this block, use depthwise, before SE
            info = dict(module='act1', hook_type='forward', num_chs=self.conv_pw.in_channels)
        elif location == 'depthwise':  # after SE
            info = dict(module='conv_pw', hook_type='forward_pre', num_chs=self.conv_pw.in_channels)
        else:  # location == 'bottleneck'
            info = dict(module='', hook_type='', num_chs=self.conv_pw.out_channels)
        return info

    def forward(self, x):
        residual = x

        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act1(x)

        if self.se is not None:
            x = self.se(x)

        x = self.conv_pw(x)
        x = self.bn2(x)
        x = self.act2(x)

        if self.has_residual:
            x += residual
        return x

class CondConv2d(nn.Module):
    """ Conditionally Parameterized Convolution
    Inspired by: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/condconv/condconv_layers.py
    Grouped convolution hackery for parallel execution of the per-sample kernel filters inspired by this discussion:
    https://github.com/pytorch/pytorch/issues/17983
    """
    __constants__ = ['bias', 'in_channels', 'out_channels', 'dynamic_padding']

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding='', dilation=1, groups=1, bias=False, num_experts=4):
        super(CondConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = tup_pair(kernel_size)
        self.stride = tup_pair(stride)
        padding_val, is_padding_dynamic = get_padding_value(
            padding, kernel_size, stride=stride, dilation=dilation)
        self.dynamic_padding = is_padding_dynamic  # if in forward to work with torchscript
        self.padding = tup_pair(padding_val)
        self.dilation = tup_pair(dilation)
        self.groups = groups
        self.num_experts = num_experts

        self.weight_shape = (self.out_channels, self.in_channels // self.groups) + self.kernel_size
        weight_num_param = 1
        for wd in self.weight_shape:
            weight_num_param *= wd
        self.weight = torch.nn.Parameter(torch.Tensor(self.num_experts, weight_num_param))

        if bias:
            self.bias_shape = (self.out_channels,)
            self.bias = torch.nn.Parameter(torch.Tensor(self.num_experts, self.out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        init_weight = get_condconv_initializer(
            partial(nn.init.kaiming_uniform_, a=math.sqrt(5)), self.num_experts, self.weight_shape)
        init_weight(self.weight)
        if self.bias is not None:
            fan_in = np.prod(self.weight_shape[1:])
            bound = 1 / math.sqrt(fan_in)
            init_bias = get_condconv_initializer(
                partial(nn.init.uniform_, a=-bound, b=bound), self.num_experts, self.bias_shape)
            init_bias(self.bias)

    def forward(self, x, routing_weights):
        B, C, H, W = x.shape
        weight = torch.matmul(routing_weights, self.weight)
        new_weight_shape = (B * self.out_channels, self.in_channels // self.groups) + self.kernel_size
        weight = weight.view(new_weight_shape)
        bias = None
        if self.bias is not None:
            bias = torch.matmul(routing_weights, self.bias)
            bias = bias.view(B * self.out_channels)
        # move batch elements with channels so each batch element can be efficiently convolved with separate kernel
        x = x.view(1, B * C, H, W)
        if self.dynamic_padding:
            out = conv2d_same(
                x, weight, bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * B)
        else:
            out = F.conv2d(
                x, weight, bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * B)
        out = out.permute([1, 0, 2, 3]).view(B, self.out_channels, out.shape[-2], out.shape[-1])

        # Literal port (from TF definition)
        # x = torch.split(x, 1, 0)
        # weight = torch.split(weight, 1, 0)
        # if self.bias is not None:
        #     bias = torch.matmul(routing_weights, self.bias)
        #     bias = torch.split(bias, 1, 0)
        # else:
        #     bias = [None] * B
        # out = []
        # for xi, wi, bi in zip(x, weight, bias):
        #     wi = wi.view(*self.weight_shape)
        #     if bi is not None:
        #         bi = bi.view(*self.bias_shape)
        #     out.append(self.conv_fn(
        #         xi, wi, bi, stride=self.stride, padding=self.padding,
        #         dilation=self.dilation, groups=self.groups))
        # out = torch.cat(out, 0)
        return out

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=sigmoid, divisor=1, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x

class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, dilation=1, pad_type='', act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(ConvBnAct, self).__init__()
        norm_kwargs = norm_kwargs or {}
        self.conv = create_conv2d(in_chs, out_chs, kernel_size, stride=stride, dilation=dilation, padding=pad_type)
        self.bn1 = norm_layer(out_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

    def feature_info(self, location):
        if location == 'expansion' or location == 'depthwise':
            # no expansion or depthwise this block, use act after conv
            info = dict(module='act1', hook_type='forward', num_chs=self.conv.out_channels)
        else:  # location == 'bottleneck'
            info = dict(module='', hook_type='', num_chs=self.conv.out_channels)
        return info

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x

class InvertedResidual(nn.Module):
    """ Inverted residual block w/ optional SE and CondConv routing"""

    def __init__(self, in_chs, out_chs, dw_kernel_size=3,
                 stride=1, dilation=1, pad_type='', act_layer=nn.ReLU, noskip=False,
                 exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1,
                 se_ratio=0., se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None,
                 conv_kwargs=None, drop_path_rate=0.):
        super(InvertedResidual, self).__init__()
        norm_kwargs = norm_kwargs or {}
        conv_kwargs = conv_kwargs or {}
        mid_chs = make_divisible(in_chs * exp_ratio)
        has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.drop_path_rate = drop_path_rate

        # Point-wise expansion
        self.conv_pw = create_conv2d(in_chs, mid_chs, exp_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn1 = norm_layer(mid_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

        # Depth-wise convolution
        self.conv_dw = create_conv2d(
            mid_chs, mid_chs, dw_kernel_size, stride=stride, dilation=dilation,
            padding=pad_type, depthwise=True, **conv_kwargs)
        self.bn2 = norm_layer(mid_chs, **norm_kwargs)
        self.act2 = act_layer(inplace=True)

        # Squeeze-and-excitation
        if has_se:
            se_kwargs = resolve_se_args(se_kwargs, in_chs, act_layer)
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio, **se_kwargs)
        else:
            self.se = None

        # Point-wise linear projection
        self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn3 = norm_layer(out_chs, **norm_kwargs)

    def feature_info(self, location):
        if location == 'expansion':
            info = dict(module='act1', hook_type='forward', num_chs=self.conv_pw.in_channels)
        elif location == 'depthwise':  # after SE
            info = dict(module='conv_pwl', hook_type='forward_pre', num_chs=self.conv_pwl.in_channels)
        else:  # location == 'bottleneck'
            info = dict(module='', hook_type='', num_chs=self.conv_pwl.out_channels)
        return info

    def forward(self, x):
        residual = x

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.has_residual:
            x += residual

        return x

def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

class HardSigmoid(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_sigmoid(x, self.inplace)

class SelectAdaptivePool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """

    def __init__(self, output_size=1, pool_type='avg', flatten=False):
        super(SelectAdaptivePool2d, self).__init__()
        self.output_size = output_size
        self.pool_type = pool_type
        self.flatten = flatten
        self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        x = self.pool(x)
        if self.flatten:
            x = x.flatten(1)
        return x

    def feat_mult(self):
        return 1

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + 'output_size=' + str(self.output_size) \
               + ', pool_type=' + self.pool_type + ')'


# ===== FROM lib/models/backbone/models/resunit.py =====

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, expansion=4):
        super(Bottleneck, self).__init__()
        planes = int(planes / expansion)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * expansion, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes * expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.expansion = expansion
        if inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * self.expansion),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def get_Bottleneck(in_c, out_c, stride):
    return Bottleneck(in_c, out_c, stride=stride)

def get_BasicBlock(in_c, out_c, stride):
    return BasicBlock(in_c, out_c, stride=stride)


# ===== FROM lib/models/backbone/models/model.py =====

# from .builder import * - already embedded above
# from .units import * - already embedded above

DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }

_DEBUG = False

class ChildNet(nn.Module):

    def __init__(self, block_args, num_classes=1000, in_chans=3, stem_size=16, num_features=1280, head_bias=True,
                 channel_multiplier=1.0, pad_type='', act_layer=nn.ReLU, drop_rate=0., drop_path_rate=0.,
                 se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None, global_pool='avg', pool_bn=False,
                 zero_gamma=False):
        super(ChildNet, self).__init__()

        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate
        self._in_chs = in_chans
        self.pool_bn = pool_bn

        # Stem
        stem_size = round_channels(stem_size, channel_multiplier)
        self.conv_stem = create_conv2d(self._in_chs, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        self._in_chs = stem_size

        # Middle stages (IR/ER/DS Blocks)
        builder = ChildNetBuilder(
            channel_multiplier, 8, None, 32, pad_type, act_layer, se_kwargs,
            norm_layer, norm_kwargs, drop_path_rate, verbose=_DEBUG)
        self.blocks = nn.Sequential(*builder(self._in_chs, block_args))
        # self.blocks = builder(self._in_chs, block_args)
        self._in_chs = builder.in_chs

        # Head + Pooling
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.conv_head = create_conv2d(self._in_chs, self.num_features, 1, padding=pad_type, bias=head_bias)
        self.act2 = act_layer(inplace=True)

        # Classifier
        self.classifier = nn.Linear(self.num_features * self.global_pool.feat_mult(), self.num_classes)

        if pool_bn:
            self.pool_bn = nn.BatchNorm1d(1)

        efficientnet_init_weights(self, zero_gamma=zero_gamma)
        self.strides = [2, 4, 8, 16, 16, 32, 32]

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.num_classes = num_classes
        self.classifier = nn.Linear(
            self.num_features * self.global_pool.feat_mult(), num_classes) if self.num_classes else None

    def forward_backbone(self, inp, stride=32, backbone_index=None):
        # architecture = [[0], [], [], [], [], [], [0]]
        x = self.conv_stem(inp)
        x = self.bn1(x)
        x = self.act1(x)
        for layer_idx, layer in enumerate(self.blocks):
            if backbone_index is not None:
                if layer_idx == backbone_index[0]:
                    for block_idx, block in enumerate(layer):
                        x = block(x)
                        if block_idx == backbone_index[1]:
                            break
            else:
                if self.strides[layer_idx] > stride:
                    break
                x = layer(x)
        return x

    def forward_features(self, x):
        # architecture = [[0], [], [], [], [], [0]]
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        return x

    def forward(self, x, stride=None, backbone_index=None):
        if stride == None:
            x = self.forward_features(x)
            x = x.flatten(1)
            if self.drop_rate > 0.:
                x = F.dropout(x, p=self.drop_rate, training=self.training)
            x = self.classifier(x)
            if self.pool_bn:
                x = torch.unsqueeze(x, 1)
                x = self.pool_bn(x)
                x = torch.squeeze(x)
            return x
        else:
            x = self.forward_backbone(x, stride=stride, backbone_index=backbone_index)
            return x

'''2020.10.18 Only keep the convolution features that we need'''

class ChildNet_FCN(nn.Module):

    def __init__(self, block_args, num_classes=1000, in_chans=3, stem_size=16, num_features=1280, head_bias=True,
                 channel_multiplier=1.0, pad_type='', act_layer=nn.ReLU, drop_rate=0., drop_path_rate=0.,
                 se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None, global_pool='avg', pool_bn=False,
                 zero_gamma=False):
        super(ChildNet_FCN, self).__init__()

        self._in_chs = in_chans

        # Stem
        stem_size = round_channels(stem_size, channel_multiplier)
        self.conv_stem = create_conv2d(self._in_chs, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        self._in_chs = stem_size

        # Middle stages (IR/ER/DS Blocks)
        builder = ChildNetBuilder(
            channel_multiplier, 8, None, 32, pad_type, act_layer, se_kwargs,
            norm_layer, norm_kwargs, drop_path_rate, verbose=_DEBUG)
        self.blocks = nn.Sequential(*builder(self._in_chs, block_args))
        # self.blocks = builder(self._in_chs, block_args)
        self._in_chs = builder.in_chs

        efficientnet_init_weights(self, zero_gamma=zero_gamma)
        self.strides = [2, 4, 8, 16, 16, 32, 32]

    def forward(self, x):
        # architecture = [[0], [], [], [], [], [], [0]]
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        return x

def modify_block_args(block_args, kernel_size, exp_ratio):
    # kernel_size: 3,5,7
    # exp_ratio: 4,6
    block_type = block_args['block_type']
    # each type of block has different valid arguments, fill accordingly
    if block_type == 'cn':
        block_args['kernel_size'] = kernel_size
    elif block_type == 'er':
        block_args['exp_kernel_size'] = kernel_size
    else:
        block_args['dw_kernel_size'] = kernel_size

    if block_type == 'ir' or block_type == 'er':
        block_args['exp_ratio'] = exp_ratio
    return block_args

def _gen_childnet(arch_list, arch_def, ops=None, **kwargs):
    # arch_list = [[0], [], [], [], [], [0]]
    choices = {'kernel_size': [3, 5, 7], 'exp_ratio': [4, 6]}
    choices_list = [[x, y] for x in choices['kernel_size'] for y in choices['exp_ratio']]

    num_features = 1280

    # act_layer = HardSwish
    act_layer = Swish

    new_arch = []
    output_flag = False
    # change to child arch_def
    for i, (layer_choice, layer_arch) in enumerate(zip(arch_list, arch_def)):
        if len(layer_arch) == 1:
            new_arch.append(layer_arch)
            continue
        else:
            new_layer = []
            for j, (block_choice, block_arch) in enumerate(zip(layer_choice, layer_arch)):
                kernel_size, exp_ratio = choices_list[block_choice]
                elements = block_arch.split('_')
                block_arch = block_arch.replace(elements[2], 'k{}'.format(str(kernel_size)))
                block_arch = block_arch.replace(elements[4], 'e{}'.format(str(exp_ratio)))
                new_layer.append(block_arch)
                if ops is not None:
                    if i == ops[0] + 1 and j == ops[1]:
                        output_flag = True
                        break
            new_arch.append(new_layer)
            if output_flag:
                break

    model_kwargs = dict(
        block_args=decode_arch_def(new_arch),
        num_features=num_features,
        stem_size=16,
        # channel_multiplier=channel_multiplier,
        norm_kwargs=resolve_bn_args(kwargs),
        act_layer=act_layer,
        se_kwargs=dict(act_layer=nn.ReLU, gate_fn=hard_sigmoid, reduce_mid=True, divisor=8),
        **kwargs,
    )
    if ops is not None:
        model = ChildNet_FCN(**model_kwargs)
    else:
        model = ChildNet(**model_kwargs)
    return model

# arch_list = [[0], [3, 2, 3, 3], [3, 2, 3, 1], [3, 0, 3, 2], [3, 3, 3, 3], [3, 3, 3, 3], [0]]
# model = _gen_childnet(arch_list, zero_gamma=True)


# ===== FROM lib/models/backbone/models/builder.py =====

import logging
import re
from collections.__init__ import OrderedDict
from copy import deepcopy
import torch.nn as nn
# from .utils import _parse_ksize - already embedded above
# from .units import * - already embedded above

def _decode_block_str(block_str):
    """ Decode block definition string
    Gets a list of block arg (dicts) through a string notation of arguments.
    E.g. ir_r2_k3_s2_e1_i32_o16_se0.25_noskip
    All args can exist in any order with the exception of the leading string which
    is assumed to indicate the block type.
    leading string - block type (
      ir = InvertedResidual, ds = DepthwiseSep, dsa = DeptwhiseSep with pw act, cn = ConvBnAct)
    r - number of repeat blocks,
    k - kernel size,
    s - strides (1-9),
    e - expansion ratio,
    c - output channels,
    se - squeeze/excitation ratio
    n - activation fn ('re', 'r6', 'hs', or 'sw')
    Args:
        block_str: a string representation of block arguments.
    Returns:
        A list of block args (dicts)
    Raises:
        ValueError: if the string def not properly specified (TODO)
    """
    assert isinstance(block_str, str)
    ops = block_str.split('_')
    block_type = ops[0]  # take the block type off the front
    ops = ops[1:]
    options = {}
    noskip = False
    for op in ops:
        # string options being checked on individual basis, combine if they grow
        if op == 'noskip':
            noskip = True
        elif op.startswith('n'):
            # activation fn
            key = op[0]
            v = op[1:]
            if v == 're':
                value = nn.ReLU
            elif v == 'r6':
                value = nn.ReLU6
            elif v == 'sw':
                value = Swish
            else:
                continue
            options[key] = value
        else:
            # all numeric options
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

    # if act_layer is None, the model default (passed to model init) will be used
    act_layer = options['n'] if 'n' in options else None
    exp_kernel_size = _parse_ksize(options['a']) if 'a' in options else 1
    pw_kernel_size = _parse_ksize(options['p']) if 'p' in options else 1
    fake_in_chs = int(options['fc']) if 'fc' in options else 0  # FIXME hack to deal with in_chs issue in TPU def

    num_repeat = int(options['r'])
    # each type of block has different valid arguments, fill accordingly
    if block_type == 'ir':
        block_args = dict(
            block_type=block_type,
            dw_kernel_size=_parse_ksize(options['k']),
            exp_kernel_size=exp_kernel_size,
            pw_kernel_size=pw_kernel_size,
            out_chs=int(options['c']),
            exp_ratio=float(options['e']),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=int(options['s']),
            act_layer=act_layer,
            noskip=noskip,
        )
        if 'cc' in options:
            block_args['num_experts'] = int(options['cc'])
    elif block_type == 'ds' or block_type == 'dsa':
        block_args = dict(
            block_type=block_type,
            dw_kernel_size=_parse_ksize(options['k']),
            pw_kernel_size=pw_kernel_size,
            out_chs=int(options['c']),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=int(options['s']),
            act_layer=act_layer,
            pw_act=block_type == 'dsa',
            noskip=block_type == 'dsa' or noskip,
        )
    elif block_type == 'cn':
        block_args = dict(
            block_type=block_type,
            kernel_size=int(options['k']),
            out_chs=int(options['c']),
            stride=int(options['s']),
            act_layer=act_layer,
        )
    else:
        assert False, 'Unknown block type (%s)' % block_type

    return block_args, num_repeat

def modify_block_args(block_args, kernel_size, exp_ratio):
    # kernel_size: 3,5,7
    # exp_ratio: 4,6
    block_type = block_args['block_type']
    # each type of block has different valid arguments, fill accordingly
    if block_type == 'cn':
        block_args['kernel_size'] = kernel_size
    elif block_type == 'er':
        block_args['exp_kernel_size'] = kernel_size
    else:
        block_args['dw_kernel_size'] = kernel_size

    if block_type == 'ir' or block_type == 'er':
        block_args['exp_ratio'] = exp_ratio
    return block_args

def _scale_stage_depth(stack_args, repeats, depth_multiplier=1.0, depth_trunc='ceil'):
    """ Per-stage depth scaling
    Scales the block repeats in each stage. This depth scaling impl maintains
    compatibility with the EfficientNet scaling method, while allowing sensible
    scaling for other models that may have multiple block arg definitions in each stage.
    """

    # We scale the total repeat count for each stage, there may be multiple
    # block arg defs per stage so we need to sum.
    num_repeat = sum(repeats)
    if depth_trunc == 'round':
        # Truncating to int by rounding allows stages with few repeats to remain
        # proportionally smaller for longer. This is a good choice when stage definitions
        # include single repeat stages that we'd prefer to keep that way as long as possible
        num_repeat_scaled = max(1, round(num_repeat * depth_multiplier))
    else:
        # The default for EfficientNet truncates repeats to int via 'ceil'.
        # Any multiplier > 1.0 will result in an increased depth for every stage.
        num_repeat_scaled = int(math.ceil(num_repeat * depth_multiplier))

    # Proportionally distribute repeat count scaling to each block definition in the stage.
    # Allocation is done in reverse as it results in the first block being less likely to be scaled.
    # The first block makes less sense to repeat in most of the arch definitions.
    repeats_scaled = []
    for r in repeats[::-1]:
        rs = max(1, round((r / num_repeat * num_repeat_scaled)))
        repeats_scaled.append(rs)
        num_repeat -= r
        num_repeat_scaled -= rs
    repeats_scaled = repeats_scaled[::-1]

    # Apply the calculated scaling to each block arg in the stage
    sa_scaled = []
    for ba, rep in zip(stack_args, repeats_scaled):
        sa_scaled.extend([deepcopy(ba) for _ in range(rep)])
    return sa_scaled

def decode_arch_def(arch_def, depth_multiplier=1.0, depth_trunc='ceil', experts_multiplier=1):
    arch_args = []
    for stack_idx, block_strings in enumerate(arch_def):
        assert isinstance(block_strings, list)
        stack_args = []
        repeats = []
        for block_str in block_strings:
            assert isinstance(block_str, str)
            ba, rep = _decode_block_str(block_str)
            if ba.get('num_experts', 0) > 0 and experts_multiplier > 1:
                ba['num_experts'] *= experts_multiplier
            stack_args.append(ba)
            repeats.append(rep)
        arch_args.append(_scale_stage_depth(stack_args, repeats, depth_multiplier, depth_trunc))
    return arch_args

class ChildNetBuilder:
    """ Build Trunk Blocks
    """

    def __init__(self, channel_multiplier=1.0, channel_divisor=8, channel_min=None,
                 output_stride=32, pad_type='', act_layer=None, se_kwargs=None,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, drop_path_rate=0., feature_location='',
                 verbose=False):
        self.channel_multiplier = channel_multiplier
        self.channel_divisor = channel_divisor
        self.channel_min = channel_min
        self.output_stride = output_stride
        self.pad_type = pad_type
        self.act_layer = act_layer
        self.se_kwargs = se_kwargs
        self.norm_layer = norm_layer
        self.norm_kwargs = norm_kwargs
        self.drop_path_rate = drop_path_rate
        self.feature_location = feature_location
        assert feature_location in ('pre_pwl', 'post_exp', '')
        self.verbose = verbose

        # state updated during build, consumed by model
        self.in_chs = None
        self.features = OrderedDict()

    def _round_channels(self, chs):
        return round_channels(chs, self.channel_multiplier, self.channel_divisor, self.channel_min)

    def _make_block(self, ba, block_idx, block_count):
        drop_path_rate = self.drop_path_rate * block_idx / block_count
        bt = ba.pop('block_type')
        ba['in_chs'] = self.in_chs
        ba['out_chs'] = self._round_channels(ba['out_chs'])
        if 'fake_in_chs' in ba and ba['fake_in_chs']:
            # FIXME this is a hack to work around mismatch in origin impl input filters
            ba['fake_in_chs'] = self._round_channels(ba['fake_in_chs'])
        ba['norm_layer'] = self.norm_layer
        ba['norm_kwargs'] = self.norm_kwargs
        ba['pad_type'] = self.pad_type
        # block act fn overrides the model default
        ba['act_layer'] = ba['act_layer'] if ba['act_layer'] is not None else self.act_layer
        assert ba['act_layer'] is not None
        if bt == 'ir':
            ba['drop_path_rate'] = drop_path_rate
            ba['se_kwargs'] = self.se_kwargs
            if self.verbose:
                logging.info('  InvertedResidual {}, Args: {}'.format(block_idx, str(ba)))
            block = InvertedResidual(**ba)
        elif bt == 'ds' or bt == 'dsa':
            ba['drop_path_rate'] = drop_path_rate
            ba['se_kwargs'] = self.se_kwargs
            if self.verbose:
                logging.info('  DepthwiseSeparable {}, Args: {}'.format(block_idx, str(ba)))
            block = DepthwiseSeparableConv(**ba)
        elif bt == 'cn':
            if self.verbose:
                logging.info('  ConvBnAct {}, Args: {}'.format(block_idx, str(ba)))
            block = ConvBnAct(**ba)
        else:
            assert False, 'Uknkown block type (%s) while building model.' % bt
        self.in_chs = ba['out_chs']  # update in_chs for arg of next block

        return block

    def __call__(self, in_chs, model_block_args):
        """ Build the blocks
        Args:
            in_chs: Number of input-channels passed to first block
            model_block_args: A list of lists, outer list defines stages, inner
                list contains strings defining block configuration(s)
        Return:
             List of block stacks (each stack wrapped in nn.Sequential)
        """
        if self.verbose:
            logging.info('Building model trunk with %d stages...' % len(model_block_args))
        self.in_chs = in_chs
        total_block_count = sum([len(x) for x in model_block_args])
        total_block_idx = 0
        current_stride = 2
        current_dilation = 1
        feature_idx = 0
        stages = []
        # outer list of block_args defines the stacks ('stages' by some conventions)
        for stage_idx, stage_block_args in enumerate(model_block_args):
            last_stack = stage_idx == (len(model_block_args) - 1)
            if self.verbose:
                logging.info('Stack: {}'.format(stage_idx))
            assert isinstance(stage_block_args, list)

            blocks = []
            # each stack (stage) contains a list of block arguments
            for block_idx, block_args in enumerate(stage_block_args):
                last_block = block_idx == (len(stage_block_args) - 1)
                extract_features = ''  # No features extracted
                if self.verbose:
                    logging.info(' Block: {}'.format(block_idx))

                # Sort out stride, dilation, and feature extraction details
                assert block_args['stride'] in (1, 2)
                if block_idx >= 1:
                    # only the first block in any stack can have a stride > 1
                    block_args['stride'] = 1

                do_extract = False
                if self.feature_location == 'pre_pwl':
                    if last_block:
                        next_stage_idx = stage_idx + 1
                        if next_stage_idx >= len(model_block_args):
                            do_extract = True
                        else:
                            do_extract = model_block_args[next_stage_idx][0]['stride'] > 1
                elif self.feature_location == 'post_exp':
                    if block_args['stride'] > 1 or (last_stack and last_block):
                        do_extract = True
                if do_extract:
                    extract_features = self.feature_location

                next_dilation = current_dilation
                if block_args['stride'] > 1:
                    next_output_stride = current_stride * block_args['stride']
                    if next_output_stride > self.output_stride:
                        next_dilation = current_dilation * block_args['stride']
                        block_args['stride'] = 1
                        if self.verbose:
                            logging.info('  Converting stride to dilation to maintain output_stride=={}'.format(
                                self.output_stride))
                    else:
                        current_stride = next_output_stride
                block_args['dilation'] = current_dilation
                if next_dilation != current_dilation:
                    current_dilation = next_dilation

                # create the block
                block = self._make_block(block_args, total_block_idx, total_block_count)
                blocks.append(block)

                # stash feature module name and channel info for model feature extraction
                if extract_features:
                    feature_module = block.feature_module(extract_features)
                    if feature_module:
                        feature_module = 'blocks.{}.{}.'.format(stage_idx, block_idx) + feature_module
                    feature_channels = block.feature_channels(extract_features)
                    self.features[feature_idx] = dict(
                        name=feature_module,
                        num_chs=feature_channels
                    )
                    feature_idx += 1

                total_block_idx += 1  # incr global block idx (across all stacks)
            stages.append(nn.Sequential(*blocks))
        return stages

def _init_weight_goog(m, n='', fix_group_fanout=True, last_bn=None):
    """ Weight initialization as per Tensorflow official implementations.
    Args:
        m (nn.Module): module to init
        n (str): module name
        fix_group_fanout (bool): enable correct (matching Tensorflow TPU impl) fanout calculation w/ group convs
    Handles layers in EfficientNet, EfficientNet-CondConv, MixNet, MnasNet, MobileNetV3, etc:
    * https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py
    * https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    """
    if isinstance(m, CondConv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        if fix_group_fanout:
            fan_out //= m.groups
        init_weight_fn = get_condconv_initializer(
            lambda w: w.data.normal_(0, math.sqrt(2.0 / fan_out)), m.num_experts, m.weight_shape)
        init_weight_fn(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        if fix_group_fanout:
            fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        if n in last_bn:
            m.weight.data.zero_()
            m.bias.data.zero_()
        else:
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)  # fan-out
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        m.weight.data.uniform_(-init_range, init_range)
        m.bias.data.zero_()

def efficientnet_init_weights(model: nn.Module, init_fn=None, zero_gamma=False):
    last_bn = []
    if zero_gamma:
        prev_n = ''
        for n, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                if ''.join(prev_n.split('.')[:-1]) != ''.join(n.split('.')[:-1]):
                    last_bn.append(prev_n)
                prev_n = n
        last_bn.append(prev_n)

    init_fn = init_fn or _init_weight_goog
    for n, m in model.named_modules():
        init_fn(m, n, last_bn=last_bn)


# ===== FROM lib/models/backbone/childnet.py =====

import warnings

warnings.filterwarnings('ignore')

import os
import logging
import torch
from collections import OrderedDict

def build_subnet(path_backbone, ops=None):
    arch_list = path_backbone

    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        # stage 1, 112x112 in
        ['ir_r1_k3_s2_e4_c24_se0.25', 'ir_r1_k3_s1_e4_c24_se0.25',
         'ir_r1_k3_s1_e4_c24_se0.25', 'ir_r1_k3_s1_e4_c24_se0.25'],
        # stage 2, 56x56 in
        ['ir_r1_k5_s2_e4_c40_se0.25', 'ir_r1_k5_s1_e4_c40_se0.25',
         'ir_r1_k5_s2_e4_c40_se0.25', 'ir_r1_k5_s2_e4_c40_se0.25'],
        # stage 3, 28x28 in
        ['ir_r1_k3_s2_e6_c80_se0.25', 'ir_r1_k3_s1_e4_c80_se0.25',
         'ir_r1_k3_s1_e4_c80_se0.25', 'ir_r1_k3_s1_e4_c80_se0.25'],
        # stage 4, 14x14in
        ['ir_r1_k3_s1_e6_c96_se0.25', 'ir_r1_k3_s1_e6_c96_se0.25',
         'ir_r1_k3_s1_e6_c96_se0.25', 'ir_r1_k3_s1_e6_c96_se0.25'],
        # stage 5, 14x14in
        ['ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s1_e6_c192_se0.25',
         'ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s2_e6_c192_se0.25'],
        # stage 6, 7x7 in
        ['cn_r1_k1_s1_c320_se0.25'],
    ]

    model = _gen_childnet(
        arch_list,
        arch_def,
        num_classes=1000,
        drop_rate=0,
        drop_path_rate=0,
        global_pool='avg',
        bn_momentum=None,
        bn_eps=None,
        pool_bn=False,
        zero_gamma=False,
        ops=ops)

    return model

def resume_checkpoint(model, checkpoint_path, ema=True):
    """2020.11.5 Modified from timm"""
    other_state = {}
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_name = 'state_dict_ema' if ema else 'state_dict'
        if isinstance(checkpoint, dict) and state_dict_name in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_name].items():
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            try:
                model.load_state_dict(new_state_dict, strict=True)
            except:
                model.load_state_dict(new_state_dict, strict=False)
                print('strict = %s' % False)
            if 'optimizer' in checkpoint:
                other_state['optimizer'] = checkpoint['optimizer']
            if 'amp' in checkpoint:
                other_state['amp'] = checkpoint['amp']
            if 'epoch' in checkpoint:
                resume_epoch = checkpoint['epoch']
                if 'version' in checkpoint and checkpoint['version'] > 1:
                    resume_epoch += 1  # start at the next epoch, old checkpoints incremented before save
            logging.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            model.load_state_dict(checkpoint)
            logging.info("Loaded checkpoint '{}'".format(checkpoint_path))
        return other_state, resume_epoch
    else:
        logging.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


# ===== FROM lib/models/super_connect.py =====

import torch
import torch.nn as nn
import torch.nn.functional as F

# from .connect import * - not needed, using super_connect below

class SeparableConv2d_BNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d_BNReLU, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        self.BN = nn.BatchNorm2d(out_channels)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        x = self.ReLU(self.BN(x))
        return x

class MC_BN(nn.Module):
    """2020.10.14 Batch Normalization with Multiple input Channels"""

    def __init__(self, inp_c=(40, 80, 96)):
        super(MC_BN, self).__init__()
        self.BN_z = nn.ModuleList()  # BN for the template branch
        self.BN_x = nn.ModuleList()  # BN for the search branch
        for idx, channel in enumerate(inp_c):
            self.BN_z.append(nn.BatchNorm2d(channel))
            self.BN_x.append(nn.BatchNorm2d(channel))

    def forward(self, kernel, search, index=None):
        if index is None:
            index = 0
        return self.BN_z[index](kernel), self.BN_x[index](search)

'''2020.10.09 Simplify prvious model'''

class Point_Neck_Mobile_simple(nn.Module):
    def __init__(self, inchannels=512, num_kernel=None, cat=False, BN_choice='before', matrix=True):
        super(Point_Neck_Mobile_simple, self).__init__()
        self.BN_choice = BN_choice
        if self.BN_choice == 'before':
            '''template and search use separate BN'''
            self.BN_adj_z = nn.BatchNorm2d(inchannels)
            self.BN_adj_x = nn.BatchNorm2d(inchannels)
        '''Point-wise Correlation'''
        self.pw_corr = PWCA(num_kernel, cat=cat, CA=True, matrix=matrix)

    def forward(self, kernel, search):
        """input: features of the template and the search region
           output: correlation features of cls and reg"""
        oup = {}
        if self.BN_choice == 'before':
            kernel, search = self.BN_adj_z(kernel), self.BN_adj_x(search)
        corr_feat = self.pw_corr([kernel], [search])
        oup['cls'], oup['reg'] = corr_feat, corr_feat
        return oup

'''2020.10.15 DP version'''

class Point_Neck_Mobile_simple_DP(nn.Module):
    def __init__(self, num_kernel_list=(256, 64), cat=False, matrix=True, adjust=True, adj_channel=128):
        super(Point_Neck_Mobile_simple_DP, self).__init__()
        self.adjust = adjust
        '''Point-wise Correlation & Adjust Layer (unify the num of channels)'''
        self.pw_corr = torch.nn.ModuleList()
        self.adj_layer = torch.nn.ModuleList()
        for num_kernel in num_kernel_list:
            self.pw_corr.append(PWCA(num_kernel, cat=cat, CA=True, matrix=matrix))
            self.adj_layer.append(nn.Conv2d(num_kernel, adj_channel, 1))

    def forward(self, kernel, search, stride_idx=None):
        """stride_idx: 0 or 1. 0 represents stride 8. 1 represents stride 16"""
        if stride_idx is None:
            stride_idx = -1
        oup = {}
        corr_feat = self.pw_corr[stride_idx]([kernel], [search])
        if self.adjust:
            corr_feat = self.adj_layer[stride_idx](corr_feat)
        oup['cls'], oup['reg'] = corr_feat, corr_feat
        return oup

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

'''2020.09.06 head supernet with mobile settings'''

class tower_supernet_singlechannel(nn.Module):
    """
    tower's supernet
    """

    def __init__(self, inchannels=256, outchannels=256, towernum=8,
                 base_op=SeparableConv2d_BNReLU, kernel_list=[3, 5, 0]):
        super(tower_supernet_singlechannel, self).__init__()
        if 0 in kernel_list:
            assert (kernel_list[-1] == 0)
        self.kernel_list = kernel_list
        self.num_choice = len(self.kernel_list)

        self.tower = nn.ModuleList()

        # tower
        for i in range(towernum):
            '''the first layer, we don't use identity'''
            if i == 0:
                op_list = nn.ModuleList()
                if self.num_choice == 1:
                    kernel_size = self.kernel_list[-1]
                    padding = (kernel_size - 1) // 2
                    op_list.append(base_op(inchannels, outchannels, kernel_size=kernel_size,
                                           stride=1, padding=padding))
                else:
                    for choice_idx in range(self.num_choice - 1):
                        kernel_size = self.kernel_list[choice_idx]
                        padding = (kernel_size - 1) // 2
                        op_list.append(base_op(inchannels, outchannels, kernel_size=kernel_size,
                                               stride=1, padding=padding))
                self.tower.append(op_list)

            else:
                op_list = nn.ModuleList()
                for choice_idx in range(self.num_choice):
                    kernel_size = self.kernel_list[choice_idx]
                    if kernel_size != 0:
                        padding = (kernel_size - 1) // 2
                        op_list.append(base_op(outchannels, outchannels, kernel_size=kernel_size,
                                               stride=1, padding=padding))
                    else:
                        op_list.append(Identity())
                self.tower.append(op_list)

    def forward(self, x, arch_list):

        for archs, arch_id in zip(self.tower, arch_list):
            x = archs[arch_id](x)

        return x

'''2020.09.06 the complete head supernet'''

class head_supernet(nn.Module):
    def __init__(self, channel_list=[112, 256, 512], kernel_list=[3, 5, 0], inchannels=64, towernum=8, linear_reg=False,
                 base_op_name='SeparableConv2d_BNReLU'):
        super(head_supernet, self).__init__()
        if base_op_name == 'SeparableConv2d_BNReLU':
            base_op = SeparableConv2d_BNReLU
        else:
            raise ValueError('Unsupported OP')
        self.num_cand = len(channel_list)
        self.cand_tower_cls = nn.ModuleList()
        self.cand_head_cls = nn.ModuleList()
        self.cand_tower_reg = nn.ModuleList()
        self.cand_head_reg = nn.ModuleList()
        self.tower_num = towernum
        # cls
        for outchannel in channel_list:
            self.cand_tower_cls.append(tower_supernet_singlechannel(inchannels=inchannels, outchannels=outchannel,
                                                                    towernum=towernum, base_op=base_op,
                                                                    kernel_list=kernel_list))
            self.cand_head_cls.append(cls_pred_head(inchannels=outchannel))
        # reg
        for outchannel in channel_list:
            self.cand_tower_reg.append(tower_supernet_singlechannel(inchannels=inchannels, outchannels=outchannel,
                                                                    towernum=towernum, base_op=base_op,
                                                                    kernel_list=kernel_list))
            self.cand_head_reg.append(reg_pred_head(inchannels=outchannel, linear_reg=linear_reg))

    def forward(self, inp, cand_dict=None):
        """cand_dict key: cls, reg
         [0/1/2, []]"""
        if cand_dict is None:
            cand_dict = {'cls': [0, [0] * self.tower_num], 'reg': [0, [0] * self.tower_num]}
        oup = {}
        # cls
        cand_list_cls = cand_dict['cls']  # [0/1/2, []]
        cls_feat = self.cand_tower_cls[cand_list_cls[0]](inp['cls'], cand_list_cls[1])
        oup['cls'] = self.cand_head_cls[cand_list_cls[0]](cls_feat)
        # reg
        cand_list_reg = cand_dict['reg']  # [0/1/2, []]
        reg_feat = self.cand_tower_cls[cand_list_reg[0]](inp['reg'], cand_list_reg[1])
        oup['reg'] = self.cand_head_reg[cand_list_reg[0]](reg_feat)

        return oup

class cls_pred_head(nn.Module):
    def __init__(self, inchannels=256):
        super(cls_pred_head, self).__init__()
        self.cls_pred = nn.Conv2d(inchannels, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """mode should be in ['all', 'cls', 'reg']"""
        x = 0.1 * self.cls_pred(x)
        return x

class reg_pred_head(nn.Module):
    def __init__(self, inchannels=256, linear_reg=False, stride=16):
        super(reg_pred_head, self).__init__()
        self.linear_reg = linear_reg
        self.stride = stride
        # reg head
        self.bbox_pred = nn.Conv2d(inchannels, 4, kernel_size=3, stride=1, padding=1)
        # adjust scale
        if not self.linear_reg:
            self.adjust = nn.Parameter(0.1 * torch.ones(1))
            self.bias = nn.Parameter(torch.Tensor(1.0 * torch.ones(1, 4, 1, 1)))

    def forward(self, x):
        if self.linear_reg:
            x = nn.functional.relu(self.bbox_pred(x)) * self.stride
        else:
            x = self.adjust * self.bbox_pred(x) + self.bias
            x = torch.exp(x)
        return x


# ===== FROM lib/models/submodels.py =====

import torch.nn as nn
import torch

class head_subnet(nn.Module):
    def __init__(self, module_dict):
        super(head_subnet, self).__init__()
        self.cls_tower = module_dict['cls_tower']
        self.reg_tower = module_dict['reg_tower']
        self.cls_perd = module_dict['cls_pred']
        self.reg_pred = module_dict['reg_pred']

    def forward(self, inp):
        oup = {}
        # cls
        cls_feat = self.cls_tower(inp['cls'])
        oup['cls'] = self.cls_perd(cls_feat)
        # reg
        reg_feat = self.reg_tower(inp['reg'])
        oup['reg'] = self.reg_pred(reg_feat)
        return oup

def get_towers(module_list: torch.nn.ModuleList, path_head, inchannels, outchannels, towernum=8, kernel_list=[3, 5, 0]):
    num_choice_kernel = len(kernel_list)
    for tower_idx in range(towernum):
        block_idx = path_head[1][tower_idx]
        kernel_sz = kernel_list[block_idx]
        if tower_idx == 0:
            assert (kernel_sz != 0)
            padding = (kernel_sz - 1) // 2
            module_list.append(SeparableConv2d_BNReLU(inchannels, outchannels, kernel_size=kernel_sz,
                                                      stride=1, padding=padding, dilation=1))
        else:
            if block_idx != num_choice_kernel - 1:  # else skip
                assert (kernel_sz != 0)
                padding = (kernel_sz - 1) // 2
                module_list.append(SeparableConv2d_BNReLU(outchannels, outchannels, kernel_size=kernel_sz,
                                                          stride=1, padding=padding, dilation=1))
    return module_list

def build_subnet_head(path_head, channel_list=[128, 192, 256], kernel_list=[3, 5, 0], inchannels=64, towernum=8,
                      linear_reg=False):
    channel_idx_cls, channel_idx_reg = path_head['cls'][0], path_head['reg'][0]
    num_channel_cls, num_channel_reg = channel_list[channel_idx_cls], channel_list[channel_idx_reg]
    tower_cls_list = nn.ModuleList()
    tower_reg_list = nn.ModuleList()
    # add operations
    tower_cls = nn.Sequential(
        *get_towers(tower_cls_list, path_head['cls'], inchannels, num_channel_cls, towernum=towernum,
                    kernel_list=kernel_list))
    tower_reg = nn.Sequential(
        *get_towers(tower_reg_list, path_head['reg'], inchannels, num_channel_reg, towernum=towernum,
                    kernel_list=kernel_list))
    # add prediction head
    cls_pred = cls_pred_head(inchannels=num_channel_cls)
    reg_pred = reg_pred_head(inchannels=num_channel_reg, linear_reg=linear_reg)

    module_dict = {'cls_tower': tower_cls, 'reg_tower': tower_reg, 'cls_pred': cls_pred, 'reg_pred': reg_pred}
    return head_subnet(module_dict)

########## BN adjust layer before Correlation ##########
class BN_adj(nn.Module):
    def __init__(self, num_channel):
        super(BN_adj, self).__init__()
        self.BN_z = nn.BatchNorm2d(num_channel)
        self.BN_x = nn.BatchNorm2d(num_channel)

    def forward(self, zf, xf):
        return self.BN_z(zf), self.BN_x(xf)

def build_subnet_BN_backup(path_ops, inp_c=(40, 80, 96)):
    num_channel = inp_c[path_ops[0] - 1]
    return BN_adj(num_channel)

def build_subnet_BN(path_ops, model_cfg):
    inc_idx = model_cfg.stage_idx.index(path_ops[0])
    num_channel = model_cfg.in_c[inc_idx]
    return BN_adj(num_channel)

class Point_Neck_Mobile_simple_DP(nn.Module):
    def __init__(self, num_kernel_list=(256, 64), cat=False, matrix=True, adjust=True, adj_channel=128):
        super(Point_Neck_Mobile_simple_DP, self).__init__()
        self.adjust = adjust
        '''Point-wise Correlation & Adjust Layer (unify the num of channels)'''
        self.pw_corr = torch.nn.ModuleList()
        self.adj_layer = torch.nn.ModuleList()
        for num_kernel in num_kernel_list:
            self.pw_corr.append(PWCA(num_kernel, cat=cat, CA=True, matrix=matrix))
            self.adj_layer.append(nn.Conv2d(num_kernel, adj_channel, 1))

    def forward(self, kernel, search, stride_idx):
        '''stride_idx: 0 or 1. 0 represents stride 8. 1 represents stride 16'''
        oup = {}
        corr_feat = self.pw_corr[stride_idx]([kernel], [search])
        if self.adjust:
            corr_feat = self.adj_layer[stride_idx](corr_feat)
        oup['cls'], oup['reg'] = corr_feat, corr_feat
        return oup

'''Point-wise Correlation & channel adjust layer'''

class PW_Corr_adj(nn.Module):
    def __init__(self, num_kernel=64, cat=False, matrix=True, adj_channel=128):
        super(PW_Corr_adj, self).__init__()
        self.pw_corr = PWCA(num_kernel, cat=cat, CA=True, matrix=matrix)
        self.adj_layer = nn.Conv2d(num_kernel, adj_channel, 1)

    def forward(self, kernel, search):
        '''stride_idx: 0 or 1. 0 represents stride 8. 1 represents stride 16'''
        oup = {}
        corr_feat = self.pw_corr([kernel], [search])
        corr_feat = self.adj_layer(corr_feat)
        oup['cls'], oup['reg'] = corr_feat, corr_feat
        return oup

def build_subnet_feat_fusor_backup(path_ops, num_kernel_list=(256, 64), cat=False, matrix=True, adj_channel=128):
    stride_idx = 0 if path_ops[0] == 1 else 1
    num_kernel = num_kernel_list[stride_idx]
    return PW_Corr_adj(num_kernel=num_kernel, cat=cat, matrix=matrix, adj_channel=adj_channel)

def build_subnet_feat_fusor(path_ops, model_cfg, cat=False, matrix=True, adj_channel=128):
    stride = model_cfg.strides[path_ops[0]]
    stride_idx = model_cfg.strides_use_new.index(stride)
    num_kernel = model_cfg.num_kernel_corr[stride_idx]
    return PW_Corr_adj(num_kernel=num_kernel, cat=cat, matrix=matrix, adj_channel=adj_channel)


# ===== FROM lib/models/super_model.py =====

import torch
import numpy as np
import torch.nn as nn

class Super_model(nn.Module):
    def __init__(self, search_size=255, template_size=127, stride=16):
        super(Super_model, self).__init__()
        self.search_size = search_size
        self.template_size = template_size
        self.stride = stride
        self.score_size = round(self.search_size / self.stride)
        self.num_kernel = round(self.template_size / self.stride) ** 2
        self.criterion = nn.BCEWithLogitsLoss()
        self.retrain = False

    def feature_extractor(self, x, cand_b=None):
        """cand_b: candidate path for backbone"""
        # if isinstance(self, nn.DataParallel):
        #     return self.features.module.forward_backbone(x, cand_b, stride=self.stride)
        # else:
        #     return self.features.forward_backbone(x, cand_b, stride=self.stride)
        if self.retrain:
            return self.features(x, stride=self.stride)
        else:
            return self.features(x, cand_b, stride=self.stride)

    def grids(self):
        """
        each element of feature map on input search image
        :return: H*W*2 (position for each element)
        """
        sz = self.score_size
        print('grids size=', sz)

        sz_x = sz // 2
        sz_y = sz // 2

        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))

        self.grid_to_search = {}
        self.grid_to_search_x = x * self.stride + self.search_size // 2
        self.grid_to_search_y = y * self.stride + self.search_size // 2

        self.grid_to_search_x = torch.Tensor(self.grid_to_search_x).unsqueeze(0).unsqueeze(0)
        self.grid_to_search_y = torch.Tensor(self.grid_to_search_y).unsqueeze(0).unsqueeze(0)

        self.grid_to_search_x = self.grid_to_search_x.repeat(self.batch, 1, 1, 1)
        self.grid_to_search_y = self.grid_to_search_y.repeat(self.batch, 1, 1, 1)

    def template(self, z, cand_b):
        self.zf = self.feature_extractor(z, cand_b)

        if self.neck is not None:
            self.zf = self.neck(self.zf, crop=False)

    def track(self, x, cand_b, cand_h_dict):
        # supernet backbone
        xf = self.feature_extractor(x, cand_b)
        # dim adjust
        if self.neck is not None:
            xf = self.neck(xf)
        # feature adjustment and correlation
        feat_dict = self.feature_fusor(self.zf, xf)
        # supernet head
        oup = self.supernet_head(feat_dict, cand_h_dict)
        return oup

    def forward(self, template, search, label=None, reg_target=None, reg_weight=None,
                cand_b=None, cand_h_dict=None):

        """run siamese network"""
        zf = self.feature_extractor(template, cand_b=cand_b)
        xf = self.feature_extractor(search, cand_b=cand_b)

        if self.neck is not None:
            zf = self.neck(zf, crop=False)
            xf = self.neck(xf, crop=False)

        # feature adjustment and correlation
        feat_dict = self.feature_fusor(zf, xf)
        # supernet head
        oup = self.supernet_head(feat_dict, cand_h_dict=cand_h_dict)
        if label is not None and reg_target is not None and reg_weight is not None:
            # compute loss
            reg_loss = self.add_iouloss(oup['reg'], reg_target, reg_weight)
            cls_loss = self._weighted_BCE(oup['cls'], label)
            return cls_loss, reg_loss
        else:
            return feat_dict

    def _weighted_BCE(self, pred, label, mode='all'):
        pred = pred.view(-1)
        label = label.view(-1)
        if mode == 'pos' or mode == 'all':
            pos = label.data.eq(1).nonzero().squeeze()
            loss_pos = self._cls_loss(pred, label, pos)
        if mode == 'neg' or mode == 'all':
            neg = label.data.eq(0).nonzero().squeeze()
            loss_neg = self._cls_loss(pred, label, neg)
        # return
        if mode == 'pos':
            return loss_pos
        elif mode == 'neg':
            return loss_neg
        elif mode == 'all':
            return loss_pos * 0.5 + loss_neg * 0.5

    def _cls_loss(self, pred, label, select):
        if len(select.size()) == 0: return 0
        pred = torch.index_select(pred, 0, select)
        label = torch.index_select(label, 0, select)
        return self.criterion(pred, label)  # the same as tf version

    def _IOULoss(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0))

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()

    def add_iouloss(self, bbox_pred, reg_target, reg_weight, iou_mode='iou'):
        """

        :param bbox_pred:
        :param reg_target:
        :param reg_weight:
        :param grid_x:  used to get real target bbox
        :param grid_y:  used to get real target bbox
        :return:
        """
        assert (iou_mode == 'iou' or iou_mode == 'diou')
        bbox_pred_flatten = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        reg_target_flatten = reg_target.reshape(-1, 4)
        reg_weight_flatten = reg_weight.reshape(-1)
        pos_inds = torch.nonzero(reg_weight_flatten > 0).squeeze(1)

        bbox_pred_flatten = bbox_pred_flatten[pos_inds]
        reg_target_flatten = reg_target_flatten[pos_inds]
        if iou_mode == 'iou':
            loss = self._IOULoss(bbox_pred_flatten, reg_target_flatten)
        elif iou_mode == 'diou':
            loss = self._DIoU_Loss(bbox_pred_flatten, reg_target_flatten)
        else:
            raise ValueError('iou_mode should be iou or diou')
        return loss

    def pred_to_image(self, bbox_pred):
        self.grid_to_search_x = self.grid_to_search_x.to(bbox_pred.device)
        self.grid_to_search_y = self.grid_to_search_y.to(bbox_pred.device)

        pred_x1 = self.grid_to_search_x - bbox_pred[:, 0, ...].unsqueeze(1)  # 17*17
        pred_y1 = self.grid_to_search_y - bbox_pred[:, 1, ...].unsqueeze(1)  # 17*17
        pred_x2 = self.grid_to_search_x + bbox_pred[:, 2, ...].unsqueeze(1)  # 17*17
        pred_y2 = self.grid_to_search_y + bbox_pred[:, 3, ...].unsqueeze(1)  # 17*17

        pred = [pred_x1, pred_y1, pred_x2, pred_y2]

        pred = torch.cat(pred, dim=1)

        return pred

'''2020.09.11 for compute MACs'''

class Super_model_MACs(nn.Module):
    def __init__(self, search_size=255, template_size=127, stride=16):
        super(Super_model_MACs, self).__init__()
        self.search_size = search_size
        self.template_size = template_size
        self.stride = stride
        self.score_size = round(self.search_size / self.stride)
        self.num_kernel = round(self.template_size / self.stride) ** 2

    def feature_extractor(self, x, cand_b):
        '''cand_b: candidate path for backbone'''
        if isinstance(self, nn.DataParallel):
            return self.features.module.forward_backbone(x, cand_b, stride=self.stride)
        else:
            return self.features.forward_backbone(x, cand_b, stride=self.stride)

    def forward(self, zf, search, cand_b, cand_h_dict):

        '''run siamese network'''
        xf = self.feature_extractor(search, cand_b)

        if self.neck is not None:
            xf = self.neck(xf, crop=False)

        # feature adjustment and correlation
        feat_dict = self.feature_fusor(zf, xf)
        # supernet head
        oup = self.supernet_head(feat_dict, cand_h_dict)

        return oup

'''2020.10.18 for retrain the searched model'''

class Super_model_retrain(Super_model):
    def __init__(self, search_size=256, template_size=128, stride=16):
        super(Super_model_retrain, self).__init__(search_size=search_size, template_size=template_size, stride=stride)

    def template(self, z):
        self.zf = self.features(z)

    def track(self, x):
        # supernet backbone
        xf = self.features(x)
        if self.neck is not None:
            raise ValueError('neck should be None')
        # Point-wise Correlation
        feat_dict = self.feature_fusor(self.zf, xf)
        # supernet head
        oup = self.head(feat_dict)
        return oup

    def forward(self, template, search, label, reg_target, reg_weight):
        '''backbone_index: which layer's feature to use'''
        zf = self.features(template, stride=self.stride)
        xf = self.features(search, stride=self.stride)
        if self.neck is not None:
            raise ValueError('neck should be None')
        # Point-wise Correlation
        feat_dict = self.feature_fusor(zf, xf)
        # supernet head
        oup = self.head(feat_dict)
        # compute loss
        reg_loss = self.add_iouloss(oup['reg'], reg_target, reg_weight)
        cls_loss = self._weighted_BCE(oup['cls'], label)
        return cls_loss, reg_loss


# ===== FROM lib/models/super_model_DP.py =====

import torch
import numpy as np
import torch.nn as nn

'''2020.10.14 Super model that supports dynamic positions for the head'''

class Super_model_DP(Super_model):
    def __init__(self, search_size=256, template_size=128, stride=16):
        super(Super_model_DP, self).__init__(search_size=search_size, template_size=template_size, stride=stride)
        self.strides = [4, 8, 16, 16, 32]
        self.channel_back = [24, 40, 80, 96, 192]
        self.num_choice_back = 6

    def feature_extractor(self, x, cand_b, backbone_index):
        """cand_b: candidate path for backbone"""
        if self.retrain:
            return self.features(x, stride=self.stride)
        else:
            return self.features(x, cand_b, stride=self.stride, backbone_index=backbone_index)

    def template(self, z, cand_b, backbone_index):
        self.zf = self.feature_extractor(z, cand_b, backbone_index)

    def track(self, x, cand_b, cand_h_dict, backbone_index):
        # supernet backbone
        xf = self.feature_extractor(x, cand_b, backbone_index)
        # Batch Normalization before Corr
        zf, xf = self.neck(self.zf, xf, self.stage_idx.index(backbone_index[0]))
        # Point-wise Correlation
        stride = self.strides[backbone_index[0]]
        stride_idx = self.strides_use_new.index(stride)
        feat_dict = self.feature_fusor(zf, xf, stride_idx)
        # supernet head
        oup = self.supernet_head(feat_dict, cand_h_dict)
        return oup

    def forward(self, template, search, label=None, reg_target=None, reg_weight=None,
                cand_b=None, cand_h_dict=None, backbone_index=None):
        """backbone_index: which layer's feature to use"""
        zf = self.feature_extractor(template, cand_b, backbone_index)
        xf = self.feature_extractor(search, cand_b, backbone_index)
        # Batch Normalization before Corr
        zf, xf = self.neck(zf, xf, self.stage_idx.index(backbone_index[0]))
        # Point-wise Correlation
        stride = self.strides[backbone_index[0]]
        stride_idx = self.strides_use_new.index(stride)
        feat_dict = self.feature_fusor(zf, xf, stride_idx)
        # supernet head
        oup = self.supernet_head(feat_dict, cand_h_dict)
        if label is not None and reg_target is not None and reg_weight is not None:
            # compute loss
            reg_loss = self.add_iouloss(oup['reg'], reg_target, reg_weight)
            cls_loss = self._weighted_BCE(oup['cls'], label)
            return cls_loss, reg_loss
        return oup

    def get_attribute(self):
        return {'search_back': self.search_back, 'search_out': self.search_ops, 'search_head': self.search_head}

    def clean_module_BN(self, model):
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean = torch.zeros_like(m.running_mean)
                m.running_var = torch.ones_like(m.running_var)

    def clean_BN(self):
        print('clear bn statics....')
        if self.search_back:
            print('cleaning backbone BN ...')
            self.clean_module_BN(self.features)
        if self.search_head:
            print('cleaning head BN ...')
            self.clean_module_BN(self.supernet_head)
        if self.search_ops:
            print('cleaning neck and feature_fusor BN ...')
            self.clean_module_BN(self.neck)
            self.clean_module_BN(self.feature_fusor)

'''2020.10.17 Compute MACs for DP networks'''

class Super_model_DP_MACs(nn.Module):
    def __init__(self, search_size=256, template_size=128, stride=16):
        super(Super_model_DP_MACs, self).__init__()
        self.search_size = search_size
        self.template_size = template_size
        self.stride = stride

    def feature_extractor(self, x, cand_b, backbone_index):
        '''cand_b: candidate path for backbone'''
        return self.model.features.forward_backbone(x, cand_b, stride=self.stride, backbone_index=backbone_index)

    def forward(self, zf, search, cand_b, cand_h_dict, backbone_index):
        xf = self.model.feature_extractor(search, cand_b, backbone_index)
        # Batch Normalization before Corr
        neck_idx = self.model.stage_idx.index(backbone_index[0])
        zf, xf = self.model.neck(zf[neck_idx], xf, neck_idx)
        # Point-wise Correlation
        stride = self.model.strides[backbone_index[0]]
        stride_idx = self.model.strides_use_new.index(stride)
        feat_dict = self.model.feature_fusor(zf, xf, stride_idx)
        # supernet head
        oup = self.model.supernet_head(feat_dict, cand_h_dict)
        return oup

'''2020.10.14 Super model that supports dynamic positions for the head'''
'''2020.10.18 for retrain the searched model'''

class Super_model_DP_retrain(Super_model):
    def __init__(self, search_size=256, template_size=128, stride=16):
        super(Super_model_DP_retrain, self).__init__(search_size=search_size, template_size=template_size,
                                                     stride=stride)

    def template(self, z):
        self.zf = self.features(z)

    def track(self, x):
        # supernet backbone
        xf = self.features(x)
        # BN before Pointwise Corr
        zf, xf = self.neck(self.zf, xf)
        # Point-wise Correlation
        feat_dict = self.feature_fusor(zf, xf)
        # supernet head
        oup = self.head(feat_dict)
        return oup['cls'], oup['reg']

    def forward(self, template, search, label, reg_target, reg_weight):
        '''backbone_index: which layer's feature to use'''
        zf = self.features(template)
        xf = self.features(search)
        # Batch Normalization before Corr
        zf, xf = self.neck(zf, xf)
        # Point-wise Correlation
        feat_dict = self.feature_fusor(zf, xf)
        # supernet head
        oup = self.head(feat_dict)
        # compute loss
        reg_loss = self.add_iouloss(oup['reg'], reg_target, reg_weight)
        cls_loss = self._weighted_BCE(oup['cls'], label)
        return cls_loss, reg_loss


# ===== FROM lib/utils/transform.py =====

import numpy as np
import random

def get_cand_with_prob(CHOICE_NUM, prob=None, sta_num=(4, 4, 4, 4, 4)):
    if prob is None:
        get_random_cand = [np.random.choice(CHOICE_NUM, item).tolist() for item in sta_num]
    else:
        get_random_cand = [np.random.choice(CHOICE_NUM, item, prob).tolist() for item in sta_num]
    # print(get_random_cand)
    return get_random_cand

def get_cand_head():
    oup = [random.randint(0, 2)]  # num of channels (3 choices)
    arch = [random.randint(0, 1)]
    arch += list(np.random.choice(3, 7))  # 3x3 conv, 5x5 conv, skip
    oup.append(arch)
    return oup

def get_cand_head_wo_ID():
    """2020.10.24 Without using IDentity"""
    oup = [random.randint(0, 2)]  # num of channels (3 choices)
    arch = []
    arch.append(random.randint(0, 1))  # 3x3 conv, 5x5 conv
    arch += list(np.random.choice(2, 7))  # 3x3 conv, 5x5 conv
    oup.append(arch)
    return oup

def get_oup_pos(sta_num):
    stage_idx = random.randint(2, 3)  # 1, 2, 3
    block_num = sta_num[stage_idx]
    block_idx = random.randint(0, block_num - 1)
    return [stage_idx, block_idx]

'''2020.10.5 name --> path'''
'''2020.10.17 modified version'''

def name2path_backhead(path_name, sta_num=(4, 4, 4, 4, 4), head_only=False, backbone_only=False):
    print(path_name)
    backbone_name, head_name = path_name.split('+cls_')
    if not head_only:
        # process backbone
        backbone_name = backbone_name.strip('back_')[1:-1]  # length = 20 when 600M, length = 18 when 470M
        backbone_path = [[], [], [], [], []]
        for stage_idx in range(len(sta_num)):
            for block_idx in range(sta_num[stage_idx]):
                str_idx = block_idx + sum(sta_num[:stage_idx])
                backbone_path[stage_idx].append(int(backbone_name[str_idx]))
        backbone_path.insert(0, [0])
        backbone_path.append([0])
    if not backbone_only:
        # process head
        cls_name, reg_name = head_name.split('+reg_')
        head_path = {}
        cls_path = [int(cls_name[0])]
        cls_path.append([int(item) for item in cls_name[1:]])
        head_path['cls'] = cls_path
        reg_path = [int(reg_name[0])]
        reg_path.append([int(item) for item in reg_name[1:]])
        head_path['reg'] = reg_path
    # combine
    if head_only:
        backbone_path = None
    if backbone_only:
        head_path = None
    return tuple([backbone_path, head_path])

'''2020.10.5 name --> path'''
'''2020.10.17 modified version'''

def name2path(path_name, sta_num=(4, 4, 4, 4, 4), head_only=False, backbone_only=False):
    if '_ops_' in path_name:
        first_name, ops_name = path_name.split('_ops_')
        backbone_path, head_path = name2path_backhead(first_name, sta_num=sta_num, head_only=head_only,
                                                      backbone_only=backbone_only)
        ops_path = (int(ops_name[0]), int(ops_name[1]))
        return backbone_path, head_path, ops_path
    else:
        return name2path_backhead(path_name, sta_num=sta_num, head_only=head_only, backbone_only=backbone_only)

def name2path_ablation(path_name, sta_num=(4, 4, 4, 4, 4), num_tower=8):
    back_path, head_path, ops_path = None, None, None
    if 'back' in path_name:
        back_str_len = sum(sta_num) + 2  # head0, tail0
        back_str = path_name.split('back_')[1][:back_str_len]
        back_str = back_str[1:-1]  # remove head0 and tail0
        back_path = [[], [], [], [], []]
        for stage_idx in range(len(sta_num)):
            for block_idx in range(sta_num[stage_idx]):
                str_idx = block_idx + sum(sta_num[:stage_idx])
                back_path[stage_idx].append(int(back_str[str_idx]))
        back_path.insert(0, [0])
        back_path.append([0])
    if 'cls' in path_name and 'reg' in path_name:
        head_path = {}
        cls_str_len = num_tower + 1  # channel idx
        cls_str = path_name.split('cls_')[1][:cls_str_len]
        cls_path = [int(cls_str[0]), [int(item) for item in cls_str[1:]]]
        head_path['cls'] = cls_path
        reg_str_len = num_tower + 1  # channel idx
        reg_str = path_name.split('reg_')[1][:reg_str_len]
        reg_path = [int(reg_str[0]), [int(item) for item in reg_str[1:]]]
        head_path['reg'] = reg_path
    if 'ops' in path_name:
        ops_str = path_name.split('ops_')[1]
        ops_path = (int(ops_str[0]), int(ops_str[1]))
    return {'back': back_path, 'head': head_path, 'ops': ops_path}

if __name__ == "__main__":
    for _ in range(10):
        print(get_cand_head())


# ===== FROM lib/models/models.py =====



import numpy as np
import random

class LightTrackM_Supernet(Super_model_DP):
    def __init__(self, search_size=256, template_size=128, stride=16, adj_channel=128, build_module=True):
        """subclass calls father class's __init__ func"""
        super(LightTrackM_Supernet, self).__init__(search_size=search_size, template_size=template_size,
                                                   stride=stride)  # ATTENTION
        # config #
        # which parts to search
        self.search_back, self.search_ops, self.search_head = 1, 1, 1
        # backbone config
        self.stage_idx = [1, 2, 3]  # which stages to use
        self.max_flops_back = 470
        # head config
        self.channel_head = [128, 192, 256]
        self.kernel_head = [3, 5, 0]  # 0 means skip connection
        self.tower_num = 8  # max num of layers in the head
        self.num_choice_channel_head = len(self.channel_head)
        self.num_choice_kernel_head = len(self.kernel_head)
        # Compute some values #
        self.in_c = [self.channel_back[idx] for idx in self.stage_idx]
        strides_use = [self.strides[idx] for idx in self.stage_idx]
        strides_use_new = []
        for item in strides_use:
            if item not in strides_use_new:
                strides_use_new.append(item)  # remove repeated elements
        self.strides_use_new = strides_use_new
        self.num_kernel_corr = [int(round(template_size / stride) ** 2) for stride in strides_use_new]
        # build the architecture #
        if build_module:
            self.features, self.sta_num = build_supernet_DP(flops_maximum=self.max_flops_back)
            self.neck = MC_BN(inp_c=self.in_c)  # BN with multiple types of input channels
            self.feature_fusor = Point_Neck_Mobile_simple_DP(num_kernel_list=self.num_kernel_corr, matrix=True,
                                                             adj_channel=adj_channel)  # stride=8, stride=16
            self.supernet_head = head_supernet(channel_list=self.channel_head, kernel_list=self.kernel_head,
                                               linear_reg=True, inchannels=adj_channel, towernum=self.tower_num)
        else:
            _, self.sta_num = build_supernet_DP(flops_maximum=self.max_flops_back)

class LightTrackM_FLOPs(Super_model_DP_MACs):
    def __init__(self, search_size=256, template_size=128, stride=16, adj_channel=128):
        '''subclass calls father class's __init__ func'''
        super(LightTrackM_FLOPs, self).__init__(search_size=search_size, template_size=template_size,
                                                stride=stride)  # ATTENTION
        self.model = LightTrackM_Supernet(search_size=search_size, template_size=template_size,
                                          stride=stride, adj_channel=adj_channel)

class LightTrackM_Subnet(Super_model_DP_retrain):
    def __init__(self, path_name, search_size=256, template_size=128, stride=16, adj_channel=128):
        """subclass calls father class's __init__ func"""
        super(LightTrackM_Subnet, self).__init__(search_size=search_size, template_size=template_size,
                                                 stride=stride)  # ATTENTION
        model_cfg = LightTrackM_Supernet(search_size=search_size, template_size=template_size,
                                         stride=stride, adj_channel=adj_channel, build_module=False)

        path_backbone, path_head, path_ops = name2path(path_name, sta_num=model_cfg.sta_num)
        # build the backbone
        self.features = build_subnet(path_backbone, ops=path_ops)  # sta_num is based on previous flops
        # build the neck layer
        self.neck = build_subnet_BN(path_ops, model_cfg)
        # build the Correlation layer and channel adjustment layer
        self.feature_fusor = build_subnet_feat_fusor(path_ops, model_cfg, matrix=True, adj_channel=adj_channel)
        # build the head
        self.head = build_subnet_head(path_head, channel_list=model_cfg.channel_head, kernel_list=model_cfg.kernel_head,
                                      inchannels=adj_channel, linear_reg=True, towernum=model_cfg.tower_num)

class LightTrackM_Speed(LightTrackM_Subnet):
    def __init__(self, path_name, search_size=256, template_size=128, stride=16, adj_channel=128):
        super(LightTrackM_Speed, self).__init__(path_name, search_size=search_size, template_size=template_size,
                                                stride=stride, adj_channel=adj_channel)

    def forward(self, x, zf):
        # backbone
        xf = self.features(x)
        # BN before Point-wise Corr
        zf, xf = self.neck(zf, xf)
        # Point-wise Correlation
        feat_dict = self.feature_fusor(zf, xf)
        # head
        oup = self.head(feat_dict)
        return oup

class SuperNetToolbox(object):
    def __init__(self, model):
        self.model = model

    def get_path_back(self, prob=None):
        """randomly sample one path from the backbone supernet"""
        if prob is None:
            path_back = [np.random.choice(self.model.num_choice_back, item).tolist() for item in self.model.sta_num]
        else:
            path_back = [np.random.choice(self.model.num_choice_back, item, prob).tolist() for item in
                         self.model.sta_num]
        # add head and tail
        path_back.insert(0, [0])
        path_back.append([0])
        return path_back

    def get_path_head_single(self):
        num_choice_channel_head = self.model.num_choice_channel_head
        num_choice_kernel_head = self.model.num_choice_kernel_head
        tower_num = self.model.tower_num
        oup = [random.randint(0, num_choice_channel_head - 1)]  # num of choices for head's channel
        arch = [random.randint(0, num_choice_kernel_head - 2)]
        arch += list(np.random.choice(num_choice_kernel_head, tower_num - 1))  # 3x3 conv, 5x5 conv, skip
        oup.append(arch)
        return oup

    def get_path_head(self):
        """randomly sample one path from the head supernet"""
        cand_h_dict = {'cls': self.get_path_head_single(), 'reg': self.get_path_head_single()}
        return cand_h_dict

    def get_path_ops(self):
        """randomly sample an output position"""
        stage_idx = random.choice(self.model.stage_idx)
        block_num = self.model.sta_num[stage_idx]
        block_idx = random.randint(0, block_num - 1)
        return [stage_idx, block_idx]

    def get_one_path(self):
        """randomly sample one complete path from the whole supernet"""
        cand_back, cand_OP, cand_h_dict = None, None, None
        tower_num = self.model.tower_num
        if self.model.search_back or self.model.search_ops:
            # backbone operations
            cand_back = self.get_path_back()
        if self.model.search_ops:
            # backbone output positions
            cand_OP = self.get_path_ops()
        if self.model.search_head:
            # head operations
            cand_h_dict = self.get_path_head()
        else:
            cand_h_dict = {'cls': [0, [0] * tower_num], 'reg': [0, [0] * tower_num]}  # use fix head (only one choice)
        return {'back': cand_back, 'ops': cand_OP, 'head': cand_h_dict}


# ============================================================================
# MODEL LOADING UTILITIES
# ============================================================================

def check_keys(model, pretrained_state_dict, print_unuse=True):
    """Check which keys from pretrained model are used/unused"""
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = list(ckpt_keys - model_keys)
    missing_keys = list(model_keys - ckpt_keys)

    for k in sorted(missing_keys):
        if 'num_batches_tracked' in k:
            missing_keys.remove(k)

    print('missing keys:{}'.format(missing_keys))
    if print_unuse:
        print('unused checkpoint keys:{}'.format(unused_pretrained_keys))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    """Remove prefix from state dict keys"""
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_pretrain(model, pretrained_path, print_unuse=True):
    """Load pretrained weights into model"""
    print('load pretrained model from {}'.format(pretrained_path))
    
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
        pretrained_dict = remove_prefix(pretrained_dict, 'feature_extractor.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
        pretrained_dict = remove_prefix(pretrained_dict, 'feature_extractor.')

    check_keys(model, pretrained_dict, print_unuse=print_unuse)
    model.load_state_dict(pretrained_dict, strict=False)
    return model



# ============================================================================
# MULTI-OBJECT TRACKER WRAPPER
# ============================================================================

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
            self.siam_net = LightTrackM_Subnet(path_name, stride=stride)
        else:
            self.siam_net = LightTrackM_Subnet(stride=stride)
        
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
        # Convert BGR to RGB if needed
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
