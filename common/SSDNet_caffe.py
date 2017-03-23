
#
#
#
import sys

import caffe
from caffe.model_libs import *

sys.path.append("/home/u2459/weiliu89_caffe/python/caffe")




batch_sampler = [
    { # Non-modified batch
        'sampler': {},
        'max_trials': 1,
        'max_sample': 1,
    },
    { # 
        'sampler': {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
        },
        'sample_constraint': {
            'min_jaccard_overlap': 0.1,
        },
        'max_trials': 50,
        'max_sample': 1,
    },
    {
        'sampler': {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
        },
        'sample_constraint': {
            'min_jaccard_overlap': 0.3,
        },
        'max_trials': 50,
        'max_sample': 1,
    },
    {
        'sampler': {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
        },
        'sample_constraint': {
            'min_jaccard_overlap': 0.5,
        },
        'max_trials': 50,
        'max_sample': 1,
    },
    {
        'sampler': {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
        },
        'sample_constraint': {
            'min_jaccard_overlap': 0.7,
        },
        'max_trials': 50,
        'max_sample': 1,
    },
    {
        'sampler': {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
        },
        'sample_constraint': {
            'min_jaccard_overlap': 0.9,
        },
        'max_trials': 50,
        'max_sample': 1,
    },
    {
        'sampler': {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
        },
        'sample_constraint': {
            'max_jaccard_overlap': 1.0,
        },
        'max_trials': 50,
        'max_sample': 1,
    },]


train_transform_param = {
    'mirror': True,
    'mean_value': [104, 117, 123],
    'resize_param': {
        'prob': 1,
        'resize_mode': P.Resize.WARP,
        'height': resize_height,
        'width': resize_width,
        'interp_mode': [
            P.Resize.LINEAR,
            P.Resize.AREA,
            P.Resize.NEAREST,
            P.Resize.CUBIC,
            P.Resize.LANCZOS4,
        ],
    },
    'distort_param': {
            'brightness_prob': 0.5,
            'brightness_delta': 32,
            'contrast_prob': 0.5,
            'contrast_lower': 0.5,
            'contrast_upper': 1.5,
            'hue_prob': 0.5,
            'hue_delta': 18,
            'saturation_prob': 0.5,
            'saturation_lower': 0.5,
            'saturation_upper': 1.5,
            'random_order_prob': 0.0,
            },
    'expand_param': {
            'prob': 0.5,
            'max_expand_ratio': 4.0,
            },
    'emit_constraint': {
        'emit_type': caffe_pb2.EmitConstraint.CENTER,
        }
}