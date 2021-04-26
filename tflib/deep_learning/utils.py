import functools

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tflib import deep_learning


# ==============================================================================
# =                           normalization utility                            =
# ==============================================================================

def get_feature_norm(norm, training, updates_collections=None):
    if norm in [None, 'none']:
        return None
    elif norm == 'identity':
        return lambda x: x
    elif norm == 'batch_norm':
        return functools.partial(slim.batch_norm, scale=True, is_training=training, updates_collections=updates_collections)
    elif norm == 'instance_norm':
        return slim.instance_norm
    elif norm == 'layer_norm':
        return slim.layer_norm
    else:
        raise NotImplementedError


def get_weight_norm(norm, training, transposed=False):
    if norm in [None, 'none']:
        return None
    elif norm == 'spectral_norm':
        return functools.partial(deep_learning.spectral_normalization, transposed=transposed, is_training=training)
    elif norm == 'weight_norm':
        return functools.partial(deep_learning.weight_normalization, transposed=transposed)
    else:
        raise NotImplementedError


# ==============================================================================
# =                           initialization utility                           =
# ==============================================================================

def get_initialization_gain(activation_fn_or_name, activation_params=None):
    activation_params = {} if activation_params is None else activation_params

    if activation_fn_or_name in [tf.nn.relu, 'relu']:
        gain = 2.0**0.5
    elif activation_fn_or_name in [tf.nn.leaky_relu, 'leaky_relu']:
        alpha = activation_params.pop('alpha', 0.2)
        gain = (2.0 / (1 + alpha**2))**0.5
    elif activation_fn_or_name in [tf.nn.tanh, 'tanh']:
        gain = 5.0 / 3
    elif activation_fn_or_name in [tf.nn.sigmoid, 'sigmoid', None, 'none']:
        gain = 1.0
    else:
        raise ValueError("Unsupported activation!")

    return gain


def get_fan(weights, mode='fan_in'):
    # modified from https://github.com/tensorflow/tensorflow/blob/v1.15.0/tensorflow/python/ops/init_ops.py#L451-L542
    shape = weights.shape
    if len(shape) < 1:  # Just to avoid errors for constants.
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        # Assuming convolution kernels (2D, 3D, or more).
        # kernel shape: (..., input_depth, depth)
        receptive_field_size = 1
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return {'fan_in': int(fan_in), 'fan_out': int(fan_out), 'fan_avg': int((fan_in + fan_out) // 2)}[mode]


def get_initializer(activation_fn_or_name, activation_params=None, mode='fan_in', distribution='truncated_normal', seed=None, dtype=tf.dtypes.float32):
    gain = get_initialization_gain(activation_fn_or_name, activation_params=activation_params)
    return tf.initializers.variance_scaling(scale=gain**2, mode=mode, distribution=distribution, seed=seed, dtype=dtype)
