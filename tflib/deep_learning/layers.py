import functools

import tensorflow as tf


# ==============================================================================
# =                                   filter                                   =
# ==============================================================================

def fully_connected(inputs,
                    num_outputs,
                    activation_fn=None,
                    normalizer_fn=None,
                    normalizer_params=None,
                    weights_normalizer_fn=None,
                    weights_normalizer_params=None,
                    weights_initializer=tf.glorot_uniform_initializer(),
                    weights_regularizer=None,
                    biases_initializer=tf.zeros_initializer(),
                    biases_regularizer=None,
                    reuse=None,
                    trainable=True,
                    scope=None):
    with tf.variable_scope(scope, 'fully_connected', reuse=reuse):
        num_inputs = inputs.shape[-1]
        outputs_shape_static = inputs.shape[:-1] + [num_outputs]
        outputs_shape = tf.concat([tf.shape(inputs)[:-1], [num_outputs]], axis=0)

        weights_shape = [num_inputs, num_outputs]
        weights = tf.get_variable('weights',
                                  shape=weights_shape,
                                  initializer=weights_initializer,
                                  regularizer=weights_regularizer,
                                  trainable=trainable)
        if weights_normalizer_fn is not None:
            weights_normalizer_params = weights_normalizer_params or {}
            weights = weights_normalizer_fn(weights, **weights_normalizer_params)

        if len(outputs_shape_static) > 2:
            inputs = tf.reshape(inputs, [-1, num_inputs])
        outputs = tf.matmul(inputs, weights)

        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            outputs = normalizer_fn(outputs, **normalizer_params)
        else:
            if biases_initializer is not None:
                biases = tf.get_variable('biases',
                                         shape=[num_outputs],
                                         initializer=biases_initializer,
                                         regularizer=biases_regularizer,
                                         trainable=trainable)
                outputs = tf.nn.bias_add(outputs, biases)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        if len(outputs_shape_static) > 2:
            outputs = tf.reshape(outputs, outputs_shape)
            outputs.set_shape(outputs_shape_static)

        return outputs

dense = fc = fully_connected


def convolution(inputs,
                num_outputs,
                kernel_size,
                stride=1,
                padding='SAME',
                data_format=None,
                rate=1,
                activation_fn=None,
                normalizer_fn=None,
                normalizer_params=None,
                weights_normalizer_fn=None,
                weights_normalizer_params=None,
                weights_initializer=tf.glorot_uniform_initializer(),
                weights_regularizer=None,
                biases_initializer=tf.zeros_initializer(),
                biases_regularizer=None,
                reuse=None,
                trainable=True,
                scope=None):
    with tf.variable_scope(scope, 'convolution', reuse=reuse):
        conv_dims = inputs.shape.rank - 2
        kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else [kernel_size] * conv_dims
        stride = stride if isinstance(stride, (list, tuple)) else [stride] * conv_dims
        rate = rate if isinstance(rate, (list, tuple)) else [rate] * conv_dims
        if data_format is None or data_format.endswith('C'):
            num_inputs = inputs.shape[-1]
        elif data_format.startswith('NC'):
            num_inputs = inputs.shape[1]
        else:
            raise ValueError('Invalid data_format')

        weights = tf.get_variable('weights',
                                  shape=list(kernel_size) + [num_inputs, num_outputs],
                                  initializer=weights_initializer,
                                  regularizer=weights_regularizer,
                                  trainable=trainable)
        if weights_normalizer_fn is not None:
            weights_normalizer_params = weights_normalizer_params or {}
            weights = weights_normalizer_fn(weights, **weights_normalizer_params)

        outputs = tf.nn.convolution(input=inputs,
                                    filter=weights,
                                    dilation_rate=rate,
                                    strides=stride,
                                    padding=padding,
                                    data_format=data_format)

        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            outputs = normalizer_fn(outputs, **normalizer_params)
        else:
            if biases_initializer is not None:
                biases = tf.get_variable('biases',
                                         shape=[num_outputs],
                                         initializer=biases_initializer,
                                         regularizer=biases_regularizer,
                                         trainable=trainable)
                outputs = tf.nn.bias_add(outputs, biases, data_format=data_format)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs

conv2d = convolution2d = convolution
conv3d = convolution3d = convolution


def transposed_convolution2d(inputs,
                             num_outputs,
                             kernel_size,
                             stride=1,
                             padding='SAME',
                             data_format='NHWC',
                             activation_fn=None,
                             normalizer_fn=None,
                             normalizer_params=None,
                             weights_normalizer_fn=None,
                             weights_normalizer_params=None,
                             weights_initializer=tf.glorot_uniform_initializer(),
                             weights_regularizer=None,
                             biases_initializer=tf.zeros_initializer(),
                             biases_regularizer=None,
                             reuse=None,
                             trainable=True,
                             scope=None):
    with tf.variable_scope(scope, 'transposed_convolution2d', reuse=reuse):
        def outputs_size(inputs_size, kernel, stride):
            if inputs_size is not None:
                return inputs_size * stride + (max(kernel - stride, 0) if padding == 'VALID' else 0)
            else:
                return None

        kernel_h, kernel_w = kernel_size if isinstance(kernel_size, (list, tuple)) else (kernel_size, kernel_size)
        stride_h, stride_w = stride if isinstance(stride, (list, tuple)) else (stride, stride)
        if data_format == 'NHWC':
            h_axis, w_axis, c_axis = 1, 2, 3
        else:
            c_axis, h_axis, w_axis = 1, 2, 3

        weights = tf.get_variable('weights',
                                  shape=[kernel_h, kernel_w, num_outputs, inputs.shape[c_axis]],
                                  initializer=weights_initializer,
                                  regularizer=weights_regularizer,
                                  trainable=trainable)
        if weights_normalizer_fn is not None:
            weights_normalizer_params = weights_normalizer_params or {}
            weights = weights_normalizer_fn(weights, **weights_normalizer_params)

        outputs_h = outputs_size(tf.shape(inputs)[h_axis], kernel_h, stride_h)
        outputs_h_static = outputs_size(inputs.shape[h_axis], kernel_h, stride_h)
        outputs_w = outputs_size(tf.shape(inputs)[w_axis], kernel_w, stride_w)
        outputs_w_static = outputs_size(inputs.shape[w_axis], kernel_w, stride_w)
        if data_format == 'NHWC':
            outputs_shape = [tf.shape(inputs)[0], outputs_h, outputs_w, num_outputs]
            outputs_shape_static = [inputs.shape[0], outputs_h_static, outputs_w_static, num_outputs]
            strides = [1, stride_h, stride_w, 1]
        else:
            outputs_shape = [tf.shape(inputs)[0], num_outputs, outputs_h, outputs_w]
            outputs_shape_static = [inputs.shape[0], num_outputs, outputs_h_static, outputs_w_static]
            strides = [1, 1, stride_h, stride_w]
        outputs = tf.nn.conv2d_transpose(inputs,
                                         weights,
                                         outputs_shape,
                                         strides,
                                         padding=padding,
                                         data_format=data_format)
        outputs.set_shape(outputs_shape_static)

        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            outputs = normalizer_fn(outputs, **normalizer_params)
        else:
            if biases_initializer is not None:
                biases = tf.get_variable('biases',
                                         shape=[num_outputs],
                                         initializer=biases_initializer,
                                         regularizer=biases_regularizer,
                                         trainable=trainable)
                outputs = tf.nn.bias_add(outputs, biases, data_format=data_format)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs

dconv2d = transposed_conv2d = transposed_convolution2d


# ==============================================================================
# =                           feature normalization                            =
# ==============================================================================

def adaptive_scaling(x, gamma=None, beta=None):
    # x: (N, H, W, C), gamma: (N, C), beta: (N, C)

    if gamma is not None:
        x *= gamma[:, None, None, :]

    if beta is not None:
        x += beta[:, None, None, :]

    return x


def adaptive_instance_normalization(x, gamma=None, beta=None, epsilon=1e-5):
    # modified from https://github.com/taki0112/MUNIT-Tensorflow/blob/master/ops.py
    # x: (N, H, W, C), gamma: (N, C), beta: (N, C)

    c_mean, c_var = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
    c_std = tf.sqrt(c_var + epsilon)
    x = (x - c_mean) / c_std

    return adaptive_scaling(x, gamma, beta)


# ==============================================================================
# =                            weight normalization                            =
# ==============================================================================

def spectral_normalization(weights,
                           num_iterations=1,
                           epsilon=1e-12,
                           transposed=False,  # transposed=True for 2d transposed convolution
                           is_training=True,
                           reuse=None,
                           scope=None):
    with tf.variable_scope(scope, 'spectral_normalization', reuse=reuse):
        sigma = tf.get_variable('sigma',
                                shape=[],
                                initializer=tf.zeros_initializer(),
                                trainable=False)

        def sigma_training():
            if transposed:
                w_t = tf.reshape(tf.transpose(weights, [0, 1, 3, 2]), [-1, weights.shape[-2]])
            else:
                w_t = tf.reshape(weights, [-1, weights.shape[-1]])
            w = tf.transpose(w_t)
            u = tf.get_variable("u",
                                shape=[w.shape[0], 1],
                                initializer=tf.random_normal_initializer(),
                                trainable=False)
            u_ = u
            for _ in range(num_iterations):
                v_ = tf.nn.l2_normalize(tf.matmul(w_t, u_), epsilon=epsilon)
                u_ = tf.nn.l2_normalize(tf.matmul(w, v_), epsilon=epsilon)
            u_ = tf.stop_gradient(u_)
            v_ = tf.stop_gradient(v_)
            sigma_ = tf.matmul(tf.transpose(u_), tf.matmul(w, v_))[0, 0]
            with tf.control_dependencies([u.assign(u_), sigma.assign(sigma_)]):
                return tf.identity(sigma_)

        sigma_ = tf.contrib.framework.smart_cond(is_training, sigma_training, lambda: sigma)
        weights_sn = weights / sigma_

        return weights_sn

transposed_spectral_normalization = functools.partial(spectral_normalization, transposed=True)


def weight_normalization(weights,
                         scale=True,
                         epsilon=1e-8,
                         transposed=False,  # transposed=True for 2d transposed convolution
                         reuse=None,
                         trainable=True,
                         scope=None):
    with tf.variable_scope(scope, 'weight_normalization', reuse=reuse):
        if transposed:
            weights = tf.transpose(weights, [0, 1, 3, 2])

        weights = weights / tf.sqrt(tf.reduce_sum(weights ** 2, axis=[i for i in range(weights.shape.rank - 1)], keepdims=True) + epsilon)

        if scale:
            scale = tf.get_variable('scale',  # (CO,)
                                    shape=[weights.shape[-1]],
                                    dtype=tf.float32,
                                    initializer=tf.ones_initializer(),
                                    trainable=trainable)
            weights *= scale

        if transposed:
            weights = tf.transpose(weights, [0, 1, 3, 2])

        return weights

transposed_weight_normalization = functools.partial(weight_normalization, transposed=True)
