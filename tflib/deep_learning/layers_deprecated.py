import functools

import tensorflow as tf


# ==============================================================================
# =                            weight normalization                            =
# ==============================================================================

def weight_modulation(weights,
                      gamma=None,
                      demodulation=True,
                      epsilon=1e-8,
                      transposed=False,  # transposed=True for 2d transposed convolution
                      reuse=None,
                      trainable=True,
                      scope=None):
    # only for 2d (transposed) convolution
    # gamma: None or (N, CI)

    with tf.variable_scope(scope, 'weight_modulation', reuse=reuse):
        if transposed:
            weights = tf.transpose(weights, [0, 1, 3, 2])

        if gamma is None:  # learned gamma
            gamma = tf.get_variable('gamma',  # (N=1, 1, 1, CI, 1)
                                    shape=[1, 1, 1, weights.shape[2], 1],
                                    dtype=tf.float32,
                                    initializer=tf.ones_initializer(),
                                    trainable=trainable)
        else:  # data dependent gamma
            gamma = gamma[:, None, None, :, None]  # (N, 1, 1, CI, 1)

        # modulation
        weights = weights[None, :, :, :, :] * gamma  # (N, K, K, CI, CO)

        # demodulation
        if demodulation:
            weights = weights / tf.sqrt(tf.reduce_sum(weights ** 2, axis=[1, 2, 3], keepdims=True) + epsilon)

        weights = tf.reshape(tf.transpose(weights, [1, 2, 3, 0, 4]), [weights.shape[1], weights.shape[2], weights.shape[3], -1])  # (K, K, CI, N*CO)

        if transposed:
            weights = tf.transpose(weights, [0, 1, 3, 2])

        return weights

transposed_weight_modulation = functools.partial(weight_modulation, transposed=True)


# ==============================================================================
# =                                  special                                   =
# ==============================================================================

def modulated_convolution2d(inputs,
                            num_outputs,
                            kernel_size,
                            stride=1,
                            padding='SAME',

                            gamma=None,
                            demodulation=True,
                            epsilon=1e-8,

                            data_format=None,
                            rate=1,
                            activation_fn=None,
                            normalizer_fn=None,
                            normalizer_params=None,
                            weights_initializer=tf.glorot_uniform_initializer(),
                            weights_regularizer=None,
                            biases_initializer=tf.zeros_initializer(),
                            biases_regularizer=None,
                            reuse=None,
                            trainable=True,
                            scope=None):
    # only for 2d convolution

    with tf.variable_scope(scope, 'modulated_convolution2d', reuse=reuse):
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
        weights = weight_modulation(weights, gamma=gamma, demodulation=demodulation, epsilon=epsilon, trainable=trainable)

        if gamma is not None:
            inputs = tf.transpose(inputs, [1, 2, 0, 3])  # (H, W, N, CI)
            inputs = tf.reshape(inputs, [1, inputs.shape[0], inputs.shape[1], -1])  # (1, H, W, N * CI)
        outputs = tf.nn.convolution(input=inputs,
                                    filter=weights,
                                    dilation_rate=rate,
                                    strides=stride,
                                    padding=padding,
                                    data_format=data_format)
        if gamma is not None:
            outputs = tf.reshape(outputs, [outputs.shape[1], outputs.shape[2], -1, num_outputs])  # (H, W, N, CO)
            outputs = tf.transpose(outputs, [2, 0, 1, 3])

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
