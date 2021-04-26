import tensorflow as tf


def filter2d_v1(image, kernel, normalize_kernel=True, data_format=None):
    # image: NHWC or NCHW
    if normalize_kernel:
        kernel /= tf.reduce_sum(kernel)
    kernel = kernel[:, :, None, None]
    if data_format is None or data_format == "NHWC":
        kernel = tf.tile(kernel, [1, 1, image.shape[3], 1])
    elif data_format == "NCHW":
        kernel = tf.tile(kernel, [1, 1, image.shape[1], 1])
    return tf.nn.depthwise_conv2d(image, kernel, strides=[1, 1, 1, 1], padding='SAME', data_format=data_format)


def filter2d_v2(image, kernel, normalize_kernel=True, data_format=None):
    # image: NHWC or NCHW
    if normalize_kernel:
        kernel /= tf.reduce_sum(kernel)

    if data_format is None or data_format == "NHWC":
        image = tf.transpose(image, [0, 3, 1, 2])  # to (N, C, H, W)
    shape = tf.shape(image)
    static_shape = image.shape
    image = tf.reshape(image, [-1, shape[2], shape[3], 1])  # to (N*C, H, W, 1)

    image = tf.nn.conv2d(image, kernel[:, :, None, None], strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC')

    image = tf.reshape(image, shape)
    image.set_shape(static_shape)
    if data_format is None or data_format == "NHWC":
        image = tf.transpose(image, [0, 2, 3, 1])

    return image


filter2d = filter2d_v2


def gaussian_kernel2d(kernel_radias, std):
    d = tf.distributions.Normal(0.0, float(std))
    vals = d.prob(tf.range(start=-kernel_radias, limit=kernel_radias + 1, dtype=tf.float32))
    kernel = vals[:, None] * vals[None, :]
    return kernel


def gaussian_filter2d(image, kernel_radias, std, normalize_kernel=True, data_format=None):
    # image: NHWC or NCHW
    kernel = gaussian_kernel2d(kernel_radias, std)
    return filter2d(image, kernel, normalize_kernel=normalize_kernel, data_format=None)
