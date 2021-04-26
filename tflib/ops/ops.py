import tensorflow as tf


def tile_concat(a_list, b_list=None):
    # tile all elements of `b_list` and then concat `a_list + b_list` along the channel axis
    # `a` shape: (N, H, W, C_a)
    # `b` shape: can be (N, 1, 1, C_b) or (N, C_b)
    if b_list is None:
        b_list = []
    a_list = list(a_list) if isinstance(a_list, (list, tuple)) else [a_list]
    b_list = list(b_list) if isinstance(b_list, (list, tuple)) else [b_list]
    for i, b in enumerate(b_list):
        b = tf.reshape(b, [-1, 1, 1, b.shape[-1]])
        b = tf.tile(b, [1, a_list[0].shape[1], a_list[0].shape[2], 1])
        b_list[i] = b
    return tf.concat(a_list + b_list, axis=-1)


def reshape(x, shape):
    x = tf.convert_to_tensor(x)
    shape = [x.shape[i] if shape[i] == 0 else shape[i] for i in range(len(shape))]
    shape = [tf.shape(x)[i] if shape[i] is None else shape[i] for i in range(len(shape))]
    return tf.reshape(x, shape)


def minmax_norm(x, epsilon=1e-12):
    x = tf.to_float(x)
    min_val = tf.reduce_min(x)
    max_val = tf.reduce_max(x)
    norm_x = (x - min_val) / tf.maximum((max_val - min_val), epsilon)
    return norm_x


def gram_schmidt(vectors):
    """Gram-Schmidt process. Modified from https://stackoverflow.com/questions/48119473.

    Parameters
    ----------
        vectors: 2D tensor - [v1, v2, ...]

    """
    basis = (vectors[0:1, :] / tf.norm(vectors[0:1, :]))
    for i in range(1, vectors.shape[0]):
        v = vectors[i:i + 1, :]
        w = v - tf.matmul(tf.matmul(v, basis, transpose_b=True), basis)
        basis = tf.concat([basis, w / tf.norm(w)], axis=0)
    return basis
