import functools

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflib as tl


class D(tl.Module):

    def call(self,
             x,
             dim_10=4,
             fc_dim=1024,
             n_downsamplings=6,
             weight_norm='none',
             feature_norm='none',
             act=tf.nn.leaky_relu,
             training=True):
        MAX_DIM = 512
        nd = lambda size: min(int(2**(10 - np.log2(size)) * dim_10), MAX_DIM)

        w_norm = tl.get_weight_norm(weight_norm, training)
        conv = functools.partial(tl.conv2d, weights_initializer=tl.get_initializer(act), weights_normalizer_fn=w_norm, weights_regularizer=slim.l2_regularizer(1.0))
        fc = functools.partial(tl.fc, weights_initializer=tl.get_initializer(act), weights_normalizer_fn=w_norm, weights_regularizer=slim.l2_regularizer(1.0))

        f_norm = tl.get_feature_norm(feature_norm, training, updates_collections=None)
        conv_norm_act = functools.partial(conv, normalizer_fn=f_norm, activation_fn=act)

        h = x
        h = act(conv(h, nd(h.shape[1].value), 7, 1))
        for i in range(n_downsamplings):
            # h = conv_norm_act(h, nd(h.shape[1].value // 2), 4, 2)
            h = conv_norm_act(h, nd(h.shape[1].value), 3, 1)
            h = conv_norm_act(h, nd(h.shape[1].value // 2), 3, 2)
        h = conv_norm_act(h, nd(h.shape[1].value), 3, 1)
        h = slim.flatten(h)
        h = act(fc(h, min(fc_dim, MAX_DIM)))
        logit = fc(h, 1)

        return logit


class G(tl.Module):

    def call(self,
             zs,
             eps,
             dim_10=4,
             n_channels=3,
             weight_norm='none',
             feature_norm='none',
             act=tf.nn.leaky_relu,
             use_gram_schmidt=True,
             training=True):
        MAX_DIM = 512
        nd = lambda size: min(int(2**(10 - np.log2(size)) * dim_10), MAX_DIM)

        w_norm = tl.get_weight_norm(weight_norm, training)
        transposed_w_norm = tl.get_weight_norm(weight_norm, training, transposed=True)
        fc = functools.partial(tl.fc, weights_initializer=tl.get_initializer(act), weights_normalizer_fn=w_norm, weights_regularizer=slim.l2_regularizer(1.0))
        conv = functools.partial(tl.conv2d, weights_initializer=tl.get_initializer(act), weights_normalizer_fn=w_norm, weights_regularizer=slim.l2_regularizer(1.0))
        dconv = functools.partial(tl.dconv2d, weights_initializer=tl.get_initializer(act), weights_normalizer_fn=transposed_w_norm, weights_regularizer=slim.l2_regularizer(1.0))
        f_norm = tl.get_feature_norm(feature_norm, training, updates_collections=None)
        f_norm = (lambda x: x) if f_norm is None else f_norm

        def orthogonal_regularizer(U):
            with tf.name_scope('orthogonal_regularizer'):
                U = tf.reshape(U, [-1, U.shape[-1]])
                orth = tf.matmul(tf.transpose(U), U)
                tf.add_to_collections(['orth'], orth)
                return 0.5 * tf.reduce_sum((orth - tf.eye(U.shape[-1].value)) ** 2)

        h = fc(eps, 4 * 4 * nd(4))
        h = tf.reshape(h, [-1, 4, 4, nd(4)])

        for i, z in enumerate(zs):
            height = width = 4 * 2 ** i

            U = tf.get_variable('U_%d' % i,
                                shape=[height, width, nd(height), z.shape[-1]],
                                initializer=tf.initializers.orthogonal(),
                                regularizer=orthogonal_regularizer,
                                trainable=True)
            if use_gram_schmidt:
                U = tf.transpose(tf.reshape(U, [-1, U.shape[-1]]))
                U = tl.gram_schmidt(U)
                U = tf.reshape(tf.transpose(U), [height, width, nd(height), z.shape[-1]])

            L = tf.get_variable('L_%d' % i,
                                shape=[z.shape[-1]],
                                initializer=tf.initializers.constant([3 * i for i in range(z.shape[-1], 0, -1)]),
                                trainable=True)

            mu = tf.get_variable('mu_%d' % i,
                                 shape=[height, width, nd(height)],
                                 initializer=tf.initializers.zeros(),
                                 trainable=True)

            h_ = tf.reduce_sum(U[None, ...] * (L[None, :] * z)[:, None, None, None, :], axis=-1) + mu[None, ...]

            h_1 = dconv(h_, nd(height), 1, 1)
            h_2 = dconv(h_, nd(height * 2), 3, 2)

            h = dconv(act(f_norm(h + h_1)), nd(height * 2), 3, 2)
            h = dconv(act(f_norm(h + h_2)), nd(height * 2), 3, 1)

        x = tf.tanh(conv(act(h), n_channels, 7, 1))

        return x
