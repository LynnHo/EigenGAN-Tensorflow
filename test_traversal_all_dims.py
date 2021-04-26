import copy
import functools
import os

import imlib as im
import numpy as np
import pylib as py
import scipy
import tensorflow as tf
import tflib as tl

import module


# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--n_traversal', type=int, default=100)
py.arg('--n_traversal_point', type=int, default=17)
py.arg('--truncation_threshold', type=float, default=1.5)

py.arg('--experiment_name', default='default')
args_ = py.args()

# output_dir
output_dir = py.join('output', args_.experiment_name)

# save settings
args = py.args_from_yaml(py.join(output_dir, 'settings.yml'))
args.__dict__.update(args_.__dict__)

sess = tl.session()


# ==============================================================================
# =                                   graph                                    =
# ==============================================================================

def traversal_graph():

    # ======================================
    # =               graph                =
    # ======================================

    if not os.path.exists(py.join(output_dir, 'generator.pb')):
        # model
        G_test = functools.partial(module.G(scope='G_test'), n_channels=args.n_channels, use_gram_schmidt=args.g_loss_weight_orth_loss == 0, training=False)

        # placeholders & inputs
        zs = [tf.placeholder(dtype=tf.float32, shape=[args.n_traversal, z_dim]) for z_dim in args.z_dims]
        eps = tf.placeholder(dtype=tf.float32, shape=[args.n_traversal, args.eps_dim])

        # generate
        x_f = G_test(zs, eps, training=False)

        L = tl.tensors_filter(G_test.func.variables, 'L')
    else:
        # load freezed model
        with tf.gfile.GFile(py.join(output_dir, 'generator.pb'), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='generator')

        # placeholders & inputs
        zs = [sess.graph.get_tensor_by_name('generator/z_%d:0' % i) for i in range(len(args.z_dims))]
        eps = sess.graph.get_tensor_by_name('generator/eps:0')

        # sample graph
        x_f = sess.graph.get_tensor_by_name('generator/x_f:0')

        L = [sess.graph.get_tensor_by_name('generator/L_%d:0' % i) for i in range(len(args.z_dims))]

    # ======================================
    # =            run function            =
    # ======================================

    save_dir = './output/%s/samples_testing/traversal/all_dims/traversal_%d_%.2f' % (args.experiment_name, args.n_traversal_point, args.truncation_threshold)
    py.mkdir(save_dir)

    def run():
        zs_ipt_fixed = [scipy.stats.truncnorm.rvs(-args.truncation_threshold, args.truncation_threshold, size=[args.n_traversal, z_dim]) for z_dim in args.z_dims]
        eps_ipt = scipy.stats.truncnorm.rvs(-args.truncation_threshold, args.truncation_threshold, size=[args.n_traversal, args.eps_dim])

        left = -4.5
        right = 4.5
        for layer_idx in range(len(args.z_dims)):
            for eigen_idx in range(args.z_dims[layer_idx]):
                L_opt = sess.run(L)
                l = layer_idx
                j = eigen_idx
                i = np.argsort(np.abs(L_opt[l]))[::-1][j]

                x_f_opts = []
                vals = np.linspace(left, right, args.n_traversal_point)
                for v in vals:
                    zs_ipt = copy.deepcopy(zs_ipt_fixed)
                    zs_ipt[l][:, i] = v
                    feed_dict = {z: z_ipt for z, z_ipt in zip(zs, zs_ipt)}
                    feed_dict.update({eps: eps_ipt})
                    x_f_opt = sess.run(x_f, feed_dict=feed_dict)
                    x_f_opts.append(x_f_opt)

                sample = np.concatenate(x_f_opts, axis=2)
                for ii in range(args.n_traversal):
                    im.imwrite(sample[ii], '%s/%04d_%d-%d-%.3f-%d.jpg' % (save_dir, ii, l, j, np.abs(L_opt[l][i]), i))

    return run

traversal = traversal_graph()


# ==============================================================================
# =                                   train                                    =
# ==============================================================================

# init
if not os.path.exists(py.join(output_dir, 'generator.pb')):
    checkpoint, _, _ = tl.init(py.join(output_dir, 'checkpoints'), session=sess)

traversal()
sess.close()
