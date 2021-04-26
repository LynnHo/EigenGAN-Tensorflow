import copy
import functools
import traceback

import imlib as im
import numpy as np
import pylib as py
import scipy
import tensorflow as tf
import tflib as tl
import tfprob
import tqdm

import data
import module


# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--img_dir', default='./data/img_celeba/aligned/align_size(572,572)_move(0.250,0.000)_face_factor(0.450)_jpg/data')
py.arg('--load_size', type=int, default=256)
py.arg('--crop_size', type=int, default=256)
py.arg('--n_channels', type=int, choices=[1, 3], default=3)

py.arg('--n_epochs', type=int, default=160)
py.arg('--epoch_start_decay', type=int, default=160)
py.arg('--batch_size', type=int, default=64)
py.arg('--learning_rate', type=float, default=1e-4)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--moving_average_decay', type=float, default=0.999)

py.arg('--n_d', type=int, default=1)  # # d updates per g update
py.arg('--adversarial_loss_mode', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'], default='hinge_v1')
py.arg('--gradient_penalty_mode', choices=['none', '1-gp', '0-gp', 'lp'], default='0-gp')
py.arg('--gradient_penalty_sample_mode', choices=['line', 'real', 'fake', 'real+fake', 'dragan', 'dragan_fake'], default='real')

py.arg('--d_loss_weight_x_gan', type=float, default=1)
py.arg('--d_loss_weight_x_gp', type=float, default=10)
py.arg('--d_lazy_reg_period', type=int, default=3)

py.arg('--g_loss_weight_x_gan', type=float, default=1)
py.arg('--g_loss_weight_orth_loss', type=float, default=1)  # if 0, use Gram–Schmidt orthogonalization (slower)

py.arg('--weight_decay', type=float, default=0)

py.arg('--z_dims', type=int, nargs='+', default=[6] * 6)
py.arg('--eps_dim', type=int, default=512)

py.arg('--n_samples', type=int, default=100)
py.arg('--n_traversal', type=int, default=5)
py.arg('--n_left_axis_point', type=int, default=10)
py.arg('--truncation_threshold', type=int, default=1.5)

py.arg('--sample_period', type=int, default=1000)
py.arg('--traversal_period', type=int, default=2500)
py.arg('--checkpoint_save_period', type=int, default=10000)

py.arg('--experiment_name', default='default')
args = py.args()

# check
assert np.log2(args.crop_size / 4) == len(args.z_dims)

# output_dir
output_dir = py.join('output', args.experiment_name)
py.mkdir(output_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)

sess = tl.session()


# ==============================================================================
# =                                    data                                    =
# ==============================================================================

train_dataset, len_train_dataset = data.make_dataset(args.img_dir, args.batch_size, load_size=args.load_size, crop_size=args.crop_size, n_channels=args.n_channels, repeat=None)
train_iter = train_dataset.make_one_shot_iterator()


# ==============================================================================
# =                                   model                                    =
# ==============================================================================

D = functools.partial(module.D(scope='D'), n_downsamplings=len(args.z_dims))
G = functools.partial(module.G(scope='G'), n_channels=args.n_channels, use_gram_schmidt=args.g_loss_weight_orth_loss == 0)
G_test = functools.partial(module.G(scope='G_test'), n_channels=args.n_channels, use_gram_schmidt=args.g_loss_weight_orth_loss == 0, training=False)

# exponential moving average
G_ema = tf.train.ExponentialMovingAverage(decay=args.moving_average_decay, name='G_ema')

# loss function
d_loss_fn, g_loss_fn = tfprob.get_adversarial_losses_fn(args.adversarial_loss_mode)


# ==============================================================================
# =                                   graph                                    =
# =============================================================================

def D_train_graph():
    # ======================================
    # =               graph                =
    # ======================================

    # placeholders & inputs
    lr = tf.placeholder(dtype=tf.float32, shape=[])
    x_r = train_iter.get_next()
    zs = [tf.random.normal([args.batch_size, z_dim]) for z_dim in args.z_dims]
    eps = tf.random.normal([args.batch_size, args.eps_dim])

    # counter
    step_cnt, _ = tl.counter()

    # optimizer
    optimizer = tf.train.AdamOptimizer(lr, beta1=args.beta_1)

    def graph_per_gpu(x_r, zs, eps):

        # generate
        x_f = G(zs, eps)

        # discriminate
        x_r_logit = D(x_r)
        x_f_logit = D(x_f)

        # loss
        x_r_loss, x_f_loss = d_loss_fn(x_r_logit, x_f_logit)
        x_gp = tf.cond(tf.equal(step_cnt % args.d_lazy_reg_period, 0),
                       lambda: tfprob.gradient_penalty(D, x_r, x_f, args.gradient_penalty_mode, args.gradient_penalty_sample_mode) * args.d_lazy_reg_period,
                       lambda: tf.constant(0.0))
        if args.d_loss_weight_x_gp == 0:
            x_gp = tf.constant(0.0)

        reg_loss = tf.reduce_sum(D.func.reg_losses)

        loss = (
            (x_r_loss + x_f_loss) * args.d_loss_weight_x_gan +
            x_gp * args.d_loss_weight_x_gp +
            reg_loss * args.weight_decay
        )

        # optim
        grads = optimizer.compute_gradients(loss, var_list=D.func.trainable_variables)

        return grads, x_r_loss, x_f_loss, x_gp, reg_loss

    split_grads, split_x_r_loss, split_x_f_loss, split_x_gp, split_reg_loss = zip(*tl.parellel_run(tl.gpus(), graph_per_gpu, tl.split_nest((x_r, zs, eps), len(tl.gpus()))))
    # split_grads, split_x_r_loss, split_x_f_loss, split_x_gp, split_reg_loss = zip(*tl.parellel_run(['cpu:0'], graph_per_gpu, tl.split_nest((x_r, zs, eps), 1)))
    grads = tl.average_gradients(split_grads)
    x_r_loss, x_f_loss, x_gp, reg_loss = [tf.reduce_mean(t) for t in [split_x_r_loss, split_x_f_loss, split_x_gp, split_reg_loss]]

    step = optimizer.apply_gradients(grads, global_step=step_cnt)

    # summary
    summary = tl.create_summary_statistic_v2(
        {'x_gan_loss': x_r_loss + x_f_loss,
         'x_gp': x_gp,
         'reg_loss': reg_loss,
         'lr': lr},
        './output/%s/summaries/D' % args.experiment_name,
        step=step_cnt,
        n_steps_per_record=10,
        name='D'
    )

    # ======================================
    # =            run function            =
    # ======================================

    def run(**pl_ipts):
        for _ in range(args.n_d):
            sess.run([step, summary], feed_dict={lr: pl_ipts['lr']})

    return run


def G_train_graph():
    # ======================================
    # =               graph                =
    # ======================================

    # placeholders & inputs
    lr = tf.placeholder(dtype=tf.float32, shape=[])
    zs = [tf.random.normal([args.batch_size, z_dim]) for z_dim in args.z_dims]
    eps = tf.random.normal([args.batch_size, args.eps_dim])

    # counter
    step_cnt, _ = tl.counter()

    # optimizer
    optimizer = tf.train.AdamOptimizer(lr, beta1=args.beta_1)

    def graph_per_gpu(zs, eps):
        # generate
        x_f = G(zs, eps)

        # discriminate
        x_f_logit = D(x_f)

        # loss
        x_f_loss = g_loss_fn(x_f_logit)
        orth_loss = tf.reduce_sum(tl.tensors_filter(G.func.reg_losses, 'orthogonal_regularizer'))
        reg_loss = tf.reduce_sum(tl.tensors_filter(G.func.reg_losses, 'l2_regularizer'))

        loss = (
            x_f_loss * args.g_loss_weight_x_gan +
            orth_loss * args.g_loss_weight_orth_loss +
            reg_loss * args.weight_decay
        )

        # optim
        grads = optimizer.compute_gradients(loss, var_list=G.func.trainable_variables)

        return grads, x_f_loss, orth_loss, reg_loss

    split_grads, split_x_f_loss, split_orth_loss, split_reg_loss = zip(*tl.parellel_run(tl.gpus(), graph_per_gpu, tl.split_nest((zs, eps), len(tl.gpus()))))
    # split_grads, split_x_f_loss, split_orth_loss, split_reg_loss = zip(*tl.parellel_run(['cpu:0'], graph_per_gpu, tl.split_nest((zs, eps), 1)))
    grads = tl.average_gradients(split_grads)
    x_f_loss, orth_loss, reg_loss = [tf.reduce_mean(t) for t in [split_x_f_loss, split_orth_loss, split_reg_loss]]

    step = optimizer.apply_gradients(grads, global_step=step_cnt)

    # moving average
    with tf.control_dependencies([step]):
        step = G_ema.apply(G.func.trainable_variables)

    # summary
    summary_dict = {'x_f_loss': x_f_loss,
                    'orth_loss': orth_loss,
                    'reg_loss': reg_loss}
    summary_dict.update({'L_%d' % i: t for i, t in enumerate(tl.tensors_filter(G.func.variables, 'L'))})
    summary_loss = tl.create_summary_statistic_v2(
        summary_dict,
        './output/%s/summaries/G' % args.experiment_name,
        step=step_cnt,
        n_steps_per_record=10,
        name='G_loss'
    )

    summary_image = tl.create_summary_image_v2(
        {'orth_U_%d' % i: t[None, :, :, None] for i, t in enumerate(tf.get_collection('orth', G.func.scope + '/'))},
        './output/%s/summaries/G' % args.experiment_name,
        step=step_cnt,
        n_steps_per_record=10,
        name='G_image'
    )

    # ======================================
    # =             model size             =
    # ======================================

    n_params, n_bytes = tl.count_parameters(G.func.variables)
    print('Model Size: n_parameters = %d = %.2fMB' % (n_params, n_bytes / 1024 / 1024))

    # ======================================
    # =            run function            =
    # ======================================

    def run(**pl_ipts):
        sess.run([step, summary_loss, summary_image], feed_dict={lr: pl_ipts['lr']})

    return run


def sample_graph():

    # ======================================
    # =               graph                =
    # ======================================

    # placeholders & inputs
    zs = [tl.truncated_normal([args.n_samples, z_dim], minval=-args.truncation_threshold, maxval=args.truncation_threshold) for z_dim in args.z_dims]
    eps = tl.truncated_normal([args.n_samples, args.eps_dim], minval=-args.truncation_threshold, maxval=args.truncation_threshold)

    # generate
    x_f = G_test(zs, eps, training=False)

    # ======================================
    # =            run function            =
    # ======================================

    save_dir = './output/%s/samples_training/sample' % (args.experiment_name)
    py.mkdir(save_dir)

    def run(epoch, iter):
        x_f_opt = sess.run(x_f)
        sample = im.immerge(x_f_opt, n_rows=int(args.n_samples ** 0.5))
        im.imwrite(sample, '%s/Epoch-%d_Iter-%d.jpg' % (save_dir, epoch, iter))

    return run


def traversal_graph():

    # ======================================
    # =               graph                =
    # ======================================

    # placeholders & inputs
    zs = [tf.placeholder(dtype=tf.float32, shape=[args.n_traversal, z_dim]) for z_dim in args.z_dims]
    eps = tf.placeholder(dtype=tf.float32, shape=[args.n_traversal, args.eps_dim])

    # generate
    x_f = G_test(zs, eps, training=False)

    # ======================================
    # =            run function            =
    # ======================================

    save_dir = './output/%s/samples_training/traversal' % (args.experiment_name)
    py.mkdir(save_dir)

    def run(epoch, iter):
        zs_ipt_fixed = [scipy.stats.truncnorm.rvs(-args.truncation_threshold, args.truncation_threshold, size=[args.n_traversal, z_dim]) for z_dim in args.z_dims]
        eps_ipt = scipy.stats.truncnorm.rvs(-args.truncation_threshold, args.truncation_threshold, size=[args.n_traversal, args.eps_dim])

        # set the first sample as the "mode"
        for l in range(len(args.z_dims)):
            zs_ipt_fixed[l][0, ...] = 0.0
        eps_ipt[0, ...] = 0.0

        L_opt = sess.run(tl.tensors_filter(G_test.func.variables, 'L'))
        for l in range(len(args.z_dims)):
            for j, i in enumerate(np.argsort(np.abs(L_opt[l]))[::-1]):
                x_f_opts = []
                vals = np.linspace(-4.5, 4.5, args.n_left_axis_point * 2 + 1)
                for v in vals:
                    zs_ipt = copy.deepcopy(zs_ipt_fixed)
                    zs_ipt[l][:, i] = v
                    feed_dict = {z: z_ipt for z, z_ipt in zip(zs, zs_ipt)}
                    feed_dict.update({eps: eps_ipt})
                    x_f_opt = sess.run(x_f, feed_dict=feed_dict)
                    x_f_opts.append(x_f_opt)

                sample = im.immerge(np.concatenate(x_f_opts, axis=2), n_rows=args.n_traversal)
                im.imwrite(sample, '%s/Epoch-%d_Iter-%d_Traversal-%d-%d-%.3f-%d.jpg' % (save_dir, epoch, iter, l, j, np.abs(L_opt[l][i]), i))

    return run


def clone_graph():
    # ======================================
    # =               graph                =
    # ======================================

    clone_tr = G_test.func.clone_from_vars(tl.tensors_filter(tl.global_variables(), 'G_ema'), var_type='trainable')
    clone_non = G_test.func.clone_from_module(G.func, var_type='nontrainable')

    # ======================================
    # =            run function            =
    # ======================================

    def run(**pl_ipts):
        sess.run([clone_tr, clone_non])

    return run


d_train_step = D_train_graph()
g_train_step = G_train_graph()
sample = sample_graph()
traversal = traversal_graph()
clone = clone_graph()


# ==============================================================================
# =                                   train                                    =
# ==============================================================================

# init
checkpoint, step_cnt, update_cnt = tl.init(py.join(output_dir, 'checkpoints'), checkpoint_max_to_keep=1, session=sess)

# learning rate schedule
lr_fn = tl.LinearDecayLR(args.learning_rate, args.n_epochs, args.epoch_start_decay)

# train
try:
    for ep in tqdm.trange(args.n_epochs, desc='Epoch Loop'):
        # learning rate
        lr_ipt = lr_fn(ep)

        for it in tqdm.trange(len_train_dataset // (args.n_d + 1), desc='Inner Epoch Loop'):
            if it + ep * (len_train_dataset // (args.n_d + 1)) < sess.run(step_cnt):
                continue
            step = sess.run(update_cnt)

            # train D
            d_train_step(lr=lr_ipt)
            # train G
            g_train_step(lr=lr_ipt)

            # save
            if step % args.checkpoint_save_period == 0:
                checkpoint.save(step, session=sess)

            # sample
            if step % args.sample_period == 0:
                clone()
                sample(ep, it)
            if step % args.traversal_period == 0:
                clone()
                traversal(ep, it)
except Exception:
    traceback.print_exc()
finally:
    clone()
    sample(ep, it)
    traversal(ep, it)
    checkpoint.save(step, session=sess)
    sess.close()
