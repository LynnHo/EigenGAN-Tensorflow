import tensorflow as tf


# ======================================
# =               f-GANs               =
# ======================================

def get_divergence_fn(divergence):
    if divergence in ['Kullback-Leibler', 'KL']:
        def activation_fn(v):
            return v

        def conjugate_fn(t):
            return tf.exp(t - 1)

    elif divergence == 'Reverse-KL':
        def activation_fn(v):
            return -tf.exp(-v)

        def conjugate_fn(t):
            return -1 - tf.log(-t)

    elif divergence == 'Pearson-X2':
        def activation_fn(v):
            return v

        def conjugate_fn(t):
            return 0.25 * t * t + t

    elif divergence == 'Squared-Hellinger':
        def activation_fn(v):
            return 1 - tf.exp(-v)

        def conjugate_fn(t):
            return t / (1 - t)

    elif divergence in ['Jensen-Shannon', 'JS']:
        def activation_fn(v):
            return tf.log(2.0) - tf.log(1 + tf.exp(-v))

        def conjugate_fn(t):
            return -tf.log(2 - tf.exp(t))

    elif divergence == 'GAN':
        def activation_fn(v):
            return -tf.log(1 + tf.exp(-v))

        def conjugate_fn(t):
            return -tf.log(1 - tf.exp(t))

    else:
        raise NotImplementedError

    return activation_fn, conjugate_fn


def get_fgan_losses_fn(divergence, tricky=True):
    activation_fn, conjugate_fn = get_divergence_fn(divergence)

    def d_loss_fn(r_logit, f_logit):
        r_loss = -tf.reduce_mean(activation_fn(r_logit))
        f_loss = tf.reduce_mean(conjugate_fn(activation_fn(f_logit)))
        return r_loss, f_loss

    def g_loss_fn_theoretical(f_logit):
        f_loss = -tf.reduce_mean(conjugate_fn(activation_fn(f_logit)))
        return f_loss

    def g_loss_fn_tricky(f_logit):
        f_loss = -tf.reduce_mean(activation_fn(f_logit))
        return f_loss

    g_loss_fn = g_loss_fn_theoretical if not tricky else g_loss_fn_tricky

    return d_loss_fn, g_loss_fn


# ======================================
# =             Other GANs             =
# ======================================

def get_gan_losses_fn():
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def d_loss_fn(r_logit, f_logit):
        r_loss = bce(tf.ones_like(r_logit), r_logit)
        f_loss = bce(tf.zeros_like(f_logit), f_logit)
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = bce(tf.ones_like(f_logit), f_logit)
        return f_loss

    return d_loss_fn, g_loss_fn


def get_hinge_v1_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = tf.reduce_mean(tf.maximum(1 - r_logit, 0))
        f_loss = tf.reduce_mean(tf.maximum(1 + f_logit, 0))
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = tf.reduce_mean(tf.maximum(1 - f_logit, 0))
        return f_loss

    return d_loss_fn, g_loss_fn


def get_hinge_v2_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = tf.reduce_mean(tf.maximum(1 - r_logit, 0))
        f_loss = tf.reduce_mean(tf.maximum(1 + f_logit, 0))
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = tf.reduce_mean(- f_logit)
        return f_loss

    return d_loss_fn, g_loss_fn


def get_lsgan_losses_fn():
    mse = tf.keras.losses.MeanSquaredError()

    def d_loss_fn(r_logit, f_logit):
        r_loss = mse(tf.ones_like(r_logit), r_logit)
        f_loss = mse(tf.zeros_like(f_logit), f_logit)
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = mse(tf.ones_like(f_logit), f_logit)
        return f_loss

    return d_loss_fn, g_loss_fn


def get_wgan_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = - tf.reduce_mean(r_logit)
        f_loss = tf.reduce_mean(f_logit)
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = - tf.reduce_mean(f_logit)
        return f_loss

    return d_loss_fn, g_loss_fn


def get_adversarial_losses_fn(mode):
    if mode == 'gan':
        return get_gan_losses_fn()
    elif mode == 'hinge_v1':
        return get_hinge_v1_losses_fn()
    elif mode == 'hinge_v2':
        return get_hinge_v2_losses_fn()
    elif mode == 'lsgan':
        return get_lsgan_losses_fn()
    elif mode == 'wgan':
        return get_wgan_losses_fn()
    elif isinstance(mode, (list, tuple)) and mode[0] == 'fgan':
        if len(mode) == 2:
            return get_fgan_losses_fn(mode[1])
        elif len(mode) == 3:
            return get_fgan_losses_fn(mode[1], mode[2])
    elif mode.startswith('fgan'):
        mode = mode.split('_')
        if len(mode) == 2:
            return get_fgan_losses_fn(mode[1])
        elif len(mode) == 3:
            return get_fgan_losses_fn(mode[1], True if mode[2] == 'tricky' else False)
    else:
        raise NotImplementedError
