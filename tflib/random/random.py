import tensorflow as tf


def truncated_normal(shape, mean=0.0, stddev=1.0, minval=-2.0, maxval=2.0, seed=None):
    norm = tf.distributions.Normal(0.0, 1.0)
    cdf_alpha = norm.cdf((minval - mean) / stddev)
    cdf_beta = norm.cdf((maxval - mean) / stddev)
    sample = norm.quantile(cdf_alpha + tf.random.uniform(shape=shape, seed=seed) * (cdf_beta - cdf_alpha)) * stddev + mean
    return sample
