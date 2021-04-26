import tensorflow as tf

from tensorflow.python.client import device_lib


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

gpus = get_available_gpus


def split_nest(nest, num_or_size_splits, axis=0):
    """Split nested structure.

    Examples
    --------
    >>> split_nest({'a': shape(10, 20), 'b': shape(4, 15)}, 2, axis=0)
    >>> [{'a': shape(5, 20), 'b': shape(2, 15)}, {'a': shape(5, 20), 'b': shape(2, 15)}]

    """
    flatten = tf.nest.flatten(nest)
    split_flatten = [tf.split(x, num_or_size_splits, axis=axis) for x in flatten]
    return [tf.nest.pack_sequence_as(nest, x) for x in zip(*split_flatten)]


def parameter_server_strategy_run(devices, fn, split_args, split_kwargs=None):
    split_kwargs = [{}] * len(devices) if split_kwargs is None else split_kwargs

    assert len(devices) == len(split_args) == len(split_kwargs)

    split_returns = []
    for device, args, kwargs in zip(devices, split_args, split_kwargs):
        with tf.device(device):
            args = args if isinstance(args, (list, tuple)) else (args,)
            split_returns.append(fn(*args, **kwargs))

    return split_returns

parellel_run = parameter_server_strategy_run


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Parameters
    ----------
    tower_grads:
        List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.

    Returns
    -------
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.

    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
