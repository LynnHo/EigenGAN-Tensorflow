import tensorflow as tf


def summary_statistic_v1(name_data_dict,
                         types=['mean', 'std', 'max', 'min', 'sparsity', 'histogram'],
                         name='summary_statistic_v1'):
    """(deprecated, use v2) Summary of statistics.

    Examples
    --------
    >>> summary_statistic_v1({'a': data_a, 'b': data_b})

    """
    def _summary(name, data):
        summaries = []
        if data.shape == ():
            summaries.append(tf.summary.scalar(name, data))
        else:
            if 'mean' in types:
                summaries.append(tf.summary.scalar(name + '-mean', tf.math.reduce_mean(data)))
            if 'std' in types:
                summaries.append(tf.summary.scalar(name + '-std', tf.math.reduce_std(data)))
            if 'max' in types:
                summaries.append(tf.summary.scalar(name + '-max', tf.math.reduce_max(data)))
            if 'min' in types:
                summaries.append(tf.summary.scalar(name + '-min', tf.math.reduce_min(data)))
            if 'sparsity' in types:
                summaries.append(tf.summary.scalar(name + '-sparsity', tf.math.zero_fraction(data)))
            if 'histogram' in types:
                summaries.append(tf.summary.histogram(name, data))
        return tf.summary.merge(summaries)

    with tf.name_scope(name):
        summaries = []
        for name, data in name_data_dict.items():
            summaries.append(_summary(name, data))
        return tf.summary.merge(summaries)


def summary_statistic_v2(name_data_dict,
                         step,
                         types=['mean', 'std', 'max', 'min', 'sparsity', 'histogram'],
                         name='summary_statistic_v2'):
    """Summary of statistics.

    Examples
    --------
    >>> summary_statistic_v2({'a': data_a, 'b': data_b}, tf.train.get_global_step())

    """
    def _summary(name, data):
        summaries = []
        if data.shape == ():
            summaries.append(tf.contrib.summary.scalar(name, data, step=step))
        else:
            if 'mean' in types:
                summaries.append(tf.contrib.summary.scalar(name + '-mean', tf.math.reduce_mean(data), step=step))
            if 'std' in types:
                summaries.append(tf.contrib.summary.scalar(name + '-std', tf.math.reduce_std(data), step=step))
            if 'max' in types:
                summaries.append(tf.contrib.summary.scalar(name + '-max', tf.math.reduce_max(data), step=step))
            if 'min' in types:
                summaries.append(tf.contrib.summary.scalar(name + '-min', tf.math.reduce_min(data), step=step))
            if 'sparsity' in types:
                summaries.append(tf.contrib.summary.scalar(name + '-sparsity', tf.math.zero_fraction(data), step=step))
            if 'histogram' in types:
                summaries.append(tf.contrib.summary.histogram(name, data, step=step))
        return summaries

    with tf.name_scope(name):
        summaries = {}
        for name, data in name_data_dict.items():
            summaries[name] = _summary(name, data)
        return summaries


def create_summary_statistic_v2(name_data_dict,
                                logdir,
                                step,
                                n_steps_per_record=1,
                                types=['mean', 'std', 'max', 'min', 'sparsity', 'histogram'],
                                name='summary_statistic_v2'):
    with tf.contrib.summary.create_file_writer(logdir).as_default(),\
            tf.contrib.summary.record_summaries_every_n_global_steps(n_steps_per_record, global_step=step):
        return summary_statistic_v2(name_data_dict,
                                    step,
                                    types=types,
                                    name=name)


def summary_image_v2(name_data_dict,
                     step,
                     max_images=3,
                     name='summary_image_v2'):
    with tf.name_scope(name):
        summaries = {}
        for name, data in name_data_dict.items():
            summaries[name] = tf.contrib.summary.image(name, data, max_images=max_images, step=step)
        return summaries


def create_summary_image_v2(name_data_dict,
                            logdir,
                            step,
                            n_steps_per_record=1,
                            max_images=3,
                            name='summary_image_v2'):
    with tf.contrib.summary.create_file_writer(logdir).as_default(),\
            tf.contrib.summary.record_summaries_every_n_global_steps(n_steps_per_record, global_step=step):
        return summary_image_v2(name_data_dict,
                                step,
                                max_images=max_images,
                                name=name)
