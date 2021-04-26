import numpy as np
import pylib as py
import tensorflow as tf
import tflib as tl


def make_dataset(img_dir,
                 batch_size,
                 load_size=286,
                 crop_size=256,
                 n_channels=3,
                 training=True,
                 drop_remainder=True,
                 shuffle=True,
                 repeat=1):
    img_paths = sorted(py.glob(img_dir, '*'))

    if shuffle:
        img_paths = np.random.permutation(img_paths)

    if training:
        def _map_fn(img):
            if n_channels == 1:
                img = tf.image.rgb_to_grayscale(img)
            img = tf.image.resize(img, [load_size, load_size])
            img = tf.image.random_flip_left_right(img)
            img = tl.center_crop(img, size=crop_size)
            # img = tf.image.random_crop(img, [crop_size, crop_size, n_channels])
            img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
            return img
    else:
        def _map_fn(img):
            if n_channels == 1:
                img = tf.image.rgb_to_grayscale(img)
            img = tf.image.resize(img, [load_size, load_size])
            img = tl.center_crop(img, size=crop_size)
            img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
            return img

    dataset = tl.disk_image_batch_dataset(img_paths,
                                          batch_size,
                                          drop_remainder=drop_remainder,
                                          map_fn=_map_fn,
                                          shuffle=shuffle,
                                          repeat=repeat)

    if drop_remainder:
        len_dataset = len(img_paths) // batch_size
    else:
        len_dataset = int(np.ceil(len(img_paths) / batch_size))

    return dataset, len_dataset
