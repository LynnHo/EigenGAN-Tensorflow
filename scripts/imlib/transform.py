import math

import numpy as np
import skimage.color as color
import skimage.transform as transform
import skimage.util as util


rgb2gray = color.rgb2gray
gray2rgb = color.gray2rgb

imresize = transform.resize
imrescale = transform.rescale


def imcrop(image, x1, y1, x2, y2, pad_mode='constant', **pad_kwargs):
    """Crop an image with padding non-exisiting range.

    Parameters
    ----------
    pad_mode:
        To be passed to skimage.util.pad as `mode` parameter.
    pad_kwargs:
        To be passed to skimage.util.pad.

    """
    before_h = after_h = before_w = after_w = 0

    if y2 > image.shape[0]:
        after_h = y2 - image.shape[0]
    if y1 < 0:
        before_h = -y1
    if x2 > image.shape[1]:
        after_w = x2 - image.shape[1]
    if x1 < 0:
        before_w = -x1

    x1 += before_w
    x2 += before_w
    y1 += before_h
    y2 += before_h

    image = util.pad(image,
                     [(before_h, after_h), (before_w, after_w)] + [(0, 0)] * (image.ndim - 2),
                     mode=pad_mode,
                     **pad_kwargs)

    return image[y1:y2, x1:x2, ...]


def immerge(images, n_rows=None, n_cols=None, padding=0, pad_value=0):
    """Merge images to an image with (n_rows * h) * (n_cols * w).

    Parameters
    ----------
    images : numpy.array or object which can be converted to numpy.array
        Images in shape of N * H * W(* C=1 or 3).

    """
    images = np.array(images)
    n = images.shape[0]
    if n_rows:
        n_rows = max(min(n_rows, n), 1)
        n_cols = int(n - 0.5) // n_rows + 1
    elif n_cols:
        n_cols = max(min(n_cols, n), 1)
        n_rows = int(n - 0.5) // n_cols + 1
    else:
        n_rows = int(n ** 0.5)
        n_cols = int(n - 0.5) // n_rows + 1

    h, w = images.shape[1], images.shape[2]
    shape = (h * n_rows + padding * (n_rows - 1),
             w * n_cols + padding * (n_cols - 1))
    if images.ndim == 4:
        shape += (images.shape[3],)
    img = np.full(shape, pad_value, dtype=images.dtype)

    for idx, image in enumerate(images):
        i = idx % n_cols
        j = idx // n_cols
        img[j * (h + padding):j * (h + padding) + h,
            i * (w + padding):i * (w + padding) + w, ...] = image

    return img


def grid_split(image, h, w):
    """Split the image into a grid."""
    n_rows = math.ceil(image.shape[0] / h)
    n_cols = math.ceil(image.shape[1] / w)

    rows = []
    for r in range(n_rows):
        cols = []
        for c in range(n_cols):
            cols.append(image[r * h: (r + 1) * h, c * w: (c + 1) * w, ...])
        rows.append(cols)

    return rows


def grid_merge(grid, padding=(0, 0), pad_value=(0, 0)):
    """Merge the grid as an image."""
    padding = padding if isinstance(padding, (list, tuple)) else [padding, padding]
    pad_value = pad_value if isinstance(pad_value, (list, tuple)) else [pad_value, pad_value]

    new_rows = []
    for r, row in enumerate(grid):
        new_cols = []
        for c, col in enumerate(row):
            if c != 0:
                new_cols.append(np.full([col.shape[0], padding[1], col.shape[2]], pad_value[1], dtype=col.dtype))
            new_cols.append(col)

        new_cols = np.concatenate(new_cols, axis=1)
        if r != 0:
            new_rows.append(np.full([padding[0], new_cols.shape[1], new_cols.shape[2]], pad_value[0], dtype=new_cols.dtype))
        new_rows.append(new_cols)

    grid_merged = np.concatenate(new_rows, axis=0)

    return grid_merged
