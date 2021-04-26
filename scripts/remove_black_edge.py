import imlib as im
import numpy as np
import pylib as py


# ==============================================================================
# =                                   param                                    =
# ==============================================================================

img_dir = './data/anime/original_imgs'
save_dir = './data/anime/remove_black_edge_imgs'
portion = 0.075


# ==============================================================================
# =                                    run                                     =
# ==============================================================================

def count_edge(img, eps=0.4):
    up = 0
    for i in range(img.shape[0]):
        if np.mean(img[i, ...]) + 1 < eps:
            up += 1
        else:
            break

    down = 0
    for i in range(img.shape[0] - 1, -1, -1):
        if np.mean(img[i, ...]) + 1 < eps:
            down += 1
        else:
            break

    left = 0
    for i in range(img.shape[1]):
        if np.mean(img[:, i, ...]) + 1 < eps:
            left += 1
        else:
            break

    right = 0
    for i in range(img.shape[1] - 1, -1, -1):
        if np.mean(img[:, i, ...]) + 1 < eps:
            right += 1
        else:
            break

    return up, down, left, right


def work_fn(img_name):
    img = im.imread(img_name)
    u, d, l, r = count_edge(img)
    o = max(u, d, l, r)
    if o / img.shape[0] < portion:
        img = img[o:img.shape[0] - o, o:img.shape[1] - o, ...]
        im.imwrite(img, img_name.replace(img_dir, save_dir))


py.mkdir(save_dir)
img_names = py.glob(img_dir, '*')
py.run_parallels(work_fn, img_names)
