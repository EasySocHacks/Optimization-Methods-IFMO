import numpy as np


def gradient(f, coord, h=1e-5):
    f_left = np.zeros(coord.shape)
    f_right = np.zeros(coord.shape)
    for i in range(coord.shape[0]):
        h_vec = np.zeros(coord.shape)
        h_vec[i] = h

        f_left[i] = f(coord - h_vec)
        f_right[i] = f(coord + h_vec)

    grad = (f_right - f_left) / (2 * h)

    return grad
