import random

import numpy as np

from scheduler import *


def gradient(f, coord, h=1e-5):
    grad_list = np.array([], dtype=np.float64)
    for index, x in enumerate(coord):
        f_left = coord.copy()
        f_left[index] = f(x - h)

        f_right = coord.copy()
        f_right[index] = f(x + h)

        grad_2d = (f_right - f_left) / (2 * h)
        grad_list = np.concatenate((grad_list, grad_2d))

    return np.array(grad_list, dtype=np.float64)


def gradient_descent(f, dim, lr=0.1, iterations=1000, scale=100, check_batch=50, eps=1e-5, scheduler=EmptyScheduler()):
    meta = {
        "gradient_call_count": 0,
        "function_call_count": 0,
        "points": np.array([], dtype=np.float64).reshape(0, 2)
    }

    coord = np.random.rand(dim) * random.randint(-scale, scale)

    for i in range(iterations):
        meta["points"] = np.append(
            meta["points"],
            np.array([coord, f(coord)], dtype=np.float64).reshape(1, 2),
            axis=0
        )

        meta["function_call_count"] += 1

        avg_changes = np.average(
            np.abs(np.average(meta["points"][-check_batch:-1, 1]) - meta["points"][-check_batch:-1, 1]))
        if meta["points"].shape[0] > check_batch and avg_changes < eps:
            break

        coord = coord - lr * gradient(f, coord)
        lr = scheduler.decay_lr(i, lr)

        meta["gradient_call_count"] += 1

    return (coord, f(coord)), meta
