import random

import numpy as np

from HW1.coord_relaxer import BasicCoordRelaxer
from scheduler import *


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


def gradient_descent(f, dim, lr=0.1, iterations=10000, scale=100, check_batch=50, eps=1e-5, scheduler=EmptyScheduler(),
                     coord_relaxer=BasicCoordRelaxer()):
    meta = {
        "gradient_call_count": 0,
        "function_call_count": 0,
        "points": np.array([], dtype=np.float64).reshape(0, 2)
    }

    coord = np.random.rand(dim) * random.randint(-scale, scale)

    for i in range(iterations):
        meta["points"] = np.append(
            meta["points"],
            np.array([coord, f(coord)]).reshape(1, 2),
            axis=0
        )

        meta["function_call_count"] += 1

        avg_changes = np.average(
            np.abs(np.average(meta["points"][-check_batch:-1, 1]) - meta["points"][-check_batch:-1, 1]))
        if meta["points"].shape[0] > check_batch and avg_changes < eps:
            break

        (coord, _), relax_meta = coord_relaxer.relax(coord, lr, gradient(f, coord))
        meta["function_call_count"] += relax_meta["function_call_count"]

        lr = scheduler.decay_lr(i, lr)

        meta["gradient_call_count"] += 1

    return (coord, f(coord)), meta
