import random

import numpy as np

from HW1.coord_relaxer import BasicCoordRelaxer
from HW1.gradient import gradient
from scheduler import *


def gradient_descent(f,
                     dim,
                     coord=None,
                     lr=0.1,
                     iterations=10000,
                     check_batch=50,
                     eps=1e-5,
                     scheduler=EmptyScheduler(),
                     coord_relaxer=BasicCoordRelaxer()):
    if coord is None:
        coord = np.random.rand(dim) * random.randint(-100, 100)

    meta = {
        "gradient_call_count": 0,
        "function_call_count": 0,
        "points": np.array([], dtype=np.float64).reshape(0, 2)
    }

    for i in range(iterations):
        meta["points"] = np.append(
            meta["points"],
            np.array([coord, f(coord)]).reshape(1, 2),
            axis=0
        )

        meta["function_call_count"] += 1

        if meta["points"].shape[0] > check_batch:
            avg_changes = np.average(
                np.abs(np.average(meta["points"][-check_batch:-1, 1]) - meta["points"][-check_batch:-1, 1]))
            if avg_changes < eps:
                break

        (coord, _), relax_meta = coord_relaxer.relax(coord, lr, gradient(f, coord))
        meta["function_call_count"] += relax_meta["function_call_count"]

        lr = scheduler.decay_lr(i, lr)

        meta["gradient_call_count"] += 1

    return (coord, f(coord)), meta
