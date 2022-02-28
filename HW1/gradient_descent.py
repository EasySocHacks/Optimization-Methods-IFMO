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
    """
    Implementing gradient descent algorithm.

    coordinates 'coord' generates automatically in case its sets to None, or you can set it by yourself otherwise.

    The method stops either then 'iterations' count reached or the last 'check_batch' point's of current
    gradient descent result returns result from function 'f' within the 'eps' error.

    :param f: a function to do gradient descent on
    :param dim: dimension of function 'f' working with
    :param coord: optional coordinates setter
    :param lr: gradient descent learning rate
    :param iterations: gradient descent maximum iteration count
    :param check_batch: a batch size to check stop condition
    :param eps: 'check_batch' points error to check stop condition
    :param scheduler: a scheduler to schedule learning rate
    :param coord_relaxer: a relaxer to change coordinates over iterations
    :return: a local minimum estimation
    """
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
        if "gradient_call_count" in relax_meta:
            meta["gradient_call_count"] += relax_meta["gradient_call_count"]

        lr = scheduler.decay_lr(i + 1, lr)

        meta["gradient_call_count"] += 1

    return (coord, f(coord)), meta
