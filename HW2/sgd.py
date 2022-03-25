import datetime
import os

import numpy as np
import psutil

from HW2.error_calculator import SquaredErrorCalculator, Error
from HW2.optimization import DefaultOptimization, Optimization


def get_process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    # TODO: OS-dependent
    return mem_info.rss + mem_info.vms  # + mem_info.shared


def calc_smape(_ab, _points):
    smape = 0
    for _point in _points:
        y_pr = _ab[0] * _point[0] + _ab[1]
        if y_pr == 0 and _point[-1] == 0:
            smape += 0
        else:
            smape += np.abs(y_pr - _point[-1]) / (np.abs(y_pr) + np.abs(_point[-1]))
    return smape / len(_points)


def gd(
        points,
        error: Error = SquaredErrorCalculator(),
        lr=0.1,
        ab=None,
        iterations=10000,
        check_batch=50,
        eps=1e-5,
        optimization: Optimization = DefaultOptimization(),
):
    return minibatch_gd(
        points,
        error,
        lr,
        ab,
        iterations,
        check_batch,
        eps,
        optimization,
        len(points)
    )


def sgd(
        points,
        error: Error = SquaredErrorCalculator(),
        lr=0.1,
        ab=np.array([np.random.uniform(-100, 100), np.random.uniform(-100, 100)]),
        iterations=10000,
        check_batch=50,
        eps=1e-5,
        optimization: Optimization = DefaultOptimization(),
):
    return minibatch_gd(
        points,
        error,
        lr,
        ab,
        iterations,
        check_batch,
        eps,
        optimization,
        1
    )


def scaled_mini(
        points,
        error: Error = SquaredErrorCalculator(),
        lr=0.1,
        ab=np.array([np.random.uniform(-100, 100), np.random.uniform(-100, 100)]),
        iterations=10000,
        check_batch=50,
        eps=1e-5,
        optimization: Optimization = DefaultOptimization(),
        batch_size=1,
        point_scale=1
):
    scales_points = points * np.array([point_scale, 1])

    return minibatch_gd(
        scales_points,
        error,
        lr,
        ab,
        iterations,
        check_batch,
        eps,
        optimization,
        batch_size
    ), scales_points


def minibatch_gd(
        points,
        error: Error = SquaredErrorCalculator(),
        lr=0.1,
        ab=None,
        iterations=10000,
        check_batch=50,
        eps=1e-5,
        optimization: Optimization = DefaultOptimization(),
        batch_size=1
):
    if ab is None:
        ab = np.array([1.0 / len(points), 1.0 / len(points)])
    n = points.shape[0]
    start_time = datetime.datetime.now()

    meta = {
        "points": np.array([], dtype=np.float64).reshape(0, 2),
        "before": get_process_memory(),
        "gradient_call_count": 0,
        "function_call_count": 0
    }

    for i in range(iterations):
        meta["points"] = np.append(
            meta["points"], ab.reshape(1, 2),
            axis=0
        )

        if meta["points"].shape[0] > check_batch:
            avg_changes = np.average(
                np.abs(np.average(meta["points"][-check_batch:-1, 1]) - meta["points"][-check_batch:-1, 1]))
            if avg_changes < eps:
                break

        ab_grad = np.zeros(2)

        rand_points = np.random.permutation(points)[:batch_size]
        # while len(rand_points) > batch_size:
        #     pid = np.random.randint(0, len(rand_points))
        #     rand_points[pid], rand_points[-1] = rand_points[-1], rand_points[pid]
        #     rand_points = np.resize(rand_points, (len(rand_points) - 1, 2))

        for point in rand_points:
            gradient_a, gradient_b = optimization.gradient(
                ab,
                point,
                error
            )
            meta['gradient_call_count'] += 1
            ab_grad += np.array([gradient_a, gradient_b])

        ab += optimization.relax(lr, ab_grad / batch_size)

    meta["max"] = get_process_memory()
    meta["maximum-after"] = meta["max"] - meta["before"]
    meta['time'] = (datetime.datetime.now() - start_time).total_seconds()
    meta['smape'] = calc_smape(ab, points)
    return ab, meta
