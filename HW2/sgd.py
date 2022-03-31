import datetime
import os

import numpy as np
import psutil

from HW2.error_calculator import SquaredErrorCalculator, Error
from HW2.optimization import DefaultOptimization, Optimization


def get_process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss + mem_info.vms  # + mem_info.shared


def scalar(w, point):
    return np.sum(w[:-1] * point[:-1]) + w[-1]


def calc_smape(_ab, _points):
    smape = 0
    for _point in _points:
        y_pr = scalar(_ab, _point)
        if y_pr == 0 and _point[-1] == 0:
            smape += 0
        else:
            smape += np.abs(y_pr - _point[-1]) / (np.abs(y_pr) + np.abs(_point[-1]))
    return smape / len(_points)


def calc_mse(_ab, _points):
    return SquaredErrorCalculator().general_error(_ab, _points)


def calc_rmse(_ab, _points):
    return np.sqrt(calc_mse(_ab, _points))


def calc_logcosh(_ab, _points):
    return np.mean(
        np.array(list(map(lambda _point: np.log(np.cosh(scalar(_ab, _point) - _point[-1])), _points))))


def gd(
        points,
        start_point=None,
        error: Error = SquaredErrorCalculator(),
        lr=0.1,
        iterations=10000,
        check_batch=50,
        eps=8e-2,
        optimization: Optimization = DefaultOptimization(),
):
    return minibatch_gd(
        points,
        start_point,
        error,
        lr,
        iterations,
        check_batch,
        eps,
        optimization,
        len(points)
    )


def sgd(
        points,
        start_point=None,
        error: Error = SquaredErrorCalculator(),
        lr=0.1,
        iterations=10000,
        check_batch=50,
        eps=8e-2,
        optimization: Optimization = DefaultOptimization(),
):
    return minibatch_gd(
        points,
        start_point,
        error,
        lr,
        iterations,
        check_batch,
        eps,
        optimization,
        1
    )


def scaled_mini(
        points,
        start_point=None,
        error: Error = SquaredErrorCalculator(),
        lr=0.1,
        iterations=10000,
        check_batch=50,
        scale=1,
        eps=8e-2,
        optimization: Optimization = DefaultOptimization(),
        batch_size=1,
):
    scaled_points = points * np.append(np.ones(points.shape[1] - 1), scale)
    return minibatch_gd(
        scaled_points,
        start_point,
        error,
        lr,
        iterations,
        check_batch,
        eps,
        optimization,
        batch_size
    ), scaled_points


def minibatch_gd(
        points,
        start_point=None,
        error: Error = SquaredErrorCalculator(),
        lr=0.1,
        iterations=10000,
        check_batch=50,
        eps=8e-2,
        optimization: Optimization = DefaultOptimization(),
        batch_size=1
):
    n = points.shape[0]
    dim = points.shape[1]

    if start_point is None:
        start_point = np.random.uniform(-0.5 / n, 0.5 / n, dim)
    w = start_point.copy()
    start_time = datetime.datetime.now()

    meta = {
        "points": np.array([], dtype=np.float64).reshape(0, dim),
        "before": get_process_memory(),
        "gradient_call_count": 0,
        "function_call_count": 0
    }

    for i in range(iterations):
        meta["points"] = np.append(
            meta["points"], w.reshape(1, dim),
            axis=0
        )

        if meta["points"].shape[0] > check_batch:
            avg_changes = np.average(
                np.average(np.abs(
                    np.average(meta["points"][-check_batch:-1, :], axis=0) - meta["points"][-check_batch:-1, :]),
                    axis=0
                ))
            if avg_changes < eps:
                break

        ab_grad = np.zeros(dim)
        rand_points = np.random.permutation(points)[:batch_size]

        for point in rand_points:
            gradient = optimization.gradient(
                w,
                point,
                error
            )
            meta['gradient_call_count'] += 1
            ab_grad += gradient

        w += optimization.relax(lr, ab_grad / batch_size)

    meta["max"] = get_process_memory()
    meta["maximum-after"] = meta["max"] - meta["before"]
    meta['time'] = (datetime.datetime.now() - start_time).total_seconds()
    meta['smape'] = calc_smape(w, points)
    meta['rmse'] = calc_rmse(w, points)
    meta['logcosh'] = calc_logcosh(w, points)
    meta['iterations'] = meta['gradient_call_count'] / batch_size

    return w, meta
