import datetime
import os

import numpy as np
import psutil
from scipy import stats

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


def calc_mse(_ab, _points):
    return SquaredErrorCalculator().general_error(_ab, _points)


def calc_rmse(_ab, _points):
    return np.sqrt(calc_mse(_ab, _points))


def calc_logcosh(_ab, _points):
    return np.mean(
        np.array(list(map(lambda _point: np.log(np.cosh(_ab[0] * _point[0] + _ab[1] - _point[1])), _points))))


def gd(
        points,
        error: Error = SquaredErrorCalculator(),
        lr=0.1,
        ab=None,
        iterations=10000,
        check_batch=50,
        eps=5e-2,
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
        ab=None,
        iterations=10000,
        check_batch=50,
        eps=5e-2,
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


def normalised_mini(
        points,
        error: Error = SquaredErrorCalculator(),
        lr=0.1,
        ab=None,
        iterations=10000,
        check_batch=50,
        eps=5e-2,
        optimization: Optimization = DefaultOptimization(),
        batch_size=1,
):
    # scales_points = points / np.linalg.norm(points, axis=0)
    scales_points = points / np.array([stats.boxcox(points[:, 0])[0], 1])

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
        eps=5e-2,
        optimization: Optimization = DefaultOptimization(),
        batch_size=1
):
    n = points.shape[0]
    if ab is None:
        ab = np.array([
            np.random.uniform(-1.0 / 2.0 / n, 1.0 / 2.0 / n),
            np.random.uniform(-1.0 / 2.0 / n, 1.0 / 2.0 / n)
        ])

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
    meta['rmse'] = calc_rmse(ab, points)
    meta['logcosh'] = calc_logcosh(ab, points)
    meta['iterations'] = meta['gradient_call_count'] / batch_size

    return ab, meta
