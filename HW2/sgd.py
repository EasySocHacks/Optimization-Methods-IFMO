import datetime
import os

import numpy as np
import psutil

from HW2.error_calculator import SquaredErrorCalculator, Error
from HW2.optimization import DefaultOptimization, Optimization
from HW2.regression_generator import generate_regression
from HW2.visualization import visualize_regression_point, visualize_line


def get_process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss + mem_info.vms + mem_info.shared


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


def normalised_mini(
        points,
        error: Error = SquaredErrorCalculator(),
        lr=0.1,
        ab=np.array([np.random.uniform(-100, 100), np.random.uniform(-100, 100)]),
        iterations=10000,
        check_batch=50,
        eps=1e-5,
        optimization: Optimization = DefaultOptimization(),
        batch_size=1
):
    return minibatch_gd(
        points / np.linalg.norm(points),
        error,
        lr,
        ab,
        iterations,
        check_batch,
        eps,
        optimization,
        batch_size
    )


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
        for _ in range(batch_size):
            pid = np.random.randint(0, n)
            point = points[pid]

            gradient_a, gradient_b = optimization.gradient(
                ab,
                point,
                error
            )

            ab_grad += np.array([gradient_a, gradient_b])

        ab += optimization.relax(lr, ab_grad / batch_size)

    meta["max"] = get_process_memory()
    meta["maximum-after"] = meta["max"] - meta["before"]
    meta['time'] = (datetime.datetime.now() - start_time).total_seconds()
    meta['smape'] = calc_smape(ab, points)
    return ab, meta


if __name__ == "__main__":
    f, points = generate_regression()

    visualize_regression_point(f, points)

    ab, meta = minibatch_gd(points)

    print(ab)

    # for point in meta["points"]:
    #     print(point)
    #
    for key, value in meta.items():
        print(key, value)
    visualize_line(ab, points)
