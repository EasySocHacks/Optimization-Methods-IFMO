import numpy as np

from HW2.regression_generator import generate_regression
from HW2.visualization import visualize_regression_point, visualize_line


def gd(
        points,
        error,
        lr=0.1,
        ab=np.array([np.random.uniform(-100, 100), np.random.uniform(-100, 100)]),
        iterations=10000,
        check_batch=50,
        eps=1e-5,
):
    return minibatch_gd(
        points,
        error,
        lr,
        ab,
        iterations,
        check_batch,
        eps,
        len(points)
    )


def sgd(
        points,
        error,
        lr=0.1,
        ab=np.array([np.random.uniform(-100, 100), np.random.uniform(-100, 100)]),
        iterations=10000,
        check_batch=50,
        eps=1e-5,
):
    return minibatch_gd(
        points,
        error,
        lr,
        ab,
        iterations,
        check_batch,
        eps,
        1
    )


def normalised_mini(
        points,
        error,
        lr=0.1,
        ab=np.array([np.random.uniform(-100, 100), np.random.uniform(-100, 100)]),
        iterations=10000,
        check_batch=50,
        eps=1e-5,
):
    return minibatch_gd(
        points / np.linalg.norm(points),
        error,
        lr,
        ab,
        iterations,
        check_batch,
        eps,
        1
    )


def minibatch_gd(
        points,
        error,
        lr=0.1,
        ab=np.array([np.random.uniform(-100, 100), np.random.uniform(-100, 100)]),
        iterations=10000,
        check_batch=50,
        eps=1e-5,
        batch_size=1
):
    n = points.shape[0]

    meta = {
        "points": np.array([], dtype=np.float64).reshape(0, 2)
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

            error_value_without_square = (ab[0] * points[pid, 0] + ab[1] - points[pid, 1])
            gradient_a = 2 * error_value_without_square * points[pid, 0]
            gradient_b = 2 * error_value_without_square

            norm = (gradient_a ** 2 + gradient_b ** 2) ** 0.5
            ab_grad += np.array([gradient_a, gradient_b]) / norm

        ab -= lr * ab_grad / batch_size

    return ab, meta


if __name__ == "__main__":
    f, points = generate_regression()

    visualize_regression_point(f, points)

    ab, meta = normalised_mini(points, None)

    print(ab, meta)

    visualize_line(ab, points)
