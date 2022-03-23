import numpy as np


def generate_regression(scale=1, point_count=100, x_scale=10, y_scale=5):
    f_x = np.array([
        np.random.uniform(-scale, scale),
        np.random.uniform(-scale, scale)
    ])

    def f(x):
        return f_x[0] * x + f_x[1]

    points = np.array([])

    for _ in range(point_count):
        x = np.random.uniform(-x_scale, x_scale)
        y = np.random.normal(f(x), y_scale)
        points = np.append(points, [x, y])

    points = points.reshape((point_count, 2))

    return f, points