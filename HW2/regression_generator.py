import numpy as np


def generate_regression(f_scale=1, point_count=100, scale=np.array([10, 5])):
    dim = scale.shape[0]
    f_x = np.array(np.random.uniform(-f_scale, f_scale, dim))

    def f(x):
        return np.sum(f_x[:-1] * x) + f_x[-1]

    points = np.array([])

    for _ in range(point_count):
        x = np.array([])
        for x_scale in scale[:-1]:
            x = np.append(x, np.random.uniform(-x_scale, x_scale))

        y = np.random.normal(f(x), scale[-1])
        points = np.append(points, np.append(x, y))

    points = points.reshape((point_count, dim))

    return f, points, f_x
