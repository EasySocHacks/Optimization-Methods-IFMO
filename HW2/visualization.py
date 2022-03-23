import numpy as np
from matplotlib import pyplot as plt


def visualize_regression_point(f, points, scale=10, rate=100):
    xs = np.linspace(-scale, scale, rate)
    ys = f(xs)

    plt.plot(points[:, 0], points[:, 1], 'r.')
    plt.plot(xs, ys)

    plt.show()


def visualize_line(ab, points=None, scale=10, rate=100):
    def f(x):
        return ab[0] * x + ab[1]

    xs = np.linspace(-scale, scale, rate)
    ys = f(xs)

    if points is not None:
        plt.plot(points[:, 0], points[:, 1], 'r.')
    plt.plot(xs, ys)

    plt.show()
