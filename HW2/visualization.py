import numpy as np
from matplotlib import pyplot as plt

from HW2.error_calculator import SquaredErrorCalculator


def visualize_regression_point(f, points, scale=10, rate=100):
    xs = np.linspace(-scale, scale, rate)
    ys = f(xs)

    plt.plot(points[:, 0], points[:, 1], 'r.')
    plt.plot(xs, ys)

    plt.show()


def visualize_line(ab, points, rate=100):
    def f(x):
        return ab[0] * x + ab[1]

    max_x = np.max(np.abs(points[:, 0]))
    max_y = np.max(np.abs(points[:, 1]))

    #scale = max(max_x, max_y)
    scale = max_x
    xs = np.linspace(-scale, scale, rate)
    ys = f(xs)

    if points is not None:
        plt.plot(points[:, 0], points[:, 1], 'r.')
    plt.plot(xs, ys)

    plt.show()


def draw_levels(generated_points, grad_points=None, rate=100, error=SquaredErrorCalculator()):
    scale_x = np.max(np.abs(grad_points[:, 0]))
    scale_y = np.max(np.abs(grad_points[:, 1]))

    scale = max(scale_x, scale_y)

    t = np.linspace(-scale, scale, rate)
    X, Y = np.meshgrid(t, t)
    Z = np.array([])

    for i in range(rate):
        for j in range(rate):
            Z = np.append(Z, error.general_error(np.array([X[i, j], Y[i, j]]), generated_points))

    Z = Z.reshape((rate, rate))

    XS = np.array([])
    YS = np.array([])

    for point in grad_points:
        XS = np.append(XS, point[0])
        YS = np.append(YS, point[1])

    ax_levels = plt.figure().add_subplot()
    ax_levels.contour(X, Y, Z,
                      levels=sorted(
                          [error.general_error(np.array([p[0], p[1]]), generated_points) for p in
                           grad_points] + list(np.linspace(-10, 10, 100))))

    ax_levels.plot(XS, YS, "r")
    ax_levels.plot([grad_points[-1, 0]], [grad_points[-1, 1]], 'b.', markersize=20)

    plt.show()
