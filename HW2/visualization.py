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


def draw_levels(error, generated_points, grad_points=None, scale=100, rate=100):
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
                           grad_points] + list(np.linspace(-1, 1, 100))))
    ax_levels.plot(XS, YS, 'r.')

    plt.show()
