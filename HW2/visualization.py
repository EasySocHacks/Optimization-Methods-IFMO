import numpy as np
from matplotlib import pyplot as plt

from HW2.error_calculator import SquaredErrorCalculator


def visualize_regression_point(f, points, scale=100, rate=100):
    xs = np.linspace(-scale, scale, rate)
    ys = np.array([])

    for x in xs:
        ys = np.append(ys, f(np.array([x])))

    plt.plot(points[:, 0], points[:, 1], 'r.')
    plt.plot(xs, ys)

    plt.show()


def visualize_line(ab, points, rate=100):
    def f(x):
        return ab[0] * x + ab[1]

    max_x = np.max(np.abs(points[:, 0]))
    max_y = np.max(np.abs(points[:, -1]))

    # scale = max(max_x, max_y)
    scale = max_x
    xs = np.linspace(-scale, scale, rate)
    ys = f(xs)

    if points is not None:
        plt.plot(points[:, 0], points[:, -1], 'r.')
    plt.plot(xs, ys)

    plt.show()


# noinspection DuplicatedCode
def draw_levels(generated_points, grad_points, rate=100, error=SquaredErrorCalculator()):
    scale_x = np.max(np.abs(grad_points[:, 0]))
    scale_y = np.max(np.abs(grad_points[:, -1]))

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


# noinspection DuplicatedCode
def draw_multiple_levels(
        start_point,
        generated_points,
        grad_points_list,
        label_list,
        rate=100,
        error=SquaredErrorCalculator(),
        stride=10
):
    scale = 0

    ax_levels = plt.figure(figsize=(25, 25)).add_subplot()

    grad_points_concat = np.array([])
    for grad_points in grad_points_list:
        scale_x = np.max(np.abs(grad_points[:, 0]))
        scale_y = np.max(np.abs(grad_points[:, -1]))

        scale_tmp = max(scale_x, scale_y)
        scale = max(scale_tmp, scale)

        grad_points_concat = np.append(grad_points_concat, grad_points)

    t = np.linspace(-scale, scale, rate)
    X, Y = np.meshgrid(t, t)
    Z = np.array([])

    for i in range(rate):
        for j in range(rate):
            Z = np.append(Z, error.general_error(np.array([X[i, j], Y[i, j]]), generated_points))

    Z = Z.reshape((rate, rate))

    ax_levels.contour(X, Y, Z,
                      levels=np.array(sorted(
                          [error.general_error(np.array([p[0], p[1]]), generated_points) for p in
                           np.unique(grad_points_concat.reshape((-1, 2)), axis=0)] + list(np.linspace(-10, 10, 100))))[
                             ::stride])

    for grad_points, label in zip(grad_points_list, label_list):
        rgb = (np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))

        XS = np.array([])
        YS = np.array([])

        for point in grad_points:
            XS = np.append(XS, point[0])
            YS = np.append(YS, point[1])

        ax_levels.plot(XS, YS, c=rgb, label=label, linewidth=2)
        ax_levels.plot([grad_points[-1, 0]], [grad_points[-1, 1]], '.', c=rgb, markersize=30)

    ax_levels.plot([start_point[0]], [start_point[1]], 'r.', markersize=30)

    ax_levels.legend()
    plt.show()
