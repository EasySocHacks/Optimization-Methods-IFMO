import numpy as np
from matplotlib import pyplot as plt


def draw_2D(f, points=None, scale=100, rate=100):
    """
    Drawing chart of a 2D function 'f'.

    Furthermore, drawing gradient descent work result in case points is not None

    :param f: a 2D function to draw
    :param points: optional. gradient descent work result
    :param scale: chart axis scale to draw
    :param rate: count of x-axis points to build f chart.
    :return: Nothing
    """
    xs = np.linspace(-scale, scale, rate)
    ys = np.array(list(map(lambda x: f([x]), xs)))

    if points is not None:
        plt.plot(points[:, 0], points[:, 1], 'r.')
    plt.plot(xs, ys)

    plt.show()


def draw_3D(f, points=None, scale=100, rate=100):
    """
    Drawing chart of a 3D function 'f'.

    Furthermore, drawing gradient descent work result in case points is not None

    :param f: a 3D function to draw
    :param points: optional. gradient descent work result
    :param scale: chart axis scale to draw
    :param rate: count of x-axis and y-axis points to build f chart.
    :return: Nothing
    """
    t = np.linspace(-scale, scale, rate)
    X, Y = np.meshgrid(t, t)
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot_surface(X, Y, f([X, Y]))

    if points is not None:
        XS = np.array([])
        YS = np.array([])
        ZS = points[:, 1]

        for point in points[:, 0]:
            XS = np.append(XS, point[0])
            YS = np.append(YS, point[1])

        ax_points = plt.figure().add_subplot(projection='3d')
        ax_points.scatter(XS, YS, ZS, c='r')

        ax_levels = plt.figure().add_subplot()
        ax_levels.contour(X, Y, f([X, Y]), levels=sorted(
            [f([p[0], p[1]]) for p in points[:, 0]] + list(np.linspace(-1, 1, 100))))
        ax_levels.plot(XS, YS, 'r.')

    plt.show()
