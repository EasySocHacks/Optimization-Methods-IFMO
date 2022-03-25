from abc import ABC, abstractmethod

import numpy as np


class Error(ABC):
    @abstractmethod
    def gradient(self, ab, point):
        pass

    @abstractmethod
    def general_error(self, ab, points):
        pass


class AbsErrorCalculator(Error):
    def general_error(self, ab, points):
        return np.mean(list(map(lambda p: np.abs(ab[0] * p[0] + ab[1] - p[1]), points)))

    def gradient(self, ab, point):
        dif = ab[0] * point[0] + ab[1] - point[1]
        gr_a, gr_b = np.sign(dif) * point[0], np.sign(dif)
        norm = (gr_a ** 2 + gr_b ** 2) ** 0.5
        return np.array([gr_a / norm, gr_b / norm])

    def __str__(self):
        return 'Absolute error calculator'

    def __repr__(self):
        return self.__str__()


class SquaredErrorCalculator(Error):
    def general_error(self, ab, points):
        return np.mean(list(map(lambda p: np.square(ab[0] * p[0] + ab[1] - p[1]), points)))

    def gradient(self, ab, point):
        dif = ab[0] * point[0] + ab[1] - point[1]
        gr_a, gr_b = 2 * dif * point[0], 2 * dif
        norm = (gr_a ** 2 + gr_b ** 2) ** 0.5
        return np.array([gr_a / norm, gr_b / norm])

    def __str__(self):
        return 'Squared error calculator'

    def __repr__(self):
        return self.__str__()


class BoxErrorCalculator(Error):
    def general_error(self, ab, points):
        return np.mean(list(map(lambda p: (ab[0] * p[0] + ab[1] - p[1]) ** 4, points)))

    def gradient(self, ab, point):
        dif = ab[0] * point[0] + ab[1] - point[1]
        gr_a, gr_b = 4 * (dif ** 3) * point[0], 4 * (dif ** 3)
        norm = (gr_a ** 2 + gr_b ** 2) ** 0.5
        return np.array([gr_a / norm, gr_b / norm])

    def __str__(self):
        return 'Boxed error calculator'

    def __repr__(self):
        return self.__str__()
