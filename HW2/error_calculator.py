from abc import ABC, abstractmethod

import numpy as np


class Error(ABC):
    @abstractmethod
    def gradient(self, ab, point):
        pass


class AbsErrorCalculator(Error):
    def gradient(self, ab, point):
        dif = ab[0] * point[0] + ab[1] - point[1]
        gr_a, gr_b = np.sign(dif) * point[0], np.sign(dif)
        norm = (gr_a ** 2 + gr_b ** 2) ** 0.5
        return np.array([gr_a / norm, gr_b / norm])


class SquaredErrorCalculator(Error):
    def gradient(self, ab, point):
        dif = ab[0] * point[0] + ab[1] - point[1]
        gr_a, gr_b = 2 * dif * point[0], 2 * dif
        norm = (gr_a ** 2 + gr_b ** 2) ** 0.5
        return np.array([gr_a / norm, gr_b / norm])


class BoxErrorCalculator(Error):

    def gradient(self, ab, point):
        dif = ab[0] * point[0] + ab[1] - point[1]
        gr_a, gr_b = 4 * (dif ** 3) * point[0], 4 * (dif ** 3)
        norm = (gr_a ** 2 + gr_b ** 2) ** 0.5
        return np.array([gr_a / norm, gr_b / norm])
