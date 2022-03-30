from abc import ABC, abstractmethod

import numpy as np


def scalar(_ab, _point):
    result = 0.0
    for i in range(len(_point)):
        result += _ab[i] * _point[i]
    result += _ab[-1]
    return result


class Error(ABC):
    @abstractmethod
    def gradient(self, w, point):
        pass

    @abstractmethod
    def general_error(self, w, points):
        pass


class AbsErrorCalculator(Error):
    def general_error(self, w, points):
        return np.mean(list(map(lambda p: np.abs(scalar(w, p) - p[-1]), points)))

    def gradient(self, w, point):
        dif = scalar(w, point) - point[-1]
        gr = np.append(point[:-1] * np.sign(dif), np.sign(dif))
        return gr / np.linalg.norm(gr)

    def __str__(self):
        return 'Absolute error calculator'

    def __repr__(self):
        return self.__str__()


class SquaredErrorCalculator(Error):
    def general_error(self, w, points):
        return np.mean(list(map(lambda p: np.square(scalar(w, p) - p[-1]), points)))

    def gradient(self, w, point):
        dif = scalar(w, point) - point[-1]
        gr = np.append(point[:-1] * 2 * dif, 2 * dif)
        return gr / np.linalg.norm(gr)

    def __str__(self):
        return 'Squared error calculator'

    def __repr__(self):
        return self.__str__()


class BoxErrorCalculator(Error):
    def general_error(self, w, points):
        return np.mean(list(map(lambda p: (scalar(w, p) - p[-1]) ** 4, points)))

    def gradient(self, w, point):
        dif = scalar(w, point) - point[-1]
        gr = np.append(point[:-1] * 4 * dif ** 3, 4 * dif ** 3)
        return gr / np.linalg.norm(gr)

    def __str__(self):
        return 'Boxed error calculator'

    def __repr__(self):
        return self.__str__()
