from abc import ABC, abstractmethod

import numpy as np


class CoordRelaxer(ABC):
    @abstractmethod
    def relax(self, coord, lr, gradient):
        pass


class BasicCoordRelaxer(CoordRelaxer):
    def relax(self, coord, lr, gradient):
        return coord - lr * gradient


class LinearCoordRelaxer(CoordRelaxer):
    def __init__(self, f, alpha, eps):
        self.f = f
        self.alpha = alpha
        self.eps = eps
        self.phi = (np.sqrt(5) + 1) / 2

    def find_golden_ratio_point(self, a, b):
        return b - (b - a) / self.phi, a + (b - a) / self.phi

    # noinspection PyUnresolvedReferences
    def golden_ratio_method(self, a, b):
        function_call_count = np.array([0])

        x1, x2 = self.find_golden_ratio_point(a, b)

        y1 = self.f(x1)
        y2 = self.f(x2)

        while np.abs(b - a) > self.eps:
            if y1 < y2:
                b = x2

                x2 = x1
                x1 = self.find_golden_ratio_point(a, b)[0]

                y2 = y1
                y1 = self.f(x1)
            else:
                a = x1

                x1 = x2
                x2 = self.find_golden_ratio_point(a, b)[1]

                y1 = y2
                y2 = self.f(x2)

            function_call_count += 1

        if y1 < y2:
            return x1, y1, function_call_count
        else:
            return x2, y2, function_call_count

    def relax(self, coord, lr, gradient):
        vectorize = np.vectorize(self.golden_ratio_method)
        new_coord, _, golden_ratio_function_call_count = vectorize(coord, coord - self.alpha * lr * gradient)

        return new_coord
