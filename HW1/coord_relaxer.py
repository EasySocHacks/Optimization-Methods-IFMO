from abc import ABC, abstractmethod

import numpy as np

from HW1.step_length_computer import NoStepLengthComputer


class CoordRelaxer(ABC):
    """
    Class for updating coordinates using in gradient descent after each iteration.
    """
    @abstractmethod
    def relax(self, coord, lr, gradient):
        """
        Using coordinate 'coord', learning rate 'lr' and computed gradient 'gradient' of whatever function 'f'
        in point 'coord' belongs to iteration 'n' of a gradient descent, compute a new coordinates belongs to
        'n+1' iteration of the gradient descent.

        'coord' stands for a numpy array with a 'k' dimension.
        'gradient' stands for a numpy array with a 'k' dimension.

        :param coord: coordinates belongs to n'th gradient's descent iteration
        :param lr: learning rate belongs to n'th gradient's descent iteration
        :param gradient: computed gradient of a function in point 'coord'
        :return: new coordinates belongs to (n+1)'th gradient's descent iteration
        """
        pass


class BasicCoordRelaxer(CoordRelaxer):
    """
    A default coordinates relaxer.
    """
    def relax(self, coord, lr, gradient):
        return (coord - lr * gradient, None), {"function_call_count": 0}


class LinearCoordRelaxer(CoordRelaxer):
    """
    A coordinate relaxer, that computing new coordinates using a linear method.
    """
    def __init__(self, f, eps, step_length_computer=NoStepLengthComputer()):
        """
        :param f: a function, that using in gradient descent
        :param eps: a linear method accuracy threshold
        :param step_length_computer: a method to compute 'alpha' step in linear method
        """
        self.step_length_computer = step_length_computer
        self.f = f
        self.eps = eps
        self.phi = (np.sqrt(5) + 1) / 2

    def find_golden_ratio_point(self, a, b):
        """
        Finding point on the segment [a, b] for linear method called golder ratio method.

        :param a: left segment side
        :param b: right segment side
        :return: p1, p2 \in [a, b]
        """
        return b - (b - a) / self.phi, a + (b - a) / self.phi

    # noinspection PyUnresolvedReferences
    def golden_ratio_method(self, from_coord, to_coord):
        """
        Implementing golden ratio method.

        :param from_coord: left segment border
        :param to_coord:
        :return:
        """
        ans = np.array([])

        meta = {
            "function_call_count": 0
        }

        for idx, segment in enumerate(np.column_stack((from_coord, to_coord))):
            a = np.copy(from_coord)
            b = np.copy(to_coord)

            grp_x1, grp_x2 = self.find_golden_ratio_point(segment[0], segment[1])

            x1 = np.copy(a)
            x1[idx] = grp_x1
            x2 = np.copy(b)
            x2[idx] = grp_x2

            y1 = self.f(x1)
            y2 = self.f(x2)
            meta["function_call_count"] += 2

            while np.abs(a[idx] - b[idx]) > self.eps:
                if y1 < y2:
                    b[idx] = x2[idx]

                    x2[idx] = x1[idx]
                    x1[idx] = self.find_golden_ratio_point(a[idx], b[idx])[0]

                    y2 = y1
                    y1 = self.f(x1)
                else:
                    a[idx] = x1[idx]

                    x1[idx] = x2[idx]
                    x2[idx] = self.find_golden_ratio_point(a[idx], b[idx])[1]

                    y1 = y2
                    y2 = self.f(x2)

                meta["function_call_count"] += 1

            if y1 < y2:
                ans = np.append(ans, x1[idx])
            else:
                ans = np.append(ans, x2[idx])

        return (ans, self.f(ans)), meta

    def relax(self, coord, lr, gradient):
        alpha, meta_wolfe = self.step_length_computer.compute_length(coord, -gradient)
        result, meta_relax = self.golden_ratio_method(coord, coord - lr * gradient * alpha)
        meta_wolfe["function_call_count"] += meta_relax["function_call_count"]

        return result, meta_wolfe
