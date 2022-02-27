from abc import ABC, abstractmethod
import numpy as np

from HW1.step_length_computer import NoStepLengthComputer


class CoordRelaxer(ABC):
    @abstractmethod
    def relax(self, coord, lr, gradient):
        pass


class BasicCoordRelaxer(CoordRelaxer):
    def relax(self, coord, lr, gradient):
        return (coord - lr * gradient, None), {"function_call_count": 0}


class LinearCoordRelaxer(CoordRelaxer):
    def __init__(self, f, alpha, eps, step_length_computer=NoStepLengthComputer()):
        self.step_length_computer = step_length_computer
        self.f = f
        self.alpha = alpha
        self.eps = eps
        self.phi = (np.sqrt(5) + 1) / 2

    def find_golden_ratio_point(self, a, b):
        return b - (b - a) / self.phi, a + (b - a) / self.phi

    # noinspection PyUnresolvedReferences
    def golden_ratio_method(self, from_coord, to_coord):
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
