from abc import ABC, abstractmethod

import numpy as np

from HW1.gradient import gradient


class StepLengthComputer(ABC):
    """
    Class for computing segment due to linear search
    """

    @abstractmethod
    def compute_length(self, coord, direction):
        """
        Computes length of segment where linear search would be performed
        :param coord: current point of gradient search
        :param direction: direction where the optimal value should be searched
        :return: the second value of segment <'coord', 'coord' + t * 'direction'> where linear search would be run
        """
        pass


class NoStepLengthComputer(StepLengthComputer):
    """
    Implementing computing fixed-scaled length of segment
    """

    def __init__(self, alpha=100):
        """
        :param alpha: fixed-scaled coefficient
        """
        self.alpha = alpha

    def compute_length(self, coord, direction):
        meta = {
            "function_call_count": 0,
            "gradient_call_count": 0
        }
        return self.alpha, meta


class WolfeStepLengthComputer(StepLengthComputer):
    """
    Implementing computing segment that would satisfy the Wolfe's conditions
    """

    def __init__(self, c1, c2, f):
        """
        :param c1: first coefficient
        :param c2: second coefficient
        :param f: researching function
        """
        self.c1 = c1
        self.c2 = c2
        self.f = f

    def compute_length(self, coord, direction):
        alpha = 0
        t = 1
        beta = np.inf
        coord_gradient = gradient(self.f, coord)
        coord_f = self.f(coord)
        coord_dot = np.dot(coord_gradient.T, direction)
        meta = {
            "function_call_count": 1,
            "gradient_call_count": 1
        }
        while True:
            if self.f(coord + t * direction) > coord_f + self.c1 * t * coord_dot:
                meta["function_call_count"] += 1
                beta = t
                t = 0.5 * (beta + alpha)
            elif np.dot(gradient(self.f, coord + t * direction), direction) < self.c2 * coord_dot:
                meta["function_call_count"] += 1
                meta["gradient_call_count"] += 1
                alpha = t
                t = 2 * alpha if beta == np.inf else 0.5 * (beta + alpha)
            else:
                meta["function_call_count"] += 1
                meta["gradient_call_count"] += 1
                break

        return t, meta
