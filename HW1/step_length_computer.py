from abc import ABC, abstractmethod

import numpy as np

from HW1.gradient import gradient


class StepLengthComputer(ABC):
    @abstractmethod
    def computeLength(self, coord, direction): pass


class NoStepLengthComputer(StepLengthComputer):
    def computeLength(self, coord, direction):
        return 1


class WolfeStepLengthComputer(StepLengthComputer):
    def __init__(self, c1, c2, f):
        self.c1 = c1
        self.c2 = c2
        self.f = f

    def computeLength(self, coord, direction):
        alpha = 0
        t = 1
        beta = float("inf")
        coord_gradient = gradient(self.f, coord)
        coord_f = self.f(coord)
        coord_dot = np.dot(coord_gradient, direction)
        while True:
            if self.f(coord + t * direction) > coord_f + self.c1 * t * coord_dot:
                beta = t
                t = 0.5 * (beta + alpha)
            elif np.dot(gradient(self.f, coord + t * direction), direction) < self.c2 * coord_dot:
                alpha = t
                t = 2 * alpha if beta == float("inf") else 0.5 * (beta + alpha)
            else:
                break
        print("t={}".format(t))

        return t
