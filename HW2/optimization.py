from abc import ABC, abstractmethod

import numpy as np


class Optimization(ABC):
    @abstractmethod
    def gradient(self, ab, point, error):
        pass

    @abstractmethod
    def relax(self, lr, grad):
        pass


class DefaultOptimization(Optimization):
    def gradient(self, ab, point, error):
        return error.gradient(ab, point)

    def relax(self, lr, grad):
        return -lr * grad


class MomentumOptimization(Optimization):
    def __init__(self, beta=0):
        self.beta = beta

        self.grad_plain_sum = np.zeros(2)

    def gradient(self, ab, point, error):
        return error.gradient(ab, point)

    def relax(self, lr, grad):
        self.grad_plain_sum = self.beta * self.grad_plain_sum + (1.0 - self.beta) * grad
        return -lr * self.grad_plain_sum


class NesterovOptimization(Optimization):
    def __init__(self, beta=0):
        self.beta = beta

        self.grad_plain_sum = np.zeros(2)

    def gradient(self, ab, point, error):
        return error.gradient(
            ab,
            np.array([point[0] - self.beta * self.grad_plain_sum[0], point[1] - self.beta * self.grad_plain_sum[1]])
        )

    def relax(self, lr, grad):
        self.grad_plain_sum = self.beta * self.grad_plain_sum + (1.0 - self.beta) * grad

        return -lr * self.grad_plain_sum


class AdaGradOptimization(Optimization):
    def __init__(self, eps=1e-5):
        self.eps = eps

        self.grad_square_sum = np.zeros(2)

    def gradient(self, ab, point, error):
        return error.gradient(ab, point)

    def relax(self, lr, grad):
        self.grad_square_sum += grad ** 2

        return -lr / (np.sqrt(self.grad_square_sum) + self.eps) * grad


class RMSPropOptimization(Optimization):
    def __init__(self, gamma=0.0, eps=1e-5):
        self.gamma = gamma
        self.eps = eps

        self.grad_square_plain_sum = np.zeros(2)

    def gradient(self, ab, point, error):
        return error.gradient(ab, point)

    def relax(self, lr, grad):
        self.grad_square_plain_sum = self.gamma * self.grad_square_plain_sum + \
                                     (1.0 - self.gamma) * grad ** 2

        return -lr / (np.sqrt(self.grad_square_plain_sum) + self.eps) * grad


class AdamOptimization(Optimization):
    def __init__(self, beta_1=0.9, beta_2=0.999, eps=1e-10):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

        self.grad_plain_sum = np.zeros(2)
        self.grad_square_plain_sum = np.zeros(2)

        self.iteration = 0

    def gradient(self, ab, point, error):
        return error.gradient(ab, point)

    def relax(self, lr, grad):
        self.iteration += 1

        self.grad_plain_sum = self.beta_1 * self.grad_plain_sum + (1.0 - self.beta_1) * grad
        self.grad_square_plain_sum = self.beta_2 * self.grad_square_plain_sum + \
                                     (1 - self.beta_2) * grad ** 2

        grad_plain_sum_norm = self.grad_plain_sum / (1.0 - self.beta_1 ** (self.iteration - 1))
        grad_square_plain_sum_norm = self.grad_square_plain_sum / (1.0 - self.beta_2 ** (self.iteration - 1))

        return -lr / (np.sqrt(grad_square_plain_sum_norm) + self.eps) * grad_plain_sum_norm
