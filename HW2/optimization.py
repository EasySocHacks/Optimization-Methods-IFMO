from abc import ABC, abstractmethod

import numpy as np


class Optimization(ABC):
    @abstractmethod
    def gradient(self, ab, point, error):
        pass

    @abstractmethod
    def relax(self, lr, grad, batch_size):
        pass


class DefaultOptimization(Optimization):
    def gradient(self, ab, point, error):
        return error.gradient(ab, point)

    def relax(self, lr, grad, batch_size):
        return -lr * grad / batch_size


class MomentumOptimization(Optimization):
    def __init__(self, beta=0):
        self.beta = beta

        self.last_grad = np.zeros(2)

    def gradient(self, ab, point, error):
        return self.beta * self.last_grad + (1.0 - self.beta) * error.gradient(ab, point)

    def relax(self, lr, grad, batch_size):
        return -lr * grad / batch_size


class NesterovOptimization(Optimization):
    def __init__(self, beta=0):
        self.beta = beta

        self.last_grad = np.zeros(2)

    def gradient(self, ab, point, error):
        return self.beta * self.last_grad + \
               (1.0 - self.beta) * \
               error.gradient(ab, np.array(
                   [point[0] - self.beta * self.last_grad[0], point[1] - self.beta * self.last_grad[1]]))

    def relax(self, lr, grad, batch_size):
        return -lr * grad / batch_size


class AdaGradOptimization(Optimization):
    def __init__(self, eps=1e-10):
        self.eps = eps

        self.s = np.zeros(2)

    def gradient(self, ab, point, error):
        grad = error.gradient(ab, point)
        self.s += grad ** 2

        return grad

    def relax(self, lr, grad, batch_size):
        return -lr / (np.sqrt(self.s) + self.eps) * grad / batch_size


class RMSPropOptimization(Optimization):
    def __init__(self, gamma, eps=1e-10):
        self.gamma = gamma
        self.eps = eps

        self.s = 0.0

    def gradient(self, ab, point, error):
        grad = error.gradient(ab, point)
        self.s += self.gamma * self.s + (1.0 - self.gamma) * grad ** 2

        return grad

    def relax(self, lr, grad, batch_size):
        return -lr / (np.sqrt(self.s) + self.eps) * grad / batch_size


class AdamOptimization(Optimization):
    def __init__(self, beta_1, beta_2, eps=1e-10):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

    def gradient(self, ab, point, error):
        pass

    def relax(self, lr, grad, batch_size):
        pass