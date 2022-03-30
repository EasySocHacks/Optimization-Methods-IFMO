from abc import ABC, abstractmethod

import numpy as np


class Optimization(ABC):
    @abstractmethod
    def gradient(self, w, point, error):
        pass

    @abstractmethod
    def relax(self, lr, grad):
        pass


class DefaultOptimization(Optimization):
    def gradient(self, w, point, error):
        return error.gradient(w, point)

    def relax(self, lr, grad):
        return -lr * grad

    def __str__(self):
        return 'Default(no) optimization'

    def __repr__(self):
        return self.__str__()


class MomentumOptimization(Optimization):
    def __init__(self, beta=0, dim=2):
        self.beta = beta

        self.grad_plain_sum = np.zeros(dim)

    def gradient(self, w, point, error):
        return error.gradient(w, point)

    def relax(self, lr, grad):
        self.grad_plain_sum = self.beta * self.grad_plain_sum + (1.0 - self.beta) * grad
        return -lr * self.grad_plain_sum

    def __str__(self):
        return 'Momentum optimization (beta={})'.format(self.beta)

    def __repr__(self):
        return self.__str__()


class NesterovOptimization(Optimization):
    def __init__(self, beta=0, dim=2):
        self.beta = beta

        self.grad_plain_sum = np.zeros(dim)

    def gradient(self, w, point, error):
        return error.gradient(
            w,
            point - self.beta * self.grad_plain_sum
        )

    def relax(self, lr, grad):
        self.grad_plain_sum = self.beta * self.grad_plain_sum + (1.0 - self.beta) * grad

        return -lr * self.grad_plain_sum

    def __str__(self):
        return 'Nesterov optimization (beta={})'.format(self.beta)

    def __repr__(self):
        return self.__str__()


class AdaGradOptimization(Optimization):
    def __init__(self, eps=1e-5, dim=2):
        self.eps = eps

        self.grad_square_sum = np.zeros(dim)

    def gradient(self, w, point, error):
        return error.gradient(w, point)

    def relax(self, lr, grad):
        self.grad_square_sum += grad ** 2

        return -lr / (np.sqrt(self.grad_square_sum) + self.eps) * grad

    def __str__(self):
        return 'Ada gradient optimization (eps={})'.format(self.eps)

    def __repr__(self):
        return self.__str__()


class RMSPropOptimization(Optimization):
    def __init__(self, gamma=0.0, eps=1e-5, dim=2):
        self.gamma = gamma
        self.eps = eps

        self.grad_square_plain_sum = np.zeros(dim)

    def gradient(self, w, point, error):
        return error.gradient(w, point)

    def relax(self, lr, grad):
        self.grad_square_plain_sum = self.gamma * self.grad_square_plain_sum + \
                                     (1.0 - self.gamma) * grad ** 2

        return -lr / (np.sqrt(self.grad_square_plain_sum) + self.eps) * grad

    def __str__(self):
        return 'RMS prop optimization (gamma={}, eps={})'.format(self.gamma, self.eps)

    def __repr__(self):
        return self.__str__()


class AdamOptimization(Optimization):
    def __init__(self, beta_1=0.9, beta_2=0.999, eps=1e-5, dim=2):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

        self.grad_plain_sum = np.zeros(dim)
        self.grad_square_plain_sum = np.zeros(dim)

        self.iteration = 0

    def gradient(self, w, point, error):
        return error.gradient(w, point)

    def relax(self, lr, grad):
        self.iteration += 1

        self.grad_plain_sum = self.beta_1 * self.grad_plain_sum + (1.0 - self.beta_1) * grad
        self.grad_square_plain_sum = self.beta_2 * self.grad_square_plain_sum + \
                                     (1 - self.beta_2) * grad ** 2

        return -lr / (np.sqrt(self.grad_square_plain_sum) + self.eps) * self.grad_plain_sum

    def __str__(self):
        return 'Adam optimization (beta_1={}, beta_2={}, eps={})'.format(self.beta_1, self.beta_2, self.eps)

    def __repr__(self):
        return self.__str__()
