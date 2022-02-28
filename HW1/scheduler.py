import math
from abc import ABC, abstractmethod


class Scheduler(ABC):
    """
    Class for adapting learning rate
    """

    @abstractmethod
    def decay_lr(self, iteration, lr):
        """
        Computes new value for learning rate after 'iteration' iterations passed
        :param iteration: number of current iteration
        :param lr: current value of learning rate
        :return: new value for learning rate
        """
        pass


class EmptyScheduler(Scheduler):
    """
    Implements scheduler that don`t adapt learning rate
    """

    def decay_lr(self, iteration, lr):
        """
        Just returns current value without decaying it
        """
        return lr


class ExponentialScheduler(Scheduler):
    """
    Implements exponential decrease of the learning rate
    """

    def __init__(self, decay_period, decay_exp, lower_bound=0.01):
        """
        :param decay_period: frequency of learning rate decaying, in iterations
        :param decay_exp: value of relative decreasing
        :param lower_bound: minimum supposed value for learning rate
        """
        self.decay_period = decay_period
        self.decay_exp = decay_exp
        self.lower_bound = lower_bound

    def decay_lr(self, iteration, lr):
        if iteration % self.decay_period == 0 and lr * (math.e ** self.decay_exp) > self.lower_bound:
            return lr * (math.e ** self.decay_exp)
        else:
            return lr


class StepScheduler(Scheduler):
    """
    Implements constant decrease of the learning rate
    """

    def __init__(self, decay_period, decay_step, lower_bound=0.01):
        """
        :param decay_period: frequency of learning rate decaying, in iterations
        :param decay_step: value of absolute decreasing
        :param lower_bound: minimum supposed value for learning rate
        """
        self.lower_bound = lower_bound
        self.decay_period = decay_period
        self.decay_step = decay_step

    def decay_lr(self, iteration, lr):
        if iteration % self.decay_period == 0 and lr - self.decay_step > self.lower_bound:
            return lr - self.decay_step
        else:
            return lr
