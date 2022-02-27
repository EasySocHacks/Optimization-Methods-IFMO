import math
from abc import ABC, abstractmethod


class Scheduler(ABC):
    @abstractmethod
    def decay_lr(self, iteration, lr): pass


class EmptyScheduler(Scheduler):

    def decay_lr(self, iteration, lr):
        return lr


class ExponentialScheduler(Scheduler):

    def __init__(self, decay_period, decay_exp, lower_bound=0.01):
        self.decay_period = decay_period
        self.decay_exp = decay_exp
        self.lower_bound = lower_bound

    def decay_lr(self, iteration, lr):
        if iteration % self.decay_period == 0 and lr * (math.e ** self.decay_exp) > self.lower_bound:
            return lr * (math.e ** self.decay_exp)
        else:
            return lr


class StepScheduler(Scheduler):
    def __init__(self, decay_period, decay_step, lower_bound=0.01):
        self.lower_bound = lower_bound
        self.decay_period = decay_period
        self.decay_step = decay_step

    def decay_lr(self, iteration, lr):
        if iteration % self.decay_period == 0 and lr - self.decay_step > self.lower_bound:
            return lr - self.decay_step
        else:
            return lr
