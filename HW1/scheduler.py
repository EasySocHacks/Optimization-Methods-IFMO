import math
from abc import ABC, abstractmethod


class Scheduler(ABC):
    @abstractmethod
    def decay_lr(self, iteration, lr): pass


class EmptyScheduler(Scheduler):

    def decay_lr(self, iteration, lr):
        return lr


class ExponentialScheduler(Scheduler):

    def __init__(self, decay_period, decay_exp):
        self.decay_period = decay_period
        self.decay_exp = decay_exp

    def decay_lr(self, iteration, lr):
        if iteration % self.decay_period == 0:
            return lr * (math.e ** self.decay_exp)
        else:
            return lr


class StepScheduler(Scheduler):
    def __init__(self, decay_period, decay_step):
        self.decay_period = decay_period
        self.decay_step = decay_step

    def decay_lr(self, iteration, lr):
        if iteration % self.decay_period == 0:
            return lr - self.decay_step
        else:
            return lr
