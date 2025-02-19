from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LearningRateScheduler(_LRScheduler):
    r"""
    Provides inteface of learning rate scheduler.

    Note:
        Do not use this class directly, use one of the sub classes.
    """

    def __init__(self, optimizer: Optimizer, lr):
        self.optimizer = optimizer
        self.lr = lr

    def step(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def set_lr(optimizer, lr):
        for g in optimizer.param_groups:
            g["lr"] = lr

    def get_lr(self):
        for g in self.optimizer.param_groups:
            return g["lr"]
