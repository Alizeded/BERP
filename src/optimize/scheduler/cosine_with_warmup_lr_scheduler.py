# adopted from https://github.com/sooftware/pytorch-lr-scheduler/blob/main/lr_scheduler/transformer_lr_scheduler.py

import math
from typing import Optional

import torch
from torch.optim import Optimizer

from src.optimize.scheduler.base_lr_scheduler import LearningRateScheduler


class CosineAnnealingWithWarmUpLRScheduler(LearningRateScheduler):
    r"""
    Cosine Annealing with Warmup Learning Rate Scheduler.

    Args:
        optimizer (Optimizer): Optimizer.
        init_lr (float): Initial learning rate.
        peak_lr (float): Maximum learning rate.
        max_steps (int): Maximum steps in training.
        final_lr_scale (float): Final learning rate scale
        phase_ratio_warmup (float): Ratio of warmup stage
        phase_ratio_decay (float): Ratio of decay stage
    """

    def __init__(
        self,
        optimizer: Optimizer,
        init_lr: float,
        peak_lr: float,
        max_steps: int,
        final_lr_scale: float,
        phase_ratio_warmup: float = 0.2,
        phase_ratio_decay: float = 0.8,
    ) -> None:

        super().__init__(optimizer, init_lr)
        self.final_lr = peak_lr * final_lr_scale
        self.peak_lr = peak_lr
        self.warmup_steps = int(max_steps * phase_ratio_warmup)
        self.decay_steps = int(max_steps * phase_ratio_decay)

        self.warmup_rate = self.peak_lr / self.warmup_steps
        self.decay_factor = -math.log(final_lr_scale) / self.decay_steps

        self.init_lr = init_lr
        self.update_steps = 0

    def _decide_stage(self):
        if self.update_steps < self.warmup_steps:
            return 0, self.update_steps

        if (
            self.warmup_steps
            <= self.update_steps
            < self.warmup_steps + self.decay_steps
        ):
            return 1, self.update_steps - self.warmup_steps

        return 2, None

    def step(self, val_loss: Optional[torch.FloatTensor] = None):
        self.update_steps += 1
        stage, steps_in_stage = self._decide_stage()

        if stage == 0:
            self.lr = self.update_steps * self.warmup_rate
        elif stage == 1:
            self.lr = (self.init_lr * self.peak_lr) + 0.5 * (
                self.peak_lr - self.final_lr
            ) * (1 + math.cos(math.pi * steps_in_stage / self.decay_steps))
        else:
            raise ValueError("Undefined stage")

        self.set_lr(self.optimizer, self.lr)

        return self.lr
