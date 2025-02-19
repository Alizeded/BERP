import math
from typing import Optional

import torch
from torch.optim import Optimizer

from src.optimize.scheduler.base_lr_scheduler import LearningRateScheduler


class TriStageLRScheduler(LearningRateScheduler):
    r"""
    Tri-Stage Learning Rate Scheduler. Implement the learning rate scheduler in "SpecAugment"

    Args:
        optimizer (Optimizer): Optimizer.
        peak_lr (float): Maximum learning rate.
        init_lr_scale (float): Initial learning rate scale.
        final_lr_scale (float): Final learning rate scale.
        warmup_steps (int): Warmup the learning rate linearly for the first N updates.
        hold_steps (int): Hold the learning rate for the N updates.
        decay_steps (int): Decay the learning rate linearly for the first N updates.
        total_steps (int): Total steps in training.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        init_lr: float,
        peak_lr: float,
        max_steps: int,
        final_lr_scale: float = 0.01,
        phase_ratio_warmup: float = 0.1,  # fairseq default: 0.1
        phase_ratio_hold: float = 0.4,  # fairseq default: 0.4
        phase_ratio_decay: float = 0.5,  # fairseq default: 0.5
    ):

        super().__init__(optimizer, init_lr)
        self.init_lr = init_lr
        self.final_lr = peak_lr * final_lr_scale
        self.peak_lr = peak_lr
        self.max_steps = max_steps
        self.warmup_steps = int(max_steps * phase_ratio_warmup)
        self.hold_steps = int(max_steps * phase_ratio_hold)
        self.decay_steps = int(max_steps * phase_ratio_decay)

        self.warmup_rate = (
            (self.peak_lr - self.init_lr) / self.warmup_steps
            if self.warmup_steps != 0
            else 0
        )
        self.decay_factor = -math.log(final_lr_scale) / self.decay_steps

        self.lr = self.init_lr
        self.update_steps = 0

    def _decide_stage(self):
        if self.update_steps < self.warmup_steps:
            return 0, self.update_steps

        offset = self.warmup_steps

        if self.update_steps < offset + self.hold_steps:
            return 1, self.update_steps - offset

        offset += self.hold_steps

        if self.update_steps <= offset + self.decay_steps:
            # decay stage
            return 2, self.update_steps - offset

        offset += self.decay_steps

        return 3, self.update_steps - offset

    def step(self, val_loss: Optional[torch.FloatTensor] = None):
        stage, steps_in_stage = self._decide_stage()

        if stage == 0:
            self.lr = self.init_lr + self.warmup_rate * steps_in_stage
        elif stage == 1:
            self.lr = self.peak_lr
        elif stage == 2:
            self.lr = self.peak_lr * math.exp(-self.decay_factor * steps_in_stage)
        elif stage == 3:
            self.lr = self.final_lr
        else:
            raise ValueError("Undefined stage")

        self.set_lr(self.optimizer, self.lr)
        self.update_steps += 1

        return self.lr
