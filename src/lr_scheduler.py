from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.optim import Optimizer
import torch.optim.lr_scheduler as lr_scheduler

import math
import torch
import numpy as np


class CosineWithRestarts(lr_scheduler._LRScheduler):  # pylint: disable=protected-access
    """
    Cosine annealing with restarts.
    This is decribed in the paper https://arxiv.org/abs/1608.03983.
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    T_max : int
        The maximum number of iterations within the first cycle.
    eta_min : float, optional (default: 0)
        The minimum learning rate.
    last_epoch : int, optional (default: -1)
        The index of the last epoch.
    t_mult : float, optional (default: 1)
        The factor by which the cycle length (``T_max``) increasing after each
        restart.
    """

    def __init__(self,
                 optimizer,
                 last_epoch = -1,
                 T_max = 10,
                 T_mult = 2,
                 eta_min = 0.,
                 decay = 1.,
                 start_decay_cycle = 1,
                 **_):
        assert T_max > 0
        assert eta_min >= 0
        assert 1 >= decay > 0
        assert start_decay_cycle > 0
        if T_max == 1 and T_mult == 1:
            print("Cosine annealing scheduler will have no effect on the learning "
                           "rate since T_max = 1 and T_mult = 1.")
        self.t_max = T_max
        self.eta_min = eta_min
        self.t_mult = T_mult
        self.decay = decay
        self.start_decay_cycle = start_decay_cycle
        self._last_restart = 0
        self._cycle_counter = 0
        self._cycle_factor = 1.
        self._updated_cycle_len = T_max
        self._initialized = False
        self._cycle_counts = 0
        super(CosineWithRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get updated learning rate."""
        # HACK: We need to check if this is the first time ``self.get_lr()`` was called,
        # since ``torch.optim.lr_scheduler._LRScheduler`` will call ``self.get_lr()``
        # when first initialized, but the learning rate should remain unchanged
        # for the first epoch.
        if not self._initialized:
            self._initialized = True
            return self.base_lrs

        step = self.last_epoch + 1
        self._cycle_counter = step - self._last_restart
        if self._cycle_counter % self._updated_cycle_len == 0:
            self._cycle_counts += 1
        if self._cycle_counts >= self.start_decay_cycle:
            lr_decay = self.decay**(self._cycle_counts//self.start_decay_cycle)
            self.eta_min *= self.decay
        else:
            lr_decay = 1

        lrs = [
                self.eta_min + ((lr*lr_decay - self.eta_min) / 2) * (
                        np.cos(
                                np.pi *
                                (self._cycle_counter % self._updated_cycle_len) /
                                self._updated_cycle_len
                        ) + 1
                )
                for lr in self.base_lrs
        ]

        if self._cycle_counter % self._updated_cycle_len == 0:
            # Adjust the cycle length.
            self._cycle_factor *= self.t_mult
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.t_max)
            self._last_restart = step

        return lrs
