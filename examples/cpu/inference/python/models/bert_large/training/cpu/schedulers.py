# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LRScheduler(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        # Check if using mixed precision training
        self.mixed_training = False
        base_optimizer = optimizer

        # Check that optimizer param is valid
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))

        super(LRScheduler, self).__init__(base_optimizer, last_epoch)

    def step(self, epoch=None):
        # Set the current training step
        # ('epoch' is used to be consistent with _LRScheduler)
        if self.mixed_training:
            # The assumption is that the step will be constant
            state_dict = self.optimizer.state[
                self.optimizer.param_groups[0]["params"][0]
            ]
            if "step" in state_dict:
                self.last_epoch = state_dict["step"] + 1
            else:
                self.last_epoch = 1
        else:
            self.last_epoch = epoch if epoch is not None else self.last_epoch + 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class LinearWarmUpScheduler(LRScheduler):
    """
    Applies a warm up period to the learning rate.
    """

    def __init__(self, optimizer, warmup, total_steps, last_epoch=-1):
        self.warmup = warmup
        self.total_steps = total_steps
        super(LinearWarmUpScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        progress = self.last_epoch / self.total_steps
        if progress < self.warmup:
            return [base_lr * progress / self.warmup for base_lr in self.base_lrs]
        else:
            return [
                base_lr * max((progress - 1.0) / (self.warmup - 1.0), 0.0)
                for base_lr in self.base_lrs
            ]


class LinearWarmupPolyDecayScheduler(LRScheduler):
    """
    Applies a warm up period to the learning rate.
    """

    def __init__(
        self,
        optimizer,
        start_warmup_steps,
        warmup_steps,
        total_steps,
        end_learning_rate=0.0,
        degree=1.0,
        last_epoch=-1,
    ):
        self.num_warmup_updates = warmup_steps
        self.start_warmup_steps = start_warmup_steps
        self.total_steps = total_steps
        self.end_learning_rate = end_learning_rate
        self.degree = degree
        self.offset_step = int(self.start_warmup_steps == 0)
        super(LinearWarmupPolyDecayScheduler, self).__init__(optimizer, last_epoch)

    def step(self, epoch=None):
        # Instead of optimizer.param_groups['lr'],
        # update optimizer._lr to avoid sync
        state_dict = self.optimizer.state[self.optimizer.param_groups[0]["params"][0]]
        if "step" in state_dict:
            self.last_epoch = state_dict["step"] + 1
        else:
            self.last_epoch = 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        mod_step = self.last_epoch - self.offset_step - self.start_warmup_steps
        cond = mod_step < self.num_warmup_updates
        progress = (cond * (mod_step / (self.num_warmup_updates + 1e-6))) + (
            (1.0 - cond)
            * (min((self.last_epoch - self.offset_step) / self.total_steps, 1))
        )
        base_lr = self.base_lrs[0]
        lr = (cond * (base_lr * progress)) + (
            (1.0 - cond)
            * (
                (base_lr - self.end_learning_rate) * (1 - progress) ** self.degree
                + self.end_learning_rate
            )
        )
        return lr
