# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de
from typing import Dict

from loguru import logger

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler


def build_optimizer(
    model: nn.Module,
    optim_cfg: Dict,
    exclude: str = '',
) -> optim.Optimizer:
    params = []

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = optim_cfg.lr
        weight_decay = optim_cfg.weight_decay
        if "bias" in key:
            lr = optim_cfg.lr * optim_cfg.bias_lr_factor
            weight_decay = optim_cfg.weight_decay_bias

        if len(exclude) > 0 and exclude in key:
            continue
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    lr = optim_cfg.lr

    optimizer = get_optimizer(params, optim_cfg)
    return optimizer


def get_optimizer(params, optim_cfg):
    lr = optim_cfg.lr
    optimizer_type = optim_cfg.type
    logger.debug('Building optimizer: {}', optimizer_type.upper())
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(params, lr,
                              **optim_cfg.sgd)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(params, lr, **optim_cfg.adam)
    elif optimizer_type == 'rmsprop':
        optimizer = optim.RMSprop(params, lr, **optim_cfg.rmsprop)
    elif optimizer_type == 'lbfgs':
        optimizer = optim.LBFGS(params, **optim_cfg.get('lbfgs', {}))
    else:
        raise ValueError(f'Unknown optimizer type: {optimizer_type}')
    return optimizer


def build_scheduler(
        optimizer: optim.Optimizer,
        sched_cfg: Dict
) -> optim.lr_scheduler._LRScheduler:
    scheduler_type = sched_cfg.type
    if scheduler_type == 'none':
        return None
    elif scheduler_type == 'step-lr':
        step_size = sched_cfg.step_size
        gamma = sched_cfg.gamma
        logger.info('Building scheduler: StepLR(step_size={}, gamma={})',
                    step_size, gamma)
        return scheduler.StepLR(optimizer, step_size, gamma)
    elif scheduler_type == 'multi-step-lr':
        gamma = sched_cfg.gamma
        milestones = sched_cfg.milestones
        logger.info('Building scheduler: MultiStepLR(milestone={}, gamma={})',
                    milestones, gamma)
        return scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma)
    else:
        raise ValueError(f'Unknown scheduler type: {scheduler_type}')
