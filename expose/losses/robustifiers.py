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

import sys
import time

import os.path as osp

import numpy as np

import torch
import torch.nn as nn
from loguru import logger


def build_robustifier(robustifier_type: str = None, **kwargs) -> nn.Module:
    if robustifier_type is None or robustifier_type == 'none':
        return None
    elif robustifier_type == 'gmof':
        return GMOF(**kwargs)
    else:
        raise ValueError(f'Unknown robustifier: {robustifier_type}')


class GMOF(nn.Module):
    def __init__(self, rho: float = 100, **kwargs) -> None:
        super(GMOF, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return f'Rho = {self.rho}'

    def forward(self, residual):
        squared_residual = residual.pow(2)
        return torch.div(squared_residual, squared_residual + self.rho ** 2)
