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

import numpy as np
import torch
import torch.nn as nn

from loguru import logger

from .attention import build_attention_head


class SMPLXNet(nn.Module):

    def __init__(self, exp_cfg):
        super(SMPLXNet, self).__init__()

        self.exp_cfg = exp_cfg.clone()
        network_cfg = exp_cfg.get('network', {})
        self.net_type = network_cfg.get('type', 'attention')
        if self.net_type == 'attention':
            self.smplx = build_attention_head(exp_cfg)
        else:
            raise ValueError(f'Unknown network type: {self.net_type}')

    def toggle_hands_and_face(self, iteration):
        pass

    def toggle_losses(self, iteration):
        self.smplx.toggle_losses(iteration)

    def get_hand_model(self) -> nn.Module:
        return self.smplx.get_hand_model()

    def get_head_model(self) -> nn.Module:
        return self.smplx.get_head_model()

    def toggle_param_prediction(self, iteration) -> None:
        self.smplx.toggle_param_prediction(iteration)

    def forward(self, images, targets,
                hand_imgs=None, hand_targets=None,
                head_imgs=None, head_targets=None,
                full_imgs=None,
                device=None):

        if not self.training:
            pass
        if device is None:
            device = torch.device('cpu')

        losses = {}

        output = self.smplx(images, targets=targets,
                            hand_imgs=hand_imgs, hand_targets=hand_targets,
                            head_imgs=head_imgs, head_targets=head_targets,
                            full_imgs=full_imgs)

        output['losses'] = losses
        return output
