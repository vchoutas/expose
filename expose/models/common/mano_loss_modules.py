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
import os
import os.path as osp
import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import defaultdict

from loguru import logger

from .rigid_alignment import RotationTranslationAlignment
from expose.data.targets.keypoints import (
    get_part_idxs, KEYPOINT_NAMES, HAND_CONNECTIONS)

from expose.losses import build_loss, build_prior
from expose.utils.typing_utils import Tensor


PARAM_KEYS = ['hand_pose',
              'left_to_right_hand_pose',
              'right_hand_keyps',
              'left_to_right_hand_keyps',
              ]


class MANOLossModule(nn.Module):
    '''
    '''

    def __init__(self, loss_cfg):
        super(MANOLossModule, self).__init__()

        self.penalize_final_only = loss_cfg.get('penalize_final_only', True)

        self.loss_enabled = defaultdict(lambda: True)
        self.loss_activ_step = {}

        idxs_dict = get_part_idxs()
        hand_idxs = idxs_dict['hand']
        left_hand_idxs = idxs_dict['left_hand']
        right_hand_idxs = idxs_dict['right_hand']

        self.register_buffer('hand_idxs', torch.tensor(hand_idxs))

        self.register_buffer('left_hand_idxs', torch.tensor(left_hand_idxs))
        self.register_buffer('right_hand_idxs', torch.tensor(right_hand_idxs))

        shape_loss_cfg = loss_cfg.shape
        self.shape_weight = shape_loss_cfg.get('weight', 0.0)
        self.shape_loss = build_loss(**shape_loss_cfg)
        self.loss_activ_step['shape'] = shape_loss_cfg.enable

        vertices_loss_cfg = loss_cfg.vertices
        self.vertices_weight = vertices_loss_cfg.get('weight', 0.0)
        self.vertices_loss = build_loss(**vertices_loss_cfg)
        self.loss_activ_step['vertices'] = vertices_loss_cfg.enable

        self.use_alignment = vertices_loss_cfg.get('use_alignment', False)
        if self.use_alignment:
            self.alignment = RotationTranslationAlignment()

        edge_loss_cfg = loss_cfg.get('edge', {})
        self.edge_weight = edge_loss_cfg.get('weight', 0.0)
        self.edge_loss = build_loss(**edge_loss_cfg)
        self.loss_activ_step['edge'] = edge_loss_cfg.get('enable', 0)

        global_orient_cfg = loss_cfg.global_orient
        self.global_orient_loss = build_loss(**global_orient_cfg)
        logger.debug('Global pose loss: {}', self.global_orient_loss)
        self.global_orient_weight = global_orient_cfg.weight
        self.loss_activ_step['global_orient'] = global_orient_cfg.enable

        hand_pose_cfg = loss_cfg.get('hand_pose', {})
        hand_pose_loss_type = loss_cfg.hand_pose.type
        self.hand_use_conf = hand_pose_cfg.get('use_conf_weight', False)

        self.hand_pose_weight = loss_cfg.hand_pose.weight
        if self.hand_pose_weight > 0:
            self.hand_pose_loss_type = hand_pose_loss_type
            self.hand_pose_loss = build_loss(**loss_cfg.hand_pose)
            self.loss_activ_step['hand_pose'] = loss_cfg.hand_pose.enable

        joints2d_cfg = loss_cfg.joints_2d
        self.joints_2d_weight = joints2d_cfg.weight
        self.joints_2d_enable_at = joints2d_cfg.enable
        if self.joints_2d_weight > 0:
            self.joints_2d_loss = build_loss(**joints2d_cfg)
            logger.debug('2D hand joints loss: {}', self.joints_2d_loss)
            self.joints_2d_active = False

        hand_edge_2d_cfg = loss_cfg.get('hand_edge_2d', {})
        self.hand_edge_2d_weight = hand_edge_2d_cfg.get('weight', 0.0)
        self.hand_edge_2d_enable_at = hand_edge_2d_cfg.get('enable', 0)
        if self.hand_edge_2d_weight > 0:
            self.hand_edge_2d_loss = build_loss(
                type='edge', connections=HAND_CONNECTIONS, **hand_edge_2d_cfg)
            logger.debug('2D hand edge loss: {}', self.hand_edge_2d_loss)
            self.hand_edge_2d_active = False

        joints3d_cfg = loss_cfg.joints_3d
        self.joints_3d_weight = joints3d_cfg.weight
        self.joints_3d_enable_at = joints3d_cfg.enable
        if self.joints_3d_weight > 0:
            joints_3d_loss_type = joints3d_cfg.type
            self.joints_3d_loss = build_loss(**joints3d_cfg)
            logger.debug('3D hand joints loss: {}', self.joints_3d_loss)
            self.joints_3d_active = False

    def is_active(self) -> bool:
        return any(self.loss_enabled.values())

    def toggle_losses(self, step) -> None:
        for key in self.loss_activ_step:
            self.loss_enabled[key] = step >= self.loss_activ_step[key]

    def extra_repr(self) -> str:
        msg = []
        msg.append('Shape weight: {self.shape_weight}')
        msg.append(f'Global pose weight: {self.global_orient_weight}')
        if self.hand_pose_weight > 0:
            msg.append(f'Hand pose weight: {self.hand_pose_weight}')
        return '\n'.join(msg)

    def single_loss_step(self, parameters,
                         global_orient=None,
                         hand_pose=None,
                         gt_hand_pose_idxs=None,
                         shape=None,
                         gt_vertices=None,
                         gt_vertex_idxs=None,
                         device=None,
                         keyp_confs=None):
        losses = defaultdict(
            lambda: torch.tensor(0, device=device, dtype=torch.float32))

        param_vertices = parameters.get('vertices', None)
        compute_vertex_loss = (self.vertices_weight > 0 and
                               len(gt_vertex_idxs) > 0 and
                               param_vertices is not None and
                               gt_vertices is not None)
        if gt_vertex_idxs is not None:
            if len(gt_vertex_idxs) > 0:
                param_vertices = param_vertices[gt_vertex_idxs]

        if compute_vertex_loss:
            if self.use_alignment:
                aligned_verts = self.alignment(param_vertices, gt_vertices)
            else:
                aligned_verts = param_vertices
            losses['vertex_loss'] = self.vertices_weight * self.vertices_loss(
                aligned_verts, gt_vertices)

        compute_edge_loss = (self.edge_weight > 0 and
                             len(gt_vertex_idxs) > 0 and
                             param_vertices is not None and
                             gt_vertices is not None)
        if compute_edge_loss:
            edge_loss_val = self.edge_loss(
                gt_vertices=gt_vertices,
                est_vertices=param_vertices)
            losses['mesh_edge_loss'] = self.edge_weight * edge_loss_val

        if (self.shape_weight > 0 and self.loss_enabled['betas'] and
                shape is not None):
            losses['shape_loss'] = (
                self.shape_loss(parameters['betas'], shape) *
                self.shape_weight)

        if (self.global_orient_weight > 0 and self.loss_enabled['globals'] and
                global_orient is not None):
            losses['global_orient_loss'] = (
                self.global_orient_loss(
                    parameters['wrist_pose'], global_orient) *
                self.global_orient_weight)

        if (self.hand_pose_weight > 0 and
                self.loss_enabled['hand_pose'] and
                hand_pose is not None):
            #  num_joints = parameters['hand_pose'].shape[1]
            #  weights = (
                #  keyp_confs['hand'].mean(axis=1, keepdim=True).expand(
                    #  -1, num_joints).reshape(-1)
                #  if self.hand_use_conf and keyp_confs is not None else None)
            #  if weights is not None:
                #  num_ones = [1] * len(
                    #  parameters['hand_pose'].shape[2:])
                #  weights = weights.view(-1, num_joints, *num_ones)
            losses['hand_pose_loss'] = (
                self.hand_pose_loss(
                    parameters['right_hand_pose'], hand_pose) *
                self.hand_pose_weight)

        return losses

    def forward(self, input_dict,
                hand_targets,
                device=None):
        if device is None:
            device = torch.device('cpu')

        # Stack the GT keypoints and conf for the predictions of the right hand
        hand_keyps = torch.stack(
            [t.smplx_keypoints for t in hand_targets])
        hand_conf = torch.stack([t.conf for t in hand_targets])

        # Get the GT pose of the right hand
        gt_hand_pose = torch.stack(
            [t.get_field('hand_pose').right_hand_pose
             for t in hand_targets
             if t.has_field('hand_pose')
             ])
        gt_hand_pose_idxs = [ii for ii, t in enumerate(hand_targets)
                             if t.has_field('hand_pose')]
        # Get the GT pose of the right hand
        global_orient = torch.stack(
            [t.get_field('global_orient').global_orient for t in hand_targets
             if t.has_field('global_orient')])

        gt_vertex_idxs = [ii for ii, t in enumerate(hand_targets)
                          if t.has_field('vertices')]
        gt_vertices = None
        if len(gt_vertex_idxs) > 0:
            gt_vertices = torch.stack([
                t.get_field('vertices').vertices
                for t in hand_targets
                if t.has_field('vertices')])

        output_losses = {}
        compute_2d_loss = ('proj_joints' in input_dict and
                           self.joints_2d_weight > 0)
        if compute_2d_loss:
            hand_proj_joints = input_dict['proj_joints']
            hand_joints2d_loss = self.joints_2d_loss(
                hand_proj_joints,
                hand_keyps[:, self.right_hand_idxs],
                weights=hand_conf[:, self.right_hand_idxs])
            output_losses['joints2d'] = (
                hand_joints2d_loss * self.joints_2d_weight)

        # Stack the GT keypoints and conf for the predictions of the
        # right hand
        hand_keyps_3d = [t.get_field('keypoints3d').smplx_keypoints
                         for t in hand_targets if t.has_field('keypoints3d')]
        hand_conf_3d = [t.get_field('keypoints3d').conf
                        for t in hand_targets if t.has_field('keypoints3d')]

        num_stages = input_dict.get('num_stages', 1)
        curr_params = input_dict.get(f'stage_{num_stages - 1:02d}', None)
        joints3d = input_dict['joints']
        compute_3d_joint_loss = (self.joints_3d_weight > 0 and
                                 len(hand_conf_3d) > 0)

        if compute_3d_joint_loss:
            hand_keyps_3d = torch.stack(hand_keyps_3d)[:, self.right_hand_idxs]
            hand_conf_3d = torch.stack(hand_conf_3d)[:, self.right_hand_idxs]

            pred_joints = joints3d
            # Center the joints according to the wrist
            centered_pred_joints = pred_joints - pred_joints[:, [0]]
            gt_hand_keyps_3d = hand_keyps_3d - hand_keyps_3d[:, [0]]
            hand_keyp3d_loss = self.joints_3d_loss(
                centered_pred_joints,
                gt_hand_keyps_3d,
                weights=hand_conf_3d,
            ) * self.joints_3d_weight
            output_losses['joints3d'] = hand_keyp3d_loss

        for n in range(1, num_stages + 1):
            if self.penalize_final_only and n < num_stages:
                continue

            curr_params = input_dict.get(f'stage_{n - 1:02d}', None)
            if curr_params is None:
                logger.warning(f'Network output for stage {n} is None')
                continue

            curr_losses = self.single_loss_step(
                curr_params,
                hand_pose=gt_hand_pose,
                gt_hand_pose_idxs=gt_hand_pose_idxs,
                global_orient=global_orient,
                gt_vertices=gt_vertices,
                gt_vertex_idxs=gt_vertex_idxs,
                device=device)
            for key in curr_losses:
                out_key = f'stage_{n - 1:02d}_{key}'
                output_losses[out_key] = curr_losses[key]

        return output_losses


class RegularizerModule(nn.Module):
    def __init__(self, loss_cfg,
                 body_pose_mean=None, hand_pose_mean=None):
        super(RegularizerModule, self).__init__()

        self.regularize_final_only = loss_cfg.get(
            'regularize_final_only', True)

        # Construct the shape prior
        shape_prior_type = loss_cfg.shape.prior.type
        self.shape_prior_weight = loss_cfg.shape.prior.weight
        if self.shape_prior_weight > 0:
            self.shape_prior = build_prior(shape_prior_type,
                                           **loss_cfg.shape.prior)
            logger.debug(f'Shape prior {self.shape_prior}')

        hand_prior_cfg = loss_cfg.hand_pose.prior
        hand_pose_prior_type = hand_prior_cfg.type
        self.hand_pose_prior_weight = hand_prior_cfg.weight
        if self.hand_pose_prior_weight > 0:
            self.hand_pose_prior = build_prior(
                hand_pose_prior_type,
                mean=hand_pose_mean,
                **hand_prior_cfg)
            logger.debug(f'Hand pose prior {self.hand_pose_prior}')

        logger.debug(self)

    def extra_repr(self) -> str:
        msg = []
        if self.shape_prior_weight > 0:
            msg.append(f'Shape prior weight: {self.shape_prior_weight}')
        if self.hand_pose_prior_weight > 0:
            msg.append(
                f'Hand pose prior weight: {self.hand_pose_prior_weight}')
        return '\n'.join(msg)

    def single_regularization_step(self, parameters, **kwargs):
        prior_losses = {}

        betas = parameters.get('betas', None)
        if self.shape_prior_weight > 0 and betas is not None:
            prior_losses['shape_prior'] = (
                self.shape_prior_weight * self.shape_prior(betas))

        hand_pose = parameters.get('right_hand_pose', None)
        if (self.hand_pose_prior_weight > 0 and
                hand_pose is not None):
            prior_losses['hand_pose_prior'] = (
                self.hand_pose_prior(hand_pose) *
                self.hand_pose_prior_weight)

        return prior_losses

    def forward(self,
                input_dict, **kwargs) -> Dict[str, Tensor]:

        prior_losses = defaultdict(lambda: 0)
        num_stages = input_dict.get('num_stages', 1)
        for n in range(1, num_stages + 1):
            if self.regularize_final_only and n < num_stages:
                continue
            curr_params = input_dict.get(f'stage_{n - 1:02d}', None)
            if curr_params is None:
                logger.warning(f'Network output for stage {n} is None')
                continue

            curr_losses = self.single_regularization_step(curr_params)
            for key in curr_losses:
                out_key = f'stage_{n - 1:02d}_{key}'
                prior_losses[out_key] = curr_losses[key]
        return prior_losses
