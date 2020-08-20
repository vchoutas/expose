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
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import defaultdict

from loguru import logger

from .rigid_alignment import RotationTranslationAlignment
from expose.data.targets.keypoints import (
    get_part_idxs, KEYPOINT_NAMES, FACE_CONNECTIONS)
from expose.losses import build_loss, build_prior
from expose.utils.typing_utils import Tensor


PARAM_KEYS = ['betas', 'expression', 'global_orient', 'jaw_pose']


class FLAMELossModule(nn.Module):
    '''
    '''

    def __init__(self, loss_cfg, use_face_contour=False):
        super(FLAMELossModule, self).__init__()

        self.penalize_final_only = loss_cfg.get('penalize_final_only', True)
        self.loss_enabled = defaultdict(lambda: True)
        self.loss_activ_step = {}

        idxs_dict = get_part_idxs()
        head_idxs = idxs_dict['flame']
        if not use_face_contour:
            head_idxs = head_idxs[:-17]

        self.register_buffer('head_idxs', torch.tensor(head_idxs))

        # TODO: Add vertex loss
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

        shape_loss_cfg = loss_cfg.shape
        self.shape_weight = shape_loss_cfg.weight
        self.shape_loss = build_loss(**shape_loss_cfg)
        self.loss_activ_step['shape'] = shape_loss_cfg.enable

        expression_cfg = loss_cfg.get('expression', {})

        self.expression_weight = expression_cfg.weight
        if self.expression_weight > 0:
            self.expression_loss = build_loss(**expression_cfg)
            self.loss_activ_step[
                'expression'] = expression_cfg.enable

        global_orient_cfg = loss_cfg.global_orient
        self.global_orient_loss = build_loss(**global_orient_cfg)
        logger.debug(f'Global pose loss: {self.global_orient_loss}')
        self.global_orient_weight = global_orient_cfg.weight
        self.loss_activ_step['global_orient'] = global_orient_cfg.enable

        jaw_pose_cfg = loss_cfg.get('jaw_pose', {})
        jaw_pose_loss_type = jaw_pose_cfg.type
        self.jaw_pose_weight = jaw_pose_cfg.weight

        if self.jaw_pose_weight > 0:
            self.jaw_pose_loss_type = jaw_pose_loss_type
            self.jaw_pose_loss = build_loss(**jaw_pose_cfg)
            logger.debug('Jaw pose loss: {}', self.jaw_pose_loss)
            self.loss_activ_step['jaw_pose'] = jaw_pose_cfg.enable

        face_edge_2d_cfg = loss_cfg.get('face_edge_2d', {})
        self.face_edge_2d_weight = face_edge_2d_cfg.get('weight', 0.0)
        self.face_edge_2d_enable_at = face_edge_2d_cfg.get('enable', 0)
        if self.face_edge_2d_weight > 0:
            face_connections = []
            for conn in FACE_CONNECTIONS:
                if ('contour' in KEYPOINT_NAMES[conn[0]] or
                        'contour' in KEYPOINT_NAMES[conn[1]]):
                    if not use_face_contour:
                        continue
                face_connections.append(conn)

            self.face_edge_2d_loss = build_loss(
                type='edge', connections=face_connections, **face_edge_2d_cfg)
            logger.debug('2D face edge loss: {}', self.face_edge_2d_loss)
            self.face_edge_2d_active = False

        face_joints2d_cfg = loss_cfg.joints_2d
        self.face_joints_2d_weight = face_joints2d_cfg.weight
        self.face_joints_2d_enable_at = face_joints2d_cfg.enable
        if self.face_joints_2d_weight > 0:
            self.face_joints_2d_loss = build_loss(**face_joints2d_cfg)
            logger.debug('2D face joints loss: {}', self.face_joints_2d_loss)
            self.face_joints_2d_active = False

        face_joints3d_cfg = loss_cfg.joints_3d
        self.face_joints_3d_weight = face_joints3d_cfg.weight
        self.face_joints_3d_enable_at = face_joints3d_cfg.enable
        if self.face_joints_3d_weight > 0:
            self.face_joints_3d_loss = build_loss(**face_joints3d_cfg)
            logger.debug('3D face joints loss: {}', self.face_joints_3d_loss)
            self.face_joints_3d_active = False

    def is_active(self) -> bool:
        return any(self.loss_enabled.values())

    def toggle_losses(self, step) -> None:
        for key in self.loss_activ_step:
            self.loss_enabled[key] = step >= self.loss_activ_step[key]

    def extra_repr(self) -> str:
        msg = []
        msg.append('Shape weight: {}'.format(self.shape_weight))
        if self.expression_weight > 0:
            msg.append(f'Expression weight: {self.expression_weight}')
        msg.append(f'Global pose weight: {self.global_orient_weight}')
        if self.jaw_pose_weight > 0:
            msg.append(f'Jaw pose weight: {self.jaw_pose_weight}')
        return '\n'.join(msg)

    def single_loss_step(self, parameters,
                         global_orient=None,
                         jaw_pose=None,
                         betas=None,
                         expression=None,
                         gt_vertices=None,
                         device=None,
                         keyp_confs=None,
                         gt_expression_idxs=None,
                         ):
        losses = defaultdict(
            lambda: torch.tensor(0, device=device, dtype=torch.float32))

        if (self.shape_weight > 0 and self.loss_enabled['betas'] and
                betas is not None):
            shape_common_dim = min(parameters['betas'].shape[-1],
                                   betas.shape[-1])
            losses['shape_loss'] = (
                self.shape_loss(parameters['betas'][:, :shape_common_dim],
                                betas[:, :shape_common_dim]) *
                self.shape_weight)

        param_vertices = parameters.get('vertices', None)
        compute_vertex_loss = (self.vertices_weight > 0 and
                               param_vertices is not None and
                               gt_vertices is not None)
        if compute_vertex_loss:
            if self.use_alignment:
                aligned_verts = self.alignment(param_vertices, gt_vertices)
            else:
                aligned_verts = param_vertices
            losses['vertex_loss'] = self.vertices_weight * self.vertices_loss(
                aligned_verts, gt_vertices)

        compute_edge_loss = (self.edge_weight > 0 and
                             param_vertices is not None and
                             gt_vertices is not None)
        if compute_edge_loss:
            edge_loss_val = self.edge_loss(
                gt_vertices=gt_vertices, est_vertices=param_vertices)
            losses['mesh_edge_loss'] = self.edge_weight * edge_loss_val

        if (self.expression_weight > 0 and self.loss_enabled['expression'] and
                expression is not None):
            expr_common_dim = min(
                parameters['expression'].shape[-1], expression.shape[-1])
            pred_expr = parameters['expression'][:, :expr_common_dim]
            if gt_expression_idxs is not None:
                pred_expr = pred_expr[gt_expression_idxs]

            losses['expression_loss'] = (
                self.expression_loss(
                    pred_expr, expression[:, :expr_common_dim]) *
                self.expression_weight)

        if (self.global_orient_weight > 0 and
                self.loss_enabled['global_orient'] and
                global_orient is not None):
            losses['global_orient_loss'] = (
                self.global_orient_loss(
                    parameters['head_pose'], global_orient) *
                self.global_orient_weight)

        if (self.jaw_pose_weight > 0 and self.loss_enabled['jaw_pose'] and
                jaw_pose is not None):
            losses['jaw_pose_loss'] = (
                self.jaw_pose_loss(
                    parameters['jaw_pose'], jaw_pose) *
                self.jaw_pose_weight)

        return losses

    def forward(self, input_dict,
                head_targets,
                device=None):
        if device is None:
            device = torch.device('cpu')

        # Stack the GT keypoints and conf for the predictions of the right hand
        face_keyps = torch.stack([t.smplx_keypoints for t in head_targets])
        face_conf = torch.stack([t.conf for t in head_targets])

        # Get the GT pose of the right hand
        global_orient = torch.stack(
            [t.get_field('global_orient').global_orient for t in head_targets])
        # Get the GT pose of the right hand
        gt_jaw_pose = torch.stack(
            [t.get_field('jaw_pose').jaw_pose
             for t in head_targets])

        has_vertices = all(
            [t.has_field('vertices') for t in head_targets])
        gt_vertices = None
        if has_vertices:
            gt_vertices = torch.stack([
                t.get_field('vertices').vertices
                for t in head_targets])
        # Get the GT pose of the right hand
        gt_expression = torch.stack([t.get_field('expression').expression
                                     for t in head_targets
                                     if t.has_field('expression')])
        gt_expression_idxs = torch.tensor(
            [idx for idx, t in enumerate(head_targets)
             if t.has_field('expression')], device=device, dtype=torch.long)

        output_losses = {}
        compute_2d_loss = ('proj_joints' in input_dict and
                           self.face_joints_2d_weight > 0)
        if compute_2d_loss:
            face_proj_joints = input_dict['proj_joints']
            face_joints2d = self.face_joints_2d_loss(
                face_proj_joints,
                face_keyps[:, self.head_idxs],
                weights=face_conf[:, self.head_idxs])
            output_losses['head_branch_joints2d'] = (
                face_joints2d * self.face_joints_2d_weight)

        head_keyps = [t.get_field('keypoints3d').smplx_keypoints
                      for t in head_targets
                      if t.has_field('keypoints3d')]
        head_conf = [t.get_field('keypoints3d').conf for t in head_targets
                     if t.has_field('keypoints3d')]
        # Keep the indices of the targets that have 3D joint annotations
        head_idxs = [idx for idx, t in enumerate(head_targets)
                     if t.has_field('keypoints3d')]

        num_stages = input_dict.get('num_stages', 1)
        curr_params = input_dict.get(f'stage_{num_stages - 1:02d}', None)
        joints3d = curr_params['joints']
        compute_3d_joint_loss = (self.face_joints_3d_weight > 0 and
                                 len(head_conf) > 0)
        if compute_3d_joint_loss:
            all_keyps3d = torch.stack(head_keyps, dim=0)[:, self.head_idxs]
            all_conf3d = torch.stack(head_conf, dim=0)[:, self.head_idxs]

            head_keyp3d_loss = self.face_joints_3d_loss(
                joints3d[head_idxs],
                all_keyps3d,
                weights=all_conf3d
            ) * self.face_joints_3d_weight
            output_losses['head_branch_joints3d'] = head_keyp3d_loss

        for n in range(1, num_stages + 1):
            if self.penalize_final_only and n < num_stages:
                continue
            curr_params = input_dict.get(f'stage_{n - 1:02d}', None)
            if curr_params is None:
                logger.warning(f'Network output for stage {n} is None')
                continue

            curr_losses = self.single_loss_step(
                curr_params,
                jaw_pose=gt_jaw_pose,
                global_orient=global_orient,
                expression=gt_expression,
                gt_vertices=gt_vertices,
                device=device,
                gt_expression_idxs=gt_expression_idxs,
            )
            for key in curr_losses:
                out_key = f'stage_{n - 1:02d}_{key}'
                output_losses[out_key] = curr_losses[key]

        return output_losses


class RegularizerModule(nn.Module):
    def __init__(self, loss_cfg,
                 num_stages=3, jaw_pose_mean=None):
        super(RegularizerModule, self).__init__()

        self.regularize_final_only = loss_cfg.get(
            'regularize_final_only', True)
        self.num_stages = num_stages

        # Construct the shape prior
        shape_prior_type = loss_cfg.shape.prior.type
        self.shape_prior_weight = loss_cfg.shape.prior.weight
        if self.shape_prior_weight > 0:
            self.shape_prior = build_prior(shape_prior_type,
                                           **loss_cfg.shape.prior)
            logger.debug(f'Shape prior {self.shape_prior}')

        # Construct the expression prior
        expression_prior_cfg = loss_cfg.expression.prior
        expression_prior_type = expression_prior_cfg.type
        self.expression_prior_weight = expression_prior_cfg.weight
        if self.expression_prior_weight > 0:
            self.expression_prior = build_prior(
                expression_prior_type,
                **expression_prior_cfg)
            logger.debug(f'Expression prior {self.expression_prior}')

        # Construct the jaw pose prior
        jaw_pose_prior_cfg = loss_cfg.jaw_pose.prior
        jaw_pose_prior_type = jaw_pose_prior_cfg.type
        self.jaw_pose_prior_weight = jaw_pose_prior_cfg.weight
        if self.jaw_pose_prior_weight > 0:
            self.jaw_pose_prior = build_prior(
                jaw_pose_prior_type, mean=jaw_pose_mean, **jaw_pose_prior_cfg)
            logger.debug(f'Jaw pose prior {self.jaw_pose_prior}')

        logger.debug(self)

    def extra_repr(self) -> str:
        msg = []
        if self.shape_prior_weight > 0:
            msg.append(f'Shape prior weight: {self.shape_prior_weight}')
        if self.expression_prior_weight > 0:
            msg.append(
                f'Expression prior weight: {self.expression_prior_weight}')
        if self.jaw_pose_prior_weight > 0:
            msg.append(f'Jaw pose prior weight: {self.jaw_pose_prior_weight}')
        return '\n'.join(msg)

    def single_regularization_step(self, parameters, **kwargs):
        prior_losses = {}

        betas = parameters.get('betas', None)
        if self.shape_prior_weight > 0 and betas is not None:
            prior_losses['shape_prior'] = (
                self.shape_prior_weight * self.shape_prior(betas))

        expression = parameters.get('expression', None)
        if self.expression_prior_weight > 0 and expression is not None:
            prior_losses['expression_prior'] = (
                self.expression_prior(expression) *
                self.expression_prior_weight)

        jaw_pose = parameters.get('jaw_pose', None)
        if self.jaw_pose_prior_weight > 0 and jaw_pose is not None:
            prior_losses['jaw_pose_prior'] = (
                self.jaw_pose_prior(jaw_pose) *
                self.jaw_pose_prior_weight)

        return prior_losses

    def forward(self,
                input_dict,
                **kwargs) -> Dict[str, Tensor]:

        prior_losses = defaultdict(lambda: 0)
        num_stages = input_dict.get('num_stages', 1)
        for n in range(1, num_stages + 1):
            if self.regularize_final_only and n < self.num_stages:
                continue
            curr_params = input_dict.get(f'stage_{n - 1:02d}', None)
            if curr_params is None:
                logger.warning(f'Network output for stage {n} is None')
                continue

            curr_losses = self.single_regularization_step(curr_params)
            for key, val in curr_losses.items():
                out_key = f'stage_{n - 1:02d}_{key}'
                prior_losses[out_key] = val

        return prior_losses
