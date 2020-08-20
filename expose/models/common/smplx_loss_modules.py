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
from typing import Dict, NewType

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import defaultdict

from loguru import logger

from .rigid_alignment import RotationTranslationAlignment
from ...data.targets.keypoints import get_part_idxs
from ...losses import build_loss, build_prior

Tensor = NewType('Tensor', torch.Tensor)

PARAM_KEYS = ['betas', 'expression', 'global_orient', 'body_pose', 'hand_pose',
              'jaw_pose']


class SMPLXLossModule(nn.Module):
    '''
    '''

    def __init__(self, loss_cfg, num_stages=3,
                 use_face_contour=False):
        super(SMPLXLossModule, self).__init__()

        self.stages_to_penalize = loss_cfg.get('stages_to_penalize', [-1])
        logger.info(f'Stages to penalize: {self.stages_to_penalize}')

        self.loss_enabled = defaultdict(lambda: True)
        self.loss_activ_step = {}

        idxs_dict = get_part_idxs()
        body_idxs = idxs_dict['body']
        hand_idxs = idxs_dict['hand']
        face_idxs = idxs_dict['face']
        left_hand_idxs = idxs_dict['left_hand']
        right_hand_idxs = idxs_dict['right_hand']
        if not use_face_contour:
            face_idxs = face_idxs[:-17]

        self.register_buffer('body_idxs', torch.tensor(body_idxs))
        self.register_buffer('hand_idxs', torch.tensor(hand_idxs))
        self.register_buffer('face_idxs', torch.tensor(face_idxs))

        self.register_buffer('left_hand_idxs', torch.tensor(left_hand_idxs))
        self.register_buffer('right_hand_idxs', torch.tensor(right_hand_idxs))

        shape_loss_cfg = loss_cfg.shape
        self.shape_weight = shape_loss_cfg.get('weight', 0.0)
        self.shape_loss = build_loss(**shape_loss_cfg)
        self.loss_activ_step['shape'] = shape_loss_cfg.enable

        expression_cfg = loss_cfg.get('expression', {})
        self.expr_use_conf_weight = expression_cfg.get(
            'use_conf_weight', False)

        self.expression_weight = expression_cfg.weight
        if self.expression_weight > 0:
            self.expression_loss = build_loss(**expression_cfg)
            self.loss_activ_step['expression'] = expression_cfg.enable

        global_orient_cfg = loss_cfg.global_orient
        global_orient_loss_type = global_orient_cfg.type
        self.global_orient_loss_type = global_orient_loss_type
        self.global_orient_loss = build_loss(**global_orient_cfg)
        logger.debug('Global pose loss: {}', self.global_orient_loss)
        self.global_orient_weight = global_orient_cfg.weight
        self.loss_activ_step['global_orient'] = global_orient_cfg.enable

        self.body_pose_weight = loss_cfg.body_pose.weight
        body_pose_loss_type = loss_cfg.body_pose.type
        self.body_pose_loss_type = body_pose_loss_type
        self.body_pose_loss = build_loss(**loss_cfg.body_pose)
        logger.debug('Body pose loss: {}', self.global_orient_loss)
        self.body_pose_weight = loss_cfg.body_pose.weight
        self.loss_activ_step['body_pose'] = loss_cfg.body_pose.enable

        left_hand_pose_cfg = loss_cfg.get('left_hand_pose', {})
        left_hand_pose_loss_type = loss_cfg.left_hand_pose.type
        self.lhand_use_conf = left_hand_pose_cfg.get('use_conf_weight', False)

        self.left_hand_pose_weight = loss_cfg.left_hand_pose.weight
        if self.left_hand_pose_weight > 0:
            self.left_hand_pose_loss_type = left_hand_pose_loss_type
            self.left_hand_pose_loss = build_loss(**loss_cfg.left_hand_pose)
            self.loss_activ_step[
                'left_hand_pose'] = loss_cfg.left_hand_pose.enable

        right_hand_pose_cfg = loss_cfg.get('right_hand_pose', {})
        right_hand_pose_loss_type = loss_cfg.right_hand_pose.type
        self.right_hand_pose_weight = loss_cfg.right_hand_pose.weight
        self.rhand_use_conf = right_hand_pose_cfg.get('use_conf_weight', False)
        if self.right_hand_pose_weight > 0:
            self.right_hand_pose_loss_type = right_hand_pose_loss_type
            self.right_hand_pose_loss = build_loss(**loss_cfg.right_hand_pose)
            self.loss_activ_step[
                'right_hand_pose'] = loss_cfg.right_hand_pose.enable

        jaw_pose_loss_type = loss_cfg.jaw_pose.type
        self.jaw_pose_weight = loss_cfg.jaw_pose.weight

        jaw_pose_cfg = loss_cfg.get('jaw_pose', {})
        self.jaw_use_conf_weight = jaw_pose_cfg.get('use_conf_weight', False)
        if self.jaw_pose_weight > 0:
            self.jaw_pose_loss_type = jaw_pose_loss_type
            self.jaw_pose_loss = build_loss(**loss_cfg.jaw_pose)
            logger.debug('Jaw pose loss: {}', self.global_orient_loss)
            self.loss_activ_step['jaw_pose'] = loss_cfg.jaw_pose.enable

        edge_loss_cfg = loss_cfg.get('edge', {})
        self.edge_weight = edge_loss_cfg.get('weight', 0.0)
        self.edge_loss = build_loss(**edge_loss_cfg)
        self.loss_activ_step['edge'] = edge_loss_cfg.get('enable', 0)

    def is_active(self) -> bool:
        return any(self.loss_enabled.values())

    def toggle_losses(self, step) -> None:
        for key in self.loss_activ_step:
            self.loss_enabled[key] = step >= self.loss_activ_step[key]

    def extra_repr(self) -> str:
        msg = []
        if self.shape_weight > 0:
            msg.append(f'Shape weight: {self.shape_weight}')
        if self.expression_weight > 0:
            msg.append(f'Expression weight: {self.expression_weight}')
        if self.global_orient_weight > 0:
            msg.append(f'Global pose weight: {self.global_orient_weight}')
        if self.body_pose_weight > 0:
            msg.append(f'Body pose weight: {self.body_pose_weight}')
        if self.left_hand_pose_weight > 0:
            msg.append(f'Left hand pose weight: {self.left_hand_pose_weight}')
        if self.right_hand_pose_weight > 0:
            msg.append(f'Right hand pose weight {self.right_hand_pose_weight}')
        if self.jaw_pose_weight > 0:
            msg.append(f'Jaw pose prior weight: {self.jaw_pose_weight}')
        return '\n'.join(msg)

    def single_loss_step(self, parameters, target_params,
                         target_param_idxs,
                         gt_vertices=None,
                         device=None,
                         keyp_confs=None,
                         penalize_only_parts=False,
                         ):
        losses = defaultdict(
            lambda: torch.tensor(0, device=device, dtype=torch.float32))

        param_vertices = parameters.get('vertices', None)
        compute_edge_loss = (self.edge_weight > 0 and
                             param_vertices is not None and
                             gt_vertices is not None and
                             not penalize_only_parts)
        if compute_edge_loss:
            edge_loss_val = self.edge_loss(
                gt_vertices=gt_vertices, est_vertices=param_vertices)
            losses['mesh_edge_loss'] = self.edge_weight * edge_loss_val

        compute_shape_loss = (
            self.shape_weight > 0 and self.loss_enabled['betas'] and
            'betas' in target_params and not penalize_only_parts
        )
        if compute_shape_loss:
            losses['shape_loss'] = (
                self.shape_loss(
                    parameters['betas'][target_param_idxs['betas']],
                    target_params['betas']) *
                self.shape_weight)

        compute_expr_loss = (self.expression_weight > 0 and
                             self.loss_enabled['expression'] and
                             'expression' in target_param_idxs)
        if compute_expr_loss:
            expr_idxs = target_param_idxs['expression']
            weights = (
                keyp_confs['face'].mean(axis=1)
                if self.expr_use_conf_weight else None)
            if weights is not None:
                num_ones = [1] * len(parameters['expression'].shape[1:])
                weights = weights.view(-1, *num_ones)
                weights = weights[expr_idxs]

            losses['expression_loss'] = (
                self.expression_loss(
                    parameters['expression'][expr_idxs],
                    target_params['expression'],
                    weights=weights) *
                self.expression_weight)

        compute_global_orient_loss = (
            self.global_orient_weight > 0 and self.loss_enabled['betas'] and
            'global_orient' in target_params and not penalize_only_parts
        )
        if compute_global_orient_loss:
            global_orient_idxs = target_param_idxs['global_orient']
            losses['global_orient_loss'] = (
                self.global_orient_loss(
                    parameters['global_orient'][global_orient_idxs],
                    target_params['global_orient']) *
                self.global_orient_weight)

        compute_body_pose_loss = (
            self.body_pose_weight > 0 and self.loss_enabled['betas'] and
            'body_pose' in target_params and not penalize_only_parts)

        if compute_body_pose_loss:
            body_pose_idxs = target_param_idxs['body_pose']
            losses['body_pose_loss'] = (
                self.body_pose_loss(
                    parameters['body_pose'][body_pose_idxs],
                    target_params['body_pose']) *
                self.body_pose_weight)

        if (self.left_hand_pose_weight > 0 and
                self.loss_enabled['left_hand_pose'] and
                'left_hand_pose' in target_param_idxs):
            num_left_hand_joints = parameters['left_hand_pose'].shape[1]
            weights = (
                keyp_confs['left_hand'].mean(axis=1, keepdim=True).expand(
                    -1, num_left_hand_joints).reshape(-1)
                if self.lhand_use_conf else None)
            if weights is not None:
                num_ones = [1] * len(
                    parameters['left_hand_pose'].shape[2:])
                weights = weights.view(-1, num_left_hand_joints, *num_ones)
                weights = weights[target_param_idxs['left_hand_pose']]
            losses['left_hand_pose_loss'] = (
                self.left_hand_pose_loss(
                    parameters['left_hand_pose'][
                        target_param_idxs['left_hand_pose']],
                    target_params['left_hand_pose'],
                    weights=weights) *
                self.left_hand_pose_weight)

        if (self.right_hand_pose_weight > 0 and
                self.loss_enabled['right_hand_pose'] and
                'right_hand_pose' in target_param_idxs):
            num_right_hand_joints = parameters['right_hand_pose'].shape[1]
            weights = (
                keyp_confs['right_hand'].mean(axis=1, keepdim=True).expand(
                    -1, num_right_hand_joints).reshape(-1)
                if self.rhand_use_conf else None)
            if weights is not None:
                num_ones = [1] * len(
                    parameters['right_hand_pose'].shape[2:])
                weights = weights.view(-1, num_left_hand_joints, *num_ones)
                weights = weights[target_param_idxs['right_hand_pose']]
            losses['right_hand_pose_loss'] = (
                self.right_hand_pose_loss(
                    parameters['right_hand_pose'][
                        target_param_idxs['right_hand_pose']],
                    target_params['right_hand_pose'],
                    weights=weights) *
                self.right_hand_pose_weight)

        if (self.jaw_pose_weight > 0 and self.loss_enabled['jaw_pose'] and
                'jaw_pose' in target_param_idxs):
            weights = (
                keyp_confs['face'].mean(axis=1)
                if self.jaw_use_conf_weight else None)
            if weights is not None:
                num_ones = [1] * len(parameters['jaw_pose'].shape[2:])
                weights = weights.view(-1, 1, *num_ones)
                weights = weights[target_param_idxs['jaw_pose']]

            losses['jaw_pose_loss'] = (
                self.jaw_pose_loss(
                    parameters['jaw_pose'][target_param_idxs['jaw_pose']],
                    target_params['jaw_pose'],
                    weights=weights) *
                self.jaw_pose_weight)

        return losses

    def forward(self, network_params, targets, num_stages=3, device=None):
        if device is None:
            device = torch.device('cpu')

        start_idxs = defaultdict(lambda: 0)
        in_target_param_idxs = defaultdict(lambda: [])
        in_target_params = defaultdict(lambda: [])

        keyp_confs = defaultdict(lambda: [])
        for idx, target in enumerate(targets):
            # If there are no 3D annotations, skip and add to the starting
            # index the number of bounding boxes
            if len(target) < 1:
                continue

            conf = target.conf

            keyp_confs['body'].append(conf[self.body_idxs])
            keyp_confs['left_hand'].append(conf[self.left_hand_idxs])
            keyp_confs['right_hand'].append(conf[self.right_hand_idxs])
            keyp_confs['face'].append(conf[self.face_idxs])

            for param_key in PARAM_KEYS:
                if not target.has_field(param_key):
                    start_idxs[param_key] += len(target)
                    continue
                end_idx = start_idxs[param_key] + 1
                in_target_param_idxs[param_key] += list(
                    range(start_idxs[param_key], end_idx))
                start_idxs[param_key] += 1

                in_target_params[param_key].append(
                    target.get_field(param_key))

        # Stack all confidences
        for key in keyp_confs:
            keyp_confs[key] = torch.stack(keyp_confs[key])

        target_params = {}
        for key, val in in_target_params.items():
            if key == 'hand_pose':
                target_params['left_hand_pose'] = torch.stack([
                    t.left_hand_pose
                    for t in val])
                target_params['right_hand_pose'] = torch.stack([
                    t.right_hand_pose
                    for t in val])
            else:
                target_params[key] = torch.stack([
                    getattr(t, key)
                    for t in val])

        target_param_idxs = {}
        for key in in_target_param_idxs.keys():
            if key == 'hand_pose':
                target_param_idxs['left_hand_pose'] = torch.tensor(
                    np.asarray(in_target_param_idxs[key]),
                    device=device,
                    dtype=torch.long)
                target_param_idxs['right_hand_pose'] = target_param_idxs[
                    'left_hand_pose'].clone()
            else:
                target_param_idxs[key] = torch.tensor(
                    np.asarray(in_target_param_idxs[key]),
                    device=device,
                    dtype=torch.long)

        has_vertices = all([t.has_field('vertices') for t in targets])
        gt_vertices = None
        if has_vertices:
            gt_vertices = torch.stack([
                t.get_field('vertices').vertices for t in targets])

        stages_to_penalize = self.stages_to_penalize.copy()
        if -1 in stages_to_penalize:
            stages_to_penalize[stages_to_penalize.index(-1)] = num_stages + 1
        output_losses = {}
        for n in range(1, len(network_params) + 1):
            if n not in stages_to_penalize:
                continue
            curr_params = network_params[n - 1]
            if curr_params is None:
                logger.warning(f'Network output for stage {n} is None')
                continue

            curr_losses = self.single_loss_step(
                curr_params, target_params,
                target_param_idxs, device=device,
                keyp_confs=keyp_confs,
                gt_vertices=gt_vertices)
            for key in curr_losses:
                output_losses[f'stage_{n - 1:02d}_{key}'] = curr_losses[key]

        return output_losses


class RegularizerModule(nn.Module):
    def __init__(self, loss_cfg,
                 body_pose_mean=None, left_hand_pose_mean=None,
                 right_hand_pose_mean=None, jaw_pose_mean=None):
        super(RegularizerModule, self).__init__()

        self.stages_to_regularize = loss_cfg.get('stages_to_penalize', [-1])
        logger.info(f'Stages to regularize: {self.stages_to_regularize}')

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

        # Construct the body pose prior
        body_pose_prior_cfg = loss_cfg.body_pose.prior
        body_pose_prior_type = body_pose_prior_cfg.type
        self.body_pose_prior_weight = body_pose_prior_cfg.weight
        if self.body_pose_prior_weight > 0:
            self.body_pose_prior = build_prior(
                body_pose_prior_type,
                mean=body_pose_mean,
                **body_pose_prior_cfg)
            logger.debug(f'Body pose prior {self.body_pose_prior}')

        # Construct the left hand pose prior
        left_hand_prior_cfg = loss_cfg.left_hand_pose.prior
        left_hand_pose_prior_type = left_hand_prior_cfg.type
        self.left_hand_pose_prior_weight = left_hand_prior_cfg.weight
        if self.left_hand_pose_prior_weight > 0:
            self.left_hand_pose_prior = build_prior(
                left_hand_pose_prior_type,
                mean=left_hand_pose_mean,
                **left_hand_prior_cfg)
            logger.debug(f'Left hand pose prior {self.left_hand_pose_prior}')

        # Construct the right hand pose prior
        right_hand_prior_cfg = loss_cfg.right_hand_pose.prior
        right_hand_pose_prior_type = right_hand_prior_cfg.type
        self.right_hand_pose_prior_weight = right_hand_prior_cfg.weight
        if self.right_hand_pose_prior_weight > 0:
            self.right_hand_pose_prior = build_prior(
                right_hand_pose_prior_type, mean=right_hand_pose_mean,
                **right_hand_prior_cfg)
            logger.debug(f'Right hand pose prior {self.right_hand_pose_prior}')

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
            msg.append('Shape prior weight: {}'.format(
                self.shape_prior_weight))
        if self.expression_prior_weight > 0:
            msg.append('Expression prior weight: {}'.format(
                self.expression_prior_weight))
        if self.body_pose_prior_weight > 0:
            msg.append('Body pose prior weight: {}'.format(
                self.body_pose_prior_weight))
        if self.left_hand_pose_prior_weight > 0:
            msg.append('Left hand pose prior weight: {}'.format(
                self.left_hand_pose_prior_weight))
        if self.right_hand_pose_prior_weight > 0:
            msg.append('Right hand pose prior weight {}'.format(
                self.right_hand_pose_prior_weight))
        if self.jaw_pose_prior_weight > 0:
            msg.append('Jaw pose prior weight: {}'.format(
                self.jaw_pose_prior_weight))
        return '\n'.join(msg)

    def single_regularization_step(self, parameters,
                                   penalize_only_parts=False,
                                   **kwargs):
        prior_losses = {}

        betas = parameters.get('betas', None)
        reg_shape = (self.shape_prior_weight > 0 and betas is not None and
                     not penalize_only_parts)
        if reg_shape:
            prior_losses['shape_prior'] = (
                self.shape_prior_weight * self.shape_prior(betas))

        expression = parameters.get('expression', None)
        reg_expression = (
            self.expression_prior_weight > 0 and expression is not None)
        if reg_expression:
            prior_losses['expression_prior'] = (
                self.expression_prior(expression) *
                self.expression_prior_weight)

        body_pose = parameters.get('body_pose', None)
        betas = parameters.get('betas', None)
        reg_body_pose = (
            self.body_pose_prior_weight > 0 and body_pose is not None and
            not penalize_only_parts)
        if reg_body_pose:
            prior_losses['body_pose_prior'] = (
                self.body_pose_prior(body_pose) *
                self.body_pose_prior_weight)

        left_hand_pose = parameters.get('left_hand_pose', None)
        if (self.left_hand_pose_prior_weight > 0 and
                left_hand_pose is not None):
            prior_losses['left_hand_pose_prior'] = (
                self.left_hand_pose_prior(left_hand_pose) *
                self.left_hand_pose_prior_weight)

        right_hand_pose = parameters.get('right_hand_pose', None)
        if (self.right_hand_pose_prior_weight > 0 and
                right_hand_pose is not None):
            prior_losses['right_hand_pose_prior'] = (
                self.right_hand_pose_prior(right_hand_pose) *
                self.right_hand_pose_prior_weight)

        jaw_pose = parameters.get('jaw_pose', None)
        if self.jaw_pose_prior_weight > 0 and jaw_pose is not None:
            prior_losses['jaw_pose_prior'] = (
                self.jaw_pose_prior(jaw_pose) *
                self.jaw_pose_prior_weight)

        return prior_losses

    def forward(self,
                param_list,
                num_stages=3,
                **kwargs) -> Dict[str, Tensor]:

        prior_losses = defaultdict(lambda: 0)
        for n in range(1, num_stages + 1):
            if n not in self.stages_to_regularize:
                continue
            curr_params = param_list[n - 1]
            if curr_params is None:
                logger.warning(f'Network output for stage {n} is None')
                continue

            curr_losses = self.single_regularization_step(curr_params)
            for key in curr_losses:
                prior_losses[f'stage_{n - 1:02d}_{key}'] = curr_losses[key]

        if num_stages < len(param_list):
            curr_params = param_list[-1]
            final_losses = self.single_regularization_step(curr_params)
            for key in final_losses:
                prior_losses[
                    f'stage_{num_stages:02d}_{key}'] = final_losses[key]
        return prior_losses
