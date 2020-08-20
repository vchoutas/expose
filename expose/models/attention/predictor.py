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

from typing import List, Dict, Tuple, Callable, Optional, Union

from yacs.config import CfgNode
import time

from collections import defaultdict

import math
import os.path as osp

from loguru import logger

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hand_predictor import HandPredictor
from .head_predictor import HeadPredictor

from ..common.keypoint_loss import KeypointLoss
from ..common.smplx_loss_modules import SMPLXLossModule, RegularizerModule
from ..common.networks import FrozenBatchNorm2d
from ..common.mano_loss_modules import (
    MANOLossModule,
    RegularizerModule as MANORegularizer,
)
from ..common.flame_loss_modules import (
    FLAMELossModule,
    RegularizerModule as FLAMERegularizer
)


from smplx import build_layer as build_body_model
from smplx.utils import find_joint_kin_chain

from ..backbone import build_backbone
from ..common.networks import MLP, IterativeRegression
from ..common.bbox_sampler import CropSampler, ToCrops
from ..nnutils import init_weights
from ..common.pose_utils import build_all_pose_params
from ..camera import build_cam_proj, CameraParams
from ...losses import build_loss

from expose.data.targets import ImageList, ImageListPacked
from expose.data.targets.keypoints import KEYPOINT_NAMES, get_part_idxs
from expose.data.utils import flip_pose, bbox_iou, center_size_to_bbox

from expose.utils.typing_utils import Tensor


class SMPLXHead(nn.Module):

    def __init__(
        self,
        exp_cfg: CfgNode,
        dtype=torch.float32
    ) -> None:
        super(SMPLXHead, self).__init__()

        network_cfg = exp_cfg.get('network', {})
        attention_net_cfg = network_cfg.get('attention', {})
        smplx_net_cfg = attention_net_cfg.get('smplx', {})

        self.predict_body = network_cfg.get('predict_body', True)
        self.apply_hand_network_on_body = network_cfg.get(
            'apply_hand_network_on_body', True)
        self.apply_hand_network_on_hands = network_cfg.get(
            'apply_hand_network_on_hands', True)
        self.predict_hands = (self.apply_hand_network_on_body or
                              self.apply_hand_network_on_hands)
        logger.warning(
            f'Apply hand network on body: {self.apply_hand_network_on_body}')
        logger.warning(
            f'Apply hand network on hands: {self.apply_hand_network_on_hands}')
        logger.warning(f'Predict hands: {self.predict_hands}')
        self.apply_head_network_on_body = network_cfg.get(
            'apply_head_network_on_body', True)
        self.apply_head_network_on_head = network_cfg.get(
            'apply_head_network_on_head', True)
        self.predict_head = (self.apply_head_network_on_body or
                             self.apply_head_network_on_head)
        logger.warning(f'Predict head: {self.predict_head}')

        self.detach_mean = attention_net_cfg.get('detach_mean', False)

        condition_hand_on_body = attention_net_cfg.get(
            'condition_hand_on_body', {})
        self.condition_hand_on_body = any(condition_hand_on_body.values())
        logger.info(f'Condition hand on body: {self.condition_hand_on_body}')
        self.condition_hand_wrist_pose = condition_hand_on_body.get(
            'wrist_pose', True)
        logger.info(
            'Condition hand wrist pose on body: '
            f'{self.condition_hand_wrist_pose}')
        self.condition_hand_finger_pose = condition_hand_on_body.get(
            'finger_pose', True)
        logger.info(
            'Condition hand finger pose on body: '
            f'{self.condition_hand_finger_pose}')
        self.condition_hand_shape = condition_hand_on_body.get('shape', True)
        logger.info(
            f'Condition hand shape on body shape: {self.condition_hand_shape}')

        self.hand_add_shape_noise = network_cfg.get(
            'hand_add_shape_noise', False)
        self.hand_shape_std = network_cfg.get('hand_shape_std', 0.0)
        self.hand_shape_prob = network_cfg.get('hand_shape_prob', 0.0)
        logger.debug(
            'Add shape noise: {} from N(0, {}), with prob {}',
            self.hand_add_shape_noise,
            self.hand_shape_std,
            self.hand_shape_prob,
        )

        self.add_hand_pose_noise = network_cfg.get(
            'add_hand_pose_noise', False)
        self.hand_pose_std = network_cfg.get('hand_pose_std', 0.0)
        self.num_hand_components = network_cfg.get(
            'num_hand_components', 3)
        self.hand_noise_prob = network_cfg.get('hand_noise_prob', 0.0)
        logger.debug(
            'Add hand pose noise to {}: {} from N(0, {}) with prob {}',
            self.num_hand_components,
            self.add_hand_pose_noise, self.hand_pose_std,
            self.hand_noise_prob,)

        self.hand_randomize_global_orient = network_cfg.get(
            'hand_randomize_global_orient', False)
        self.hand_global_rot_max = network_cfg.get('hand_global_rot_max', 0.0)
        self.hand_global_rot_min = network_cfg.get('hand_global_rot_min', 0.0)
        self.hand_global_rot_noise_prob = network_cfg.get(
            'hand_global_rot_noise_prob', 0.0)
        logger.debug('Randomize global pose: {} from U({}, {})',
                     self.hand_randomize_global_orient,
                     self.hand_global_rot_min, self.hand_global_rot_max)

        condition_head_on_body = attention_net_cfg.get(
            'condition_head_on_body', {})
        self.condition_head_on_body = any(condition_head_on_body.values())

        self.condition_head_neck_pose = condition_head_on_body.get(
            'neck_pose', True)
        self.condition_head_jaw_pose = condition_head_on_body.get(
            'jaw_pose', True)
        self.condition_head_shape = condition_head_on_body.get(
            'shape', True)
        self.condition_head_expression = condition_head_on_body.get(
            'expression', True)
        logger.info(f'Condition head on body: {self.condition_head_on_body}')
        logger.info(
            f'Condition expression on body: {self.condition_head_expression}')
        logger.info(f'Condition shape on body: {self.condition_head_shape}')
        logger.info(
            f'Condition neck pose on body: {self.condition_head_neck_pose}')
        logger.info(
            f'Condition jaw pose on body: {self.condition_head_jaw_pose}')

        self.head_add_shape_noise = network_cfg.get(
            'head_add_shape_noise', False)
        self.head_shape_std = network_cfg.get('head_shape_std', 1.0)
        self.head_shape_prob = network_cfg.get('head_shape_prob', 0.0)
        logger.debug(
            'Add head shape noise: {} from N(0, {}), with prob {}',
            self.head_add_shape_noise,
            self.head_shape_std,
            self.head_shape_prob,
        )

        self.add_expression_noise = network_cfg.get(
            'add_expression_noise', False)
        self.expression_std = network_cfg.get('expression_std', None)
        self.expression_prob = network_cfg.get('expression_prob', 1.0)
        logger.debug(
            'Add expression noise: {} from N(0, {}), with prob {}',
            self.add_expression_noise,
            self.expression_std,
            self.expression_prob,
        )

        self.add_jaw_pose_noise = network_cfg.get('add_jaw_pose_noise', False)
        self.jaw_pose_min = network_cfg.get('jaw_pose_min', 0.0)
        self.jaw_pose_max = network_cfg.get('jaw_pose_max', 0.0)
        self.jaw_noise_prob = network_cfg.get('jaw_noise_prob', 1.0)
        logger.debug(
            'Sampling random X-axis jaw rotation from U({}, {}) with prob {}',
            self.jaw_pose_min, self.jaw_pose_max, self.jaw_noise_prob)

        self.head_randomize_global_orient = network_cfg.get(
            'head_randomize_global_orient', False)
        self.head_global_rot_min = network_cfg.get('head_global_rot_min', 0.0)
        self.head_global_rot_max = network_cfg.get('head_global_rot_max', 0.0)
        self.head_global_rot_noise_prob = network_cfg.get(
            'head_global_rot_noise_prob', 1.0)
        logger.debug(
            'Randomize head global pose: {} from U({}, {}) with prob {}',
            self.head_randomize_global_orient, self.head_global_rot_min,
            self.head_global_rot_max, self.head_global_rot_noise_prob,
        )

        body_model_cfg = exp_cfg.get('body_model', {})
        body_use_face_contour = body_model_cfg.get('use_face_contour', True)

        self.refine_shape_from_hands = attention_net_cfg.get(
            'refine_shape_from_hands', False)
        logger.debug(
            f'Refine shape from hands: {self.refine_shape_from_hands}')
        self.refine_shape_from_head = attention_net_cfg.get(
            'refine_shape_from_head', False)
        logger.debug(f'Refine shape from head: {self.refine_shape_from_head}')

        self.hand_bbox_thresh = attention_net_cfg.get('hand_bbox_thresh', 0.4)
        logger.debug(
            f'Hand bounding box IoU threshold: {self.hand_bbox_thresh}')
        self.head_bbox_thresh = attention_net_cfg.get('head_bbox_thresh', 0.4)
        logger.debug(
            f'Head bounding box IoU threshold: {self.head_bbox_thresh}')

        self.num_stages = smplx_net_cfg.get('num_stages', 3)
        self.append_params = smplx_net_cfg.get('append_params', True)

        self.pose_last_stage = smplx_net_cfg.get('pose_last_stage', False)

        self.body_model_cfg = body_model_cfg.copy()

        model_path = osp.expandvars(body_model_cfg.pop('model_folder', ''))
        model_type = body_model_cfg.pop('type', 'smplx')
        self.body_model = build_body_model(
            model_path,
            model_type=model_type,
            dtype=dtype,
            **body_model_cfg)
        logger.info(f'Body model: {self.body_model}')

        # The number of shape coefficients
        num_betas = body_model_cfg.num_betas
        self.num_betas = num_betas

        shape_mean_path = body_model_cfg.get('shape_mean_path', '')
        shape_mean_path = osp.expandvars(shape_mean_path)
        if osp.exists(shape_mean_path):
            shape_mean = torch.from_numpy(
                np.load(shape_mean_path, allow_pickle=True)).to(
                dtype=dtype).reshape(1, -1)[:, :num_betas].reshape(-1)
        else:
            shape_mean = torch.zeros([num_betas], dtype=dtype)

        # The number of expression coefficients
        num_expression_coeffs = body_model_cfg.num_expression_coeffs
        self.num_expression_coeffs = num_expression_coeffs
        expression_mean = torch.zeros(
            [num_expression_coeffs], dtype=dtype)

        # Build the pose parameterization for all the parameters
        pose_desc_dict = build_all_pose_params(
            body_model_cfg, 0, self.body_model,
            append_params=self.append_params, dtype=dtype)

        self.global_orient_decoder = pose_desc_dict['global_orient'].decoder
        global_orient_mean = pose_desc_dict['global_orient'].mean

        global_orient_type = body_model_cfg.get('global_orient', {}).get(
            'param_type', 'cont_rot_repr')
        # Rotate the model 180 degrees around the x-axis
        if global_orient_type == 'aa':
            global_orient_mean[0] = math.pi
        elif global_orient_type == 'cont_rot_repr':
            global_orient_mean[3] = -1
        global_orient_dim = pose_desc_dict['global_orient'].dim

        self.body_pose_decoder = pose_desc_dict['body_pose'].decoder
        body_pose_mean = pose_desc_dict['body_pose'].mean
        body_pose_dim = pose_desc_dict['body_pose'].dim

        self.left_hand_pose_decoder = pose_desc_dict['left_hand_pose'].decoder
        left_hand_pose_mean = pose_desc_dict['left_hand_pose'].mean
        left_hand_pose_dim = pose_desc_dict['left_hand_pose'].dim
        left_hand_pose_ind_dim = pose_desc_dict['left_hand_pose'].ind_dim

        self.right_hand_pose_decoder = pose_desc_dict[
            'right_hand_pose'].decoder
        right_hand_pose_mean = pose_desc_dict['right_hand_pose'].mean
        right_hand_pose_dim = pose_desc_dict['right_hand_pose'].dim
        right_hand_pose_ind_dim = pose_desc_dict['right_hand_pose'].ind_dim

        self.jaw_pose_decoder = pose_desc_dict['jaw_pose'].decoder
        jaw_pose_mean = pose_desc_dict['jaw_pose'].mean
        jaw_pose_dim = pose_desc_dict['jaw_pose'].dim

        mean_lst = []

        start = 0
        global_orient_idxs = list(range(start, start + global_orient_dim))

        global_orient_idxs = torch.tensor(global_orient_idxs, dtype=torch.long)
        self.register_buffer('global_orient_idxs', global_orient_idxs)
        start += global_orient_dim
        mean_lst.append(global_orient_mean.view(-1))

        body_pose_idxs = list(range(
            start, start + body_pose_dim))
        self.register_buffer(
            'body_pose_idxs', torch.tensor(body_pose_idxs, dtype=torch.long))
        start += body_pose_dim
        mean_lst.append(body_pose_mean.view(-1))

        left_hand_pose_idxs = list(range(start, start + left_hand_pose_dim))
        self.register_buffer(
            'left_hand_pose_idxs',
            torch.tensor(left_hand_pose_idxs, dtype=torch.long))
        start += left_hand_pose_dim
        mean_lst.append(left_hand_pose_mean.view(-1))

        right_hand_pose_idxs = list(range(
            start, start + right_hand_pose_dim))
        self.register_buffer(
            'right_hand_pose_idxs',
            torch.tensor(right_hand_pose_idxs, dtype=torch.long))
        start += right_hand_pose_dim
        mean_lst.append(right_hand_pose_mean.view(-1))

        jaw_pose_idxs = list(range(
            start, start + jaw_pose_dim))
        self.register_buffer(
            'jaw_pose_idxs', torch.tensor(jaw_pose_idxs, dtype=torch.long))
        start += jaw_pose_dim
        mean_lst.append(jaw_pose_mean.view(-1))

        shape_idxs = list(range(start, start + num_betas))
        self.register_buffer(
            'shape_idxs', torch.tensor(shape_idxs, dtype=torch.long))
        start += num_betas
        mean_lst.append(shape_mean.view(-1))

        expression_idxs = list(range(
            start, start + num_expression_coeffs))
        self.register_buffer(
            'expression_idxs', torch.tensor(expression_idxs, dtype=torch.long))
        start += num_expression_coeffs
        mean_lst.append(expression_mean.view(-1))

        camera_cfg = smplx_net_cfg.get('camera', {})
        camera_data = build_cam_proj(camera_cfg, dtype=dtype)
        self.projection = camera_data['camera']

        camera_param_dim = camera_data['dim']
        camera_mean = camera_data['mean']
        #  self.camera_mean = camera_mean
        self.register_buffer('camera_mean', camera_mean)
        self.camera_scale_func = camera_data['scale_func']

        camera_idxs = list(range(
            start, start + camera_param_dim))
        self.register_buffer(
            'camera_idxs', torch.tensor(camera_idxs, dtype=torch.long))
        start += camera_param_dim
        mean_lst.append(camera_mean)

        param_mean = torch.cat(mean_lst).view(1, -1)
        param_dim = param_mean.numel()

        # Construct the feature extraction backbone
        backbone_cfg = smplx_net_cfg.get('backbone', {})
        self.backbone, feat_dims = build_backbone(backbone_cfg)

        self.append_params = smplx_net_cfg.get('append_params', True)
        self.num_stages = smplx_net_cfg.get('num_stages', 1)

        self.body_feature_key = smplx_net_cfg.get('feature_key', 'avg_pooling')
        feat_dim = feat_dims[self.body_feature_key]

        regressor_cfg = smplx_net_cfg.get('mlp', {})
        regressor = MLP(feat_dim + self.append_params * param_dim,
                        param_dim, **regressor_cfg)
        self.regressor = IterativeRegression(
            regressor, param_mean, num_stages=self.num_stages)

        self.update_wrists = attention_net_cfg.get('update_wrists', True)
        # Find the kinematic chain for the right wrist
        right_wrist_idx = KEYPOINT_NAMES.index('right_wrist')
        self.right_wrist_idx = right_wrist_idx
        left_wrist_idx = KEYPOINT_NAMES.index('left_wrist')
        self.left_wrist_idx = left_wrist_idx

        self.hand_predictor = HandPredictor(
            exp_cfg,
            pose_desc_dict['global_orient'],
            pose_desc_dict['right_hand_pose'],
            camera_data,
            detach_mean=self.detach_mean,
            mean_pose_path=body_model_cfg.mean_pose_path,
            dtype=dtype)

        hand_crop_size = exp_cfg.get('datasets', {}).get('hand', {}).get(
            'transforms', {}).get('crop_size', 256)
        self.hand_scale_factor = attention_net_cfg.get('hand', {}).get(
            'scale_factor', 2.0)
        self.hand_crop_size = hand_crop_size
        self.hand_cropper = CropSampler(hand_crop_size)

        head_crop_size = exp_cfg.get('datasets', {}).get('head', {}).get(
            'transforms', {}).get('crop_size', 256)
        self.head_crop_size = head_crop_size
        self.head_scale_factor = network_cfg.get('head', {}).get(
            'scale_factor', 2.0)
        self.head_cropper = CropSampler(head_crop_size)

        self.head_predictor = HeadPredictor(
            exp_cfg,
            pose_desc_dict['global_orient'],
            pose_desc_dict['jaw_pose'], camera_data,
            detach_mean=self.detach_mean,
            dtype=dtype)
        self.points_to_crops = ToCrops()

        right_wrist_kin_chain = find_joint_kin_chain(
            right_wrist_idx,
            self.body_model.parents)
        right_wrist_kin_chain = torch.tensor(
            right_wrist_kin_chain, dtype=torch.long)
        self.register_buffer('right_wrist_kin_chain', right_wrist_kin_chain)

        self.register_buffer(
            'abs_pose_mean',
            self.global_orient_decoder.get_mean().unsqueeze(dim=0))

        # Find the kinematic chain for the left wrist
        left_wrist_kin_chain = find_joint_kin_chain(
            left_wrist_idx,
            self.body_model.parents)
        left_wrist_kin_chain = torch.tensor(
            left_wrist_kin_chain, dtype=torch.long)
        self.register_buffer('left_wrist_kin_chain', left_wrist_kin_chain)

        # Find the kinematic chain for the neck
        neck_idx = KEYPOINT_NAMES.index('neck')
        neck_kin_chain = find_joint_kin_chain(
            neck_idx,
            self.body_model.parents)
        self.register_buffer('neck_kin_chain',
                             torch.tensor(neck_kin_chain, dtype=torch.long))

        idxs_dict = get_part_idxs()
        body_idxs = idxs_dict['body']
        left_hand_idxs = idxs_dict['left_hand']
        right_hand_idxs = idxs_dict['right_hand']
        head_idxs = idxs_dict['head']
        if not body_use_face_contour:
            head_idxs = head_idxs[:-17]

        self.register_buffer('body_idxs', torch.tensor(body_idxs))
        self.register_buffer('left_hand_idxs', torch.tensor(left_hand_idxs))
        self.register_buffer('right_hand_idxs', torch.tensor(right_hand_idxs))
        self.register_buffer('head_idxs', torch.tensor(head_idxs))

        self.keyp_loss = KeypointLoss(exp_cfg)

        self.mask_hand_keyps = attention_net_cfg.get('mask_hand_keyps', True)
        self.mask_head_keyps = attention_net_cfg.get('mask_head_keyps', True)

        loss_cfg = exp_cfg.get('losses', {})
        # Create a loss to apply on the keypoints from the head crop
        head_crop_keypoint_loss_cfg = loss_cfg.get('head_crop_keypoints')
        self.head_crop_keyps_weight = head_crop_keypoint_loss_cfg.get(
            'weight', 0.0)
        self.head_crop_keyps_enable_at = head_crop_keypoint_loss_cfg.get(
            'enable', True)
        if self.head_crop_keyps_weight > 0:
            self.head_crop_keyps_loss = build_loss(
                **head_crop_keypoint_loss_cfg)
            logger.info(
                '2D Head crop keyps loss: {}', self.head_crop_keyps_loss)

        left_hand_crop_keypoint_loss_cfg = loss_cfg.get(
            'left_hand_crop_keypoints')
        self.left_hand_crop_keyps_weight = (
            left_hand_crop_keypoint_loss_cfg.get('weight', 0.0))
        self.left_hand_crop_keyps_enable_at = (
            left_hand_crop_keypoint_loss_cfg.get('enable', True))
        if self.left_hand_crop_keyps_weight > 0:
            self.left_hand_crop_keyps_loss = build_loss(
                **left_hand_crop_keypoint_loss_cfg)
            logger.info(
                '2D Left hand crop keyps loss: {}',
                self.left_hand_crop_keyps_loss)

        right_hand_crop_keypoint_loss_cfg = loss_cfg.get(
            'right_hand_crop_keypoints')
        self.right_hand_crop_keyps_weight = (
            right_hand_crop_keypoint_loss_cfg.get('weight', 0.0))
        self.right_hand_crop_keyps_enable_at = (
            right_hand_crop_keypoint_loss_cfg.get('enable', True))
        if self.right_hand_crop_keyps_weight > 0:
            self.right_hand_crop_keyps_loss = build_loss(
                **right_hand_crop_keypoint_loss_cfg)
            logger.info(
                '2D Left hand crop keyps loss: {}',
                self.right_hand_crop_keyps_loss)

        self.body_loss = SMPLXLossModule(
            loss_cfg,
            use_face_contour=body_use_face_contour)
        self.body_regularizer = RegularizerModule(
            loss_cfg, body_pose_mean=body_pose_mean,
            left_hand_pose_mean=left_hand_pose_mean,
            right_hand_pose_mean=right_hand_pose_mean,
            jaw_pose_mean=jaw_pose_mean
        )
        self.hand_loss = MANOLossModule(loss_cfg.get('hand', {}))
        self.hand_regularizer = MANORegularizer(loss_cfg.get('hand', {}))
        self.head_loss = FLAMELossModule(
            loss_cfg.get('head', {}), use_face_contour=body_use_face_contour)
        self.head_regularizer = FLAMERegularizer(loss_cfg.get('head', {}))

        self.freeze_body = attention_net_cfg.get('freeze_body', False)
        if self.freeze_body:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.regressor.parameters():
                param.requires_grad = False
            # Stop updating batch norm statistics
            self.backbone = FrozenBatchNorm2d.convert_frozen_batchnorm(
                self.backbone)
            self.regressor = FrozenBatchNorm2d.convert_frozen_batchnorm(
                self.regressor)

        # Build part merging functions
        hand_feat_dim = self.hand_predictor.get_feat_dim()
        head_feat_dim = self.head_predictor.get_feat_dim()
        # Right hand pose
        merging_cfg = attention_net_cfg.get('merging', {})
        self.right_hand_pose_merging_func = self._build_merge_func(
            merging_cfg,
            'right_hand_pose',
            body_feat_dim=feat_dim,
            body_param_dim=right_hand_pose_dim,
            part_feat_dim=hand_feat_dim,
            part_param_dim=right_hand_pose_dim,
        )
        # Right wrist pose
        if self.update_wrists:
            self.right_wrist_pose_merging_func = self._build_merge_func(
                merging_cfg,
                'right_wrist_pose',
                body_feat_dim=feat_dim,
                body_param_dim=right_hand_pose_ind_dim,
                part_feat_dim=hand_feat_dim,
                part_param_dim=right_hand_pose_ind_dim,
            )
        # Left hand pose
        self.left_hand_pose_merging_func = self._build_merge_func(
            merging_cfg,
            'left_hand_pose',
            body_feat_dim=feat_dim,
            body_param_dim=left_hand_pose_dim,
            part_feat_dim=hand_feat_dim,
            part_param_dim=left_hand_pose_dim,
        )
        # Left wrist pose
        if self.update_wrists:
            self.left_wrist_pose_merging_func = self._build_merge_func(
                merging_cfg,
                'left_wrist_pose',
                body_feat_dim=feat_dim,
                body_param_dim=left_hand_pose_ind_dim,
                part_feat_dim=hand_feat_dim,
                part_param_dim=left_hand_pose_ind_dim,
            )

        # Jaw pose
        self.jaw_pose_merging_func = self._build_merge_func(
            merging_cfg,
            'jaw_pose',
            body_feat_dim=feat_dim,
            body_param_dim=jaw_pose_dim,
            part_feat_dim=head_feat_dim,
            part_param_dim=jaw_pose_dim,
        )
        # Expression
        self.expression_merging_func = self._build_merge_func(
            merging_cfg,
            'expression',
            body_feat_dim=feat_dim,
            body_param_dim=num_expression_coeffs,
            part_feat_dim=head_feat_dim,
            part_param_dim=num_expression_coeffs,
        )

        hand_soft_weight_loss_cfg = loss_cfg.get('hand_soft_weight_loss', {})
        self.hand_soft_weight_loss = build_loss(**hand_soft_weight_loss_cfg)
        self.hand_soft_weight_loss_weight = hand_soft_weight_loss_cfg.get(
            'weight', 0.0)

        head_soft_weight_loss_cfg = loss_cfg.get('head_soft_weight_loss', {})
        self.head_soft_weight_loss = build_loss(**head_soft_weight_loss_cfg)
        self.head_soft_weight_loss_weight = head_soft_weight_loss_cfg.get(
            'weight', 0.0)

    def _build_merge_func(
            self, cfg: CfgNode,
            name: str,
            body_feat_dim: int, body_param_dim: int,
            part_feat_dim: int, part_param_dim: int,
    ) -> Callable:
        merge_type = cfg.get(name, {}).get('type', 'simple')
        logger.debug(f'Building "{merge_type}" merging function for "{name}"')
        if merge_type == 'none':
            pass
        elif merge_type == 'simple':
            def func(
                from_body: Tensor, from_part: Tensor,
                body_feat: Optional[Tensor] = None,
                part_feat: Optional[Tensor] = None,
                mask: Optional[Tensor] = None
            ) -> Dict[str, Tensor]:
                output = {}
                if self.training:
                    # During training, if a mask
                    output['merged'] = (
                        torch.where(
                            mask, from_part, from_body) if mask is not None
                        else from_part
                    )
                else:
                    output['merged'] = from_part
                output['weights'] = None
                return output
            return func
        else:
            raise ValueError(f'Merge function {merge_type} is not supported')

    def toggle_losses(self, iteration):
        self.body_loss.toggle_losses(iteration)
        self.keyp_loss.toggle_losses(iteration)

    def toggle_param_prediction(self, iteration):
        pass

    def flat_body_params_to_dict(self, param_tensor):
        global_orient = torch.index_select(
            param_tensor, 1, self.global_orient_idxs)
        body_pose = torch.index_select(
            param_tensor, 1, self.body_pose_idxs)
        left_hand_pose = torch.index_select(
            param_tensor, 1, self.left_hand_pose_idxs)
        right_hand_pose = torch.index_select(
            param_tensor, 1, self.right_hand_pose_idxs)
        jaw_pose = torch.index_select(
            param_tensor, 1, self.jaw_pose_idxs)
        betas = torch.index_select(param_tensor, 1, self.shape_idxs)
        expression = torch.index_select(param_tensor, 1, self.expression_idxs)

        return {
            'betas': betas,
            'expression': expression,
            'global_orient': global_orient,
            'body_pose': body_pose,
            'left_hand_pose': left_hand_pose,
            'right_hand_pose': right_hand_pose,
            'jaw_pose': jaw_pose,
        }

    def find_joint_global_rotation(
            self,
            kin_chain: Tensor,
            root_pose: Tensor,
            body_pose: Tensor
    ) -> Tensor:
        ''' Computes the absolute rotation of a joint from the kinematic chain
        '''
        # Create a single vector with all the poses
        parents_pose = torch.cat(
            [root_pose, body_pose], dim=1)[:, kin_chain]
        output_pose = parents_pose[:, 0]
        for idx in range(1, parents_pose.shape[1]):
            output_pose = torch.bmm(
                parents_pose[:, idx], output_pose)
        return output_pose

    def build_hand_mean(self, global_orient: Tensor,
                        body_pose: Tensor,
                        betas: Tensor,
                        flipped_left_hand_pose: Tensor,
                        right_hand_pose: Tensor,
                        hand_targets: List,
                        num_body_imgs: int = 0,
                        num_hand_imgs: int = 0
                        ) -> Tuple[Tensor, Tensor]:
        ''' Builds the initial point for the iterative regressor of the hand
        '''
        device, dtype = global_orient.device, global_orient.dtype
        hand_only_mean, parent_rots = [], []
        if num_body_imgs > 0:
            batch_size = num_body_imgs
            # Compute the absolute pose of the right wrist
            right_wrist_pose_abs = self.find_joint_global_rotation(
                self.right_wrist_kin_chain, global_orient,
                body_pose)

            right_wrist_parent_rot = self.find_joint_global_rotation(
                self.right_wrist_kin_chain[1:], global_orient,
                body_pose)

            left_wrist_parent_rot = self.find_joint_global_rotation(
                self.left_wrist_kin_chain[1:], global_orient, body_pose)
            left_to_right_wrist_parent_rot = flip_pose(
                left_wrist_parent_rot, pose_format='rot-mat')

            parent_rots += [
                right_wrist_parent_rot, left_to_right_wrist_parent_rot]

            #  if self.condition_hand_on_body:
            # Convert the absolute pose to the latent representation
            if self.condition_hand_wrist_pose:
                right_wrist_pose = self.global_orient_decoder.encode(
                    right_wrist_pose_abs.unsqueeze(dim=1)).reshape(
                        batch_size, -1)

                # Compute the absolute rotation for the left wrist
                left_wrist_pose_abs = self.find_joint_global_rotation(
                    self.left_wrist_kin_chain, global_orient, body_pose)
                # Flip the left wrist to the right
                left_to_right_wrist_pose = flip_pose(
                    left_wrist_pose_abs, pose_format='rot-mat')
                # Convert to the latent representation
                left_to_right_wrist_pose = self.global_orient_decoder.encode(
                    left_to_right_wrist_pose.unsqueeze(dim=1)).reshape(
                        batch_size, -1)
            else:
                right_wrist_pose = self.hand_predictor.get_wrist_pose_mean(
                    batch_size=batch_size)
                left_to_right_wrist_pose = (
                    self.hand_predictor.get_wrist_pose_mean(
                        batch_size=batch_size))

            # Convert the pose of the left hand to the right hand and project
            # it to the encoder space
            left_to_right_hand_pose = self.right_hand_pose_decoder.encode(
                flipped_left_hand_pose).reshape(batch_size, -1)

            camera_mean = self.hand_predictor.get_camera_mean().expand(
                batch_size, -1)

            shape_condition = (
                betas if self.condition_hand_shape else
                self.hand_predictor.get_shape_mean(batch_size)
            )
            right_finger_pose_condition = (
                right_hand_pose if self.condition_hand_finger_pose else
                self.hand_predictor.get_finger_pose_mean(batch_size)
            )
            right_hand_mean = torch.cat(
                [
                    right_wrist_pose, right_finger_pose_condition,
                    shape_condition, camera_mean,
                ], dim=1)
            left_finger_pose_condition = (
                left_to_right_hand_pose if self.condition_hand_finger_pose else
                self.hand_predictor.get_finger_pose_mean(batch_size)
            )
            # Should be Bx31
            left_hand_mean = torch.cat(
                [
                    left_to_right_wrist_pose,
                    left_finger_pose_condition,
                    shape_condition,
                    camera_mean,
                ], dim=1
            )
            hand_only_mean += [right_hand_mean, left_hand_mean]

        if num_hand_imgs > 0:
            mean_param = self.hand_predictor.get_param_mean(
                batch_size=num_hand_imgs,
                add_shape_noise=self.hand_add_shape_noise,
                shape_std=self.hand_shape_std,
                shape_prob=self.hand_shape_prob,
                num_hand_components=self.num_hand_components,
                add_hand_pose_noise=self.add_hand_pose_noise,
                hand_pose_std=self.hand_pose_std,
                hand_noise_prob=self.hand_noise_prob,
                targets=hand_targets,
                randomize_global_orient=self.hand_randomize_global_orient,
                global_rot_min=self.hand_global_rot_min,
                global_rot_max=self.hand_global_rot_max,
                global_rot_noise_prob=self.hand_global_rot_noise_prob,
            )

            hand_only_mean.append(mean_param)
            hand_only_parent_rots = torch.eye(
                3, device=device, dtype=dtype).reshape(
                    1, 3, 3).expand(num_hand_imgs, -1, -1).clone()
            hand_only_parent_rots[:, 1, 1] = -1
            hand_only_parent_rots[:, 2, 2] = -1
            parent_rots.append(hand_only_parent_rots)

        hand_only_mean = torch.cat(hand_only_mean, dim=0)
        parent_rots = torch.cat(parent_rots, dim=0)
        return hand_only_mean, parent_rots

    def build_head_mean(
        self,
        global_orient: Tensor,
        body_pose: Tensor,
        betas: Tensor,
        expression: Tensor,
        jaw_pose: Tensor,
        head_targets: List,
        num_body_imgs: int = 0,
        num_head_imgs: int = 0
    ) -> Tensor:
        ''' Builds the initial point of the head regressor
        '''
        head_only_mean = []
        if num_body_imgs > 0:
            batch_size = num_body_imgs

            # Compute the absolute pose of the right wrist
            neck_pose_abs = self.find_joint_global_rotation(
                self.neck_kin_chain, global_orient, body_pose)
            # Convert the absolute neck pose to offsets
            neck_latent = self.global_orient_decoder.encode(
                neck_pose_abs.unsqueeze(dim=1))
            neck_pose = neck_latent.reshape(batch_size, -1)

            camera_mean = self.head_predictor.get_camera_mean(
                batch_size=batch_size)

            neck_pose_condition = (
                neck_pose if self.condition_head_neck_pose else
                self.head_predictor.get_neck_pose_mean(batch_size))
            jaw_pose_condition = (
                jaw_pose.reshape(batch_size, -1)
                if self.condition_head_jaw_pose else
                self.head_predictor.get_jaw_pose_mean(batch_size)
            )
            head_num_betas = self.head_predictor.get_num_betas()
            shape_padding_size = head_num_betas - self.num_betas
            betas_condition = (
                F.pad(betas.reshape(batch_size, -1), (0, shape_padding_size))
                if self.condition_head_shape else
                self.head_predictor.get_shape_mean(batch_size=batch_size)
            )

            head_num_expression_coeffs = (
                self.head_predictor.get_num_expression_coeffs())
            expr_padding_size = (head_num_expression_coeffs -
                                 self.num_expression_coeffs)
            expression_condition = (
                F.pad(
                    expression.reshape(batch_size, -1), (0, expr_padding_size))
                if self.condition_head_expression else
                self.head_predictor.get_expression_mean(batch_size=batch_size)
            )

            # Should be Bx(Head pose params)
            head_only_mean.append(torch.cat(
                [neck_pose_condition, jaw_pose_condition,
                 betas_condition, expression_condition,
                 camera_mean.reshape(batch_size, -1),
                 ], dim=1
            ))

        if num_head_imgs > 0:
            mean_param = self.head_predictor.get_param_mean(
                batch_size=num_head_imgs,
                add_shape_noise=self.head_add_shape_noise,
                shape_std=self.head_shape_std,
                shape_prob=self.head_shape_prob,
                expression_prob=self.expression_prob,
                add_expression_noise=self.add_expression_noise,
                expression_std=self.expression_std,
                add_jaw_pose_noise=self.add_jaw_pose_noise,
                jaw_noise_prob=self.jaw_noise_prob,
                jaw_pose_min=self.jaw_pose_min,
                jaw_pose_max=self.jaw_pose_max,
                randomize_global_orient=self.head_randomize_global_orient,
                global_rot_noise_prob=self.head_global_rot_noise_prob,
                global_rot_min=self.head_global_rot_min,
                global_rot_max=self.head_global_rot_max,
                targets=head_targets,
            )
            head_only_mean.append(mean_param)

        head_only_mean = torch.cat(head_only_mean, dim=0)
        return head_only_mean

    def get_hand_model(self) -> nn.Module:
        ''' Return the hand predictor '''
        return self.hand_predictor

    def get_head_model(self) -> nn.Module:
        ''' Return the head predictor '''
        return self.head_predictor

    @torch.no_grad()
    def bboxes_to_mask(
            self,
            targets: List,
            key: str,
            est_center: Tensor, est_bbox_size: Tensor,
            thresh: float = 0.0) -> Tensor:
        ''' Converts bounding boxes to a binary mask '''
        if thresh <= 0:
            return torch.ones([len(targets), 1], dtype=torch.bool,
                              device=est_center.device)

        ious = torch.zeros(len(targets), dtype=est_center.dtype,
                           device=est_center.device)
        gt_idxs = []
        gt_bboxes = []
        for ii, t in enumerate(targets):
            if not t.has_field(key):
                continue
            gt_idxs.append(ii)
            bbox_field = t.get_field(key)
            gt_bboxes.append(bbox_field.bbox)

        if len(gt_bboxes) < 1:
            return ious.unsqueeze(dim=-1).to(dtype=torch.bool)
        est_bboxes = center_size_to_bbox(est_center, est_bbox_size)
        gt_bboxes = torch.stack(gt_bboxes).to(dtype=est_bboxes.dtype)
        gt_idxs = torch.tensor(
            gt_idxs, dtype=torch.long, device=est_bboxes.device)
        ious[gt_idxs] = bbox_iou(gt_bboxes, est_bboxes[gt_idxs])

        return ious.ge(thresh).unsqueeze(dim=-1)

    def forward(self,
                images: Tensor,
                targets: List = None,
                hand_imgs: Optional[Tensor] = None,
                hand_targets: Optional[List] = None,
                head_imgs: Optional[Tensor] = None,
                head_targets: Optional[List] = None,
                full_imgs: Optional[Union[ImageList, ImageListPacked]] = None,
                ) -> Dict[str, Dict[str, Tensor]]:
        ''' Forward pass of the attention predictor
        '''
        batch_size, _, crop_size, _ = images.shape
        device = images.device
        dtype = images.dtype

        feat_dict = self.backbone(images)
        body_features = feat_dict[self.body_feature_key]

        body_parameters, body_deltas = self.regressor(body_features)

        losses = {}
        # A list of dicts for the parameters predicted at each stage. The key
        # is the name of the parameters and the value is the prediction of the
        # model at the i-th stage of the iteration
        param_dicts = []
        # A dict of lists. Each key is the name of the parameter and the
        # corresponding item is a list of offsets that are predicted by the
        # model
        deltas_dict = defaultdict(lambda: [])
        param_delta_iter = zip(body_parameters, body_deltas)
        for idx, (params, deltas) in enumerate(param_delta_iter):
            curr_params_dict = self.flat_body_params_to_dict(params)

            out_dict = {}
            for key, val in curr_params_dict.items():
                if hasattr(self, f'{key}_decoder'):
                    decoder = getattr(self, f'{key}_decoder')
                    out_dict[key] = decoder(val)
                    out_dict[f'raw_{key}'] = val.clone()
                else:
                    out_dict[key] = val

            param_dicts.append(out_dict)
            curr_params_dict.clear()
            for key, val in self.flat_body_params_to_dict(deltas).items():
                deltas_dict[key].append(val)

        for key in deltas_dict:
            deltas_dict[key] = torch.stack(deltas_dict[key], dim=1).sum(dim=1)

        if self.pose_last_stage:
            merged_params = param_dicts[-1]
        else:
            merged_params = {}
            for key in param_dicts[0].keys():
                param = []
                for idx in range(self.num_stages):
                    if param_dicts[idx][key] is None:
                        continue
                    param.append(param_dicts[idx][key])
                merged_params[key] = torch.cat(param, dim=0)

        # Compute the body surface using the current estimation of the pose and
        # the shape
        body_model_output = self.body_model(
            get_skin=True, return_shaped=True, **merged_params)

        # Split the vertices, joints, etc. to stages
        out_params = defaultdict(lambda: dict())
        for key in body_model_output:
            if torch.is_tensor(body_model_output[key]):
                curr_val = body_model_output[key]
                out_list = torch.split(
                    curr_val, batch_size, dim=0)
                # If the number of outputs is equal to the number of stages
                # then store each stage
                if len(out_list) == self.num_stages:
                    for idx in range(len(out_list)):
                        out_params[f'stage_{idx:02d}'][key] = out_list[idx]
                # Else add only the last
                else:
                    out_key = f'stage_{self.num_stages - 1:02d}'
                    out_params[out_key][key] = out_list[-1]

        # Add the predicted parameters to the output dictionary
        for stage in range(self.num_stages):
            stage_key = f'stage_{stage:02d}'
            if len(out_params[stage_key]) < 1:
                continue
            out_params[stage_key].update(param_dicts[stage])
            out_params[stage_key]['faces'] = self.body_model.faces

        global_orient_from_body_net = param_dicts[-1]['global_orient'].clone()
        body_pose_from_body_net = param_dicts[-1]['body_pose'].clone()

        raw_body_pose_from_body_net = param_dicts[-1]['raw_body_pose'].clone(
        ).reshape(batch_size, 21, -1)
        raw_right_hand_pose_from_body_net = param_dicts[-1][
            'raw_right_hand_pose'].clone()
        left_hand_pose = param_dicts[-1]['left_hand_pose'].clone()
        right_hand_pose = param_dicts[-1]['right_hand_pose'].clone()
        jaw_pose = param_dicts[-1]['jaw_pose'].clone()

        # Extract the camera parameters estimated by the body only image
        camera_params = torch.index_select(
            body_parameters[-1], 1, self.camera_idxs)
        scale = camera_params[:, 0].view(-1, 1)
        translation = camera_params[:, 1:3]
        # Pass the predicted scale through exp() to make sure that the
        # scale values are always positive
        scale = self.camera_scale_func(scale)

        # Extract the final shape and expression parameters predicted by the
        # body only model
        betas = param_dicts[-1].get('betas').clone()
        expression = param_dicts[-1].get('expression')

        # Project the joints on the image plane
        proj_joints = self.projection(
            out_params[f'stage_{self.num_stages - 1:02d}']['joints'],
            scale=scale, translation=translation)

        # Add the projected joints
        out_params['proj_joints'] = proj_joints
        # the number of stages
        out_params['num_stages'] = self.num_stages
        # and the camera parameters to the output
        out_params['camera_parameters'] = CameraParams(
            translation=translation, scale=scale)

        # Clone the body pose so that we can update it with the predicted
        # sub-parts
        if self.predict_head or self.predict_hands:
            final_body_pose = raw_body_pose_from_body_net.clone()

        hand_predictions, head_predictions = {}, {}
        num_hand_imgs = 0
        left_hand_mask, right_hand_mask = None, None
        if self.predict_hands:
            if self.apply_hand_network_on_body:
                # Get the left, right and head crops from the full body
                left_hand_joints = (
                    (torch.index_select(proj_joints, 1, self.left_hand_idxs) *
                     0.5 + 0.5) * crop_size)
                #  left_hand_joints = torch.index_select(
                #  proj_joints, 1, self.left_hand_idxs)
                left_hand_points_to_crop = self.points_to_crops(
                    full_imgs, left_hand_joints, targets,
                    scale_factor=self.hand_scale_factor, crop_size=crop_size,
                )
                left_hand_center = left_hand_points_to_crop['center']
                left_hand_orig_bbox_size = left_hand_points_to_crop[
                    'orig_bbox_size']
                left_hand_bbox_size = left_hand_points_to_crop['bbox_size']
                left_hand_inv_crop_transforms = left_hand_points_to_crop[
                    'inv_crop_transforms']

                left_hand_cropper_out = self.hand_cropper(
                    full_imgs, left_hand_center, left_hand_orig_bbox_size)
                left_hand_crops = left_hand_cropper_out['images']
                left_hand_points = left_hand_cropper_out['sampling_grid']
                left_hand_crop_transform = left_hand_cropper_out['transform']

                right_hand_joints = (torch.index_select(
                    proj_joints, 1, self.right_hand_idxs) * 0.5 + 0.5) * crop_size
                right_hand_points_to_crop = self.points_to_crops(
                    full_imgs, right_hand_joints, targets,
                    scale_factor=self.hand_scale_factor, crop_size=crop_size,
                )
                right_hand_center = right_hand_points_to_crop['center']
                right_hand_orig_bbox_size = right_hand_points_to_crop[
                    'orig_bbox_size']
                right_hand_bbox_size = right_hand_points_to_crop['bbox_size']

                right_hand_cropper_out = self.hand_cropper(
                    full_imgs, right_hand_center, right_hand_orig_bbox_size)
                right_hand_crops = right_hand_cropper_out['images']
                right_hand_points = right_hand_cropper_out['sampling_grid']
                right_hand_crop_transform = right_hand_cropper_out['transform']

                # Store the transformation parameters
                out_params['left_hand_crops'] = left_hand_crops.detach()
                out_params['left_hand_points'] = left_hand_points.detach()
                out_params['right_hand_crops'] = right_hand_crops.detach()
                out_params['right_hand_points'] = right_hand_points.detach()

                out_params['right_hand_crop_transform'] = (
                    right_hand_crop_transform.detach())
                out_params['left_hand_crop_transform'] = (
                    left_hand_crop_transform.detach())

                out_params['left_hand_hd_to_crop'] = (
                    left_hand_cropper_out['hd_to_crop'])
                out_params['left_hand_inv_crop_transforms'] = (
                    left_hand_points_to_crop['inv_crop_transforms'])

                out_params['right_hand_hd_to_crop'] = (
                    right_hand_cropper_out['hd_to_crop'])
                out_params['right_hand_inv_crop_transforms'] = (
                    right_hand_points_to_crop['inv_crop_transforms'])

            # Flip the left hand to a right hand
            all_hand_imgs = []
            hand_global_orient = []
            hand_body_pose = []
            if self.apply_hand_network_on_body:
                all_hand_imgs.append(right_hand_crops)
                all_hand_imgs.append(torch.flip(left_hand_crops, dims=(-1,)))
                hand_global_orient += [
                    global_orient_from_body_net,
                    flip_pose(
                        global_orient_from_body_net, pose_format='rot-mat')]
                hand_body_pose += [
                    body_pose_from_body_net, body_pose_from_body_net]

            if hand_imgs is not None and self.apply_hand_network_on_hands:
                # Add the hand only images
                num_hand_imgs = len(hand_imgs)
                all_hand_imgs.append(hand_imgs)

                body_identity = torch.eye(
                    3, device=device, dtype=dtype).reshape(1, 1, 3, 3).expand(
                        num_hand_imgs, body_pose_from_body_net.shape[1], -1,
                        -1)
                hand_body_pose.append(body_identity)
                global_identity = torch.eye(
                    3, device=device, dtype=dtype).reshape(
                        1, 1, 3, 3).expand(
                            num_hand_imgs,
                            global_orient_from_body_net.shape[1], -1, -1).clone()
                global_identity[:, :, 1, 1] = -1
                global_identity[:, :, 2, 2] = -1
                hand_global_orient.append(global_identity)

            num_body_imgs = (
                batch_size if self.apply_hand_network_on_body else 0)
            num_hand_net_ins = len(hand_body_pose) + num_body_imgs
            if num_hand_net_ins > 0:
                hand_body_pose = torch.cat(hand_body_pose, dim=0)
                hand_global_orient = torch.cat(hand_global_orient, dim=0)

                # Flip the pose of the left hand
                flipped_left_hand_pose = flip_pose(
                    param_dicts[-1]['left_hand_pose'], pose_format='rot-mat')

                # Build the mean used to condition the hand network using the
                # parameters estimated by the body network
                hand_mean, parent_rots = self.build_hand_mean(
                    param_dicts[-1]['global_orient'],
                    param_dicts[-1]['body_pose'],
                    betas=param_dicts[-1]['betas'],
                    flipped_left_hand_pose=flipped_left_hand_pose,
                    right_hand_pose=param_dicts[-1]['raw_right_hand_pose'],
                    hand_targets=hand_targets,
                    num_body_imgs=num_body_imgs,
                    num_hand_imgs=num_hand_imgs,
                )

                # Feed the hand images and the offsets to the hand-only
                # predictor
                all_hand_imgs = torch.cat(all_hand_imgs, dim=0)

                hand_predictions = self.hand_predictor(
                    all_hand_imgs,
                    hand_mean=hand_mean,
                    global_orient_from_body_net=hand_global_orient,
                    body_pose_from_body_net=hand_body_pose,
                    parent_rots=parent_rots,
                    num_hand_imgs=num_hand_imgs,
                )
                num_hand_stages = hand_predictions.get('num_stages', 1)
                hand_network_output = hand_predictions.get(
                    f'stage_{num_hand_stages - 1:02d}')

            if self.apply_hand_network_on_body:
                # Find which images belong to the left hand and which ones to
                # the right hand
                hands_from_body_idxs = torch.arange(
                    0, 2 * batch_size, dtype=torch.long, device=device)
                right_hand_from_body_idxs = hands_from_body_idxs[
                    :batch_size]
                left_hand_from_body_idxs = hands_from_body_idxs[batch_size:]

                right_hand_features = hand_predictions.get(
                    'features')[right_hand_from_body_idxs]
                left_hand_features = hand_predictions.get(
                    'features')[left_hand_from_body_idxs]

                right_hand_mask = None
                raw_right_hand_pose_dict = self.right_hand_pose_merging_func(
                    from_body=raw_right_hand_pose_from_body_net,
                    from_part=hand_network_output.get(
                        'raw_right_hand_pose')[right_hand_from_body_idxs],
                    body_feat=body_features,
                    part_feat=right_hand_features,
                    mask=right_hand_mask,
                )
                raw_right_hand_pose = raw_right_hand_pose_dict['merged']

                if self.update_wrists:
                    right_wrist_pose_from_part = hand_network_output.get(
                        'raw_right_wrist_pose')
                    right_wrist_pose_from_body = raw_body_pose_from_body_net[
                        :, self.right_wrist_idx - 1]
                    raw_right_wrist_pose_dict = (
                        self.right_wrist_pose_merging_func(
                            from_body=right_wrist_pose_from_body,
                            from_part=right_wrist_pose_from_part,
                            body_feat=body_features,
                            part_feat=right_hand_features,
                            mask=right_hand_mask,
                        )
                    )
                    raw_right_wrist_pose = raw_right_wrist_pose_dict['merged']
                    final_body_pose[:, self.right_wrist_idx - 1] = (
                        raw_right_wrist_pose)

                # Project the flipped left hand pose to the rotation latent
                # space using the decoder for the right hand
                raw_left_to_right_hand_pose = (
                    self.right_hand_pose_decoder.encode(
                        flipped_left_hand_pose).reshape(batch_size, -1))
                # Convert the pose of the left hand to the right hand and
                # project it to the encoder space
                raw_left_to_right_hand_pose_from_body = (
                    self.right_hand_pose_decoder.encode(
                        flipped_left_hand_pose).reshape(batch_size, -1))
                # Merge the predictions of the body network and the part
                # network for the articulation of the left hand
                left_hand_pose_from_part = hand_network_output.get(
                    'raw_right_hand_pose')[left_hand_from_body_idxs]
                raw_left_to_right_hand_pose_dict = (
                    self.left_hand_pose_merging_func(
                        from_body=raw_left_to_right_hand_pose_from_body,
                        from_part=left_hand_pose_from_part,
                        body_feat=body_features,
                        part_feat=left_hand_features,
                        mask=left_hand_mask,
                    )
                )
                raw_left_to_right_hand_pose = raw_left_to_right_hand_pose_dict[
                    'merged']

                if self.update_wrists:
                    left_wrist_pose_from_part = hand_network_output.get(
                        'raw_left_wrist_pose')
                    left_wrist_pose_from_body = raw_body_pose_from_body_net[
                        :, self.left_wrist_idx - 1]
                    raw_left_wrist_pose_dict = (
                        self.left_wrist_pose_merging_func(
                            from_body=left_wrist_pose_from_body,
                            from_part=left_wrist_pose_from_part,
                            body_feat=body_features,
                            part_feat=left_hand_features,
                            mask=left_hand_mask,
                        )
                    )
                    raw_left_wrist_pose = raw_left_wrist_pose_dict['merged']
                    final_body_pose[:, self.left_wrist_idx - 1] = (
                        raw_left_wrist_pose)

                right_hand_pose = self.right_hand_pose_decoder(
                    raw_right_hand_pose)
                # Decode the predicted pose and flip it back to the left hand
                # space
                left_hand_pose = flip_pose(self.right_hand_pose_decoder(
                    raw_left_to_right_hand_pose), pose_format='rot-mat')

        num_head_imgs = 0
        head_mask = None
        if self.predict_head:
            if self.apply_head_network_on_body:
                head_joints = (torch.index_select(
                    proj_joints, 1, self.head_idxs) * 0.5 + 0.5) * crop_size
                #  head_joints = torch.index_select(
                #  proj_joints, 1, self.head_idxs)
                head_point_to_crop_output = self.points_to_crops(
                    full_imgs, head_joints, targets,
                    scale_factor=self.head_scale_factor, crop_size=crop_size,
                )
                head_center = head_point_to_crop_output['center']
                head_orig_bbox_size = head_point_to_crop_output[
                    'orig_bbox_size']
                head_bbox_size = head_point_to_crop_output['bbox_size']
                head_inv_crop_transforms = head_point_to_crop_output[
                    'inv_crop_transforms']

                head_cropper_out = self.head_cropper(
                    full_imgs, head_center, head_orig_bbox_size)
                head_crops = head_cropper_out['images']
                head_points = head_cropper_out['sampling_grid']
                # Contains the transformation that is used to transform the
                # sampling grid from head image coordinates to HD image
                # coordinates.
                head_crop_transform = head_cropper_out['transform']

                out_params['head_crops'] = head_crops.detach()
                out_params['head_points'] = head_points.detach()
                out_params['head_crop_transform'] = (
                    head_crop_transform.detach())

                out_params['head_hd_to_crop'] = head_cropper_out['hd_to_crop']
                out_params['head_inv_crop_transforms'] = (
                    head_point_to_crop_output['inv_crop_transforms'])

            all_head_imgs = []
            if self.apply_head_network_on_body:
                all_head_imgs.append(head_crops)

            # The global and body pose data used to pose the model inside the
            # head-only sub-network.
            head_global_orient, head_body_pose = [], []
            if self.apply_head_network_on_body:
                head_global_orient += [global_orient_from_body_net]
                head_body_pose += [body_pose_from_body_net]

            if head_imgs is not None and self.apply_head_network_on_head:
                all_head_imgs.append(head_imgs)
                num_head_imgs = len(head_imgs)
                body_identity = torch.eye(
                    3, device=device, dtype=dtype).reshape(
                        1, 1, 3, 3).expand(
                            num_head_imgs, body_pose_from_body_net.shape[1],
                            -1, -1)
                head_body_pose.append(body_identity)
                global_identity = torch.eye(
                    3, device=device, dtype=dtype).reshape(
                        1, 1, 3, 3).expand(num_head_imgs, -1, -1, -1).clone()
                global_identity[:, :, 1, 1] = -1
                global_identity[:, :, 2, 2] = -1
                head_global_orient.append(global_identity)

            num_body_imgs = (
                batch_size if self.apply_head_network_on_body else 0
            )
            num_head_net_ins = len(head_global_orient) + num_body_imgs
            if num_head_net_ins > 0:
                head_global_orient = torch.cat(head_global_orient, dim=0)
                head_body_pose = torch.cat(head_body_pose, dim=0)

                head_mean = self.build_head_mean(
                    param_dicts[-1]['global_orient'],
                    param_dicts[-1]['body_pose'],
                    betas=param_dicts[-1]['betas'],
                    expression=param_dicts[-1]['expression'],
                    jaw_pose=param_dicts[-1]['raw_jaw_pose'],
                    num_head_imgs=num_head_imgs,
                    num_body_imgs=num_body_imgs,
                    head_targets=head_targets,
                )
                all_head_imgs = torch.cat(all_head_imgs, dim=0)

                head_predictions = self.head_predictor(
                    all_head_imgs,
                    head_mean=head_mean,
                    global_orient_from_body_net=head_global_orient,
                    body_pose_from_body_net=head_body_pose,
                    num_head_imgs=num_head_imgs,
                )

                num_head_stages = head_predictions.get('num_stages', 1)
                head_network_output = head_predictions.get(
                    f'stage_{num_head_stages - 1:02d}')
                if self.apply_head_network_on_body:
                    head_from_body_idxs = torch.arange(
                        0, batch_size, dtype=torch.long, device=device)
                    head_features = head_predictions.get(
                        'features')[head_from_body_idxs]
                    # During training only use predictions from bounding boxes
                    # with enough IoU.
                    head_mask = None
                    raw_jaw_pose_from_body = param_dicts[-1].get(
                        'raw_jaw_pose')
                    # Replace the jaw pose only from the predictions taken from
                    # valid head crops
                    raw_jaw_pose_from_part = head_network_output.get(
                        'raw_jaw_pose')[head_from_body_idxs]
                    raw_jaw_pose_dict = self.jaw_pose_merging_func(
                        from_body=raw_jaw_pose_from_body,
                        from_part=raw_jaw_pose_from_part,
                        body_feat=body_features,
                        part_feat=head_features,
                        mask=head_mask,
                    )
                    raw_jaw_pose = raw_jaw_pose_dict['merged']

                    expression_from_body = param_dicts[-1].get('expression')
                    expression_from_head = head_network_output.get(
                        'expression')[head_from_body_idxs,
                                      :self.num_expression_coeffs]
                    expression_dict = self.expression_merging_func(
                        from_body=expression_from_body,
                        from_part=expression_from_head,
                        body_feat=body_features,
                        part_feat=head_features,
                        mask=head_mask,
                    )
                    expression = expression_dict['merged']
                    jaw_pose = self.jaw_pose_decoder(raw_jaw_pose)


        if self.predict_head or self.predict_hands:
            body_pose = self.body_pose_decoder(
                final_body_pose.reshape(batch_size, -1))
        else:
            body_pose = body_pose_from_body_net

        final_body_parameters = {
            'global_orient': param_dicts[-1].get('global_orient'),
            'body_pose': body_pose,
            'left_hand_pose': left_hand_pose,
            'right_hand_pose': right_hand_pose,
            'jaw_pose': jaw_pose,
            'betas': betas,
            'expression': expression
        }

        if self.apply_hand_network_on_body or self.apply_head_network_on_body:
            # Compute the mesh using the new hand and face parameters
            final_body_model_output = self.body_model(
                get_skin=True, return_shaped=True, **final_body_parameters)
            param_dicts.append({
                **final_body_parameters, **final_body_model_output})

        if (self.apply_hand_network_on_body or
                self.apply_head_network_on_body):
            out_params['final'] = {
                **final_body_parameters, **final_body_model_output}
            joints3d = final_body_model_output.get('joints')
            proj_joints = self.projection(
                joints3d, scale=scale, translation=translation)
            out_params['final_proj_joints'] = proj_joints
            # Update the camera parameters with the new projected joints
            out_params['proj_joints'] = proj_joints
            out_params['final']['proj_joints'] = proj_joints
        else:
            joints3d = out_params[f'stage_{self.num_stages - 1:02d}']['joints']

        body_crop_size = images.shape[2]
        # Convert the projected joints from [-1, 1] to body image
        # coordinates
        proj_joints_in_body_crop = (
            proj_joints * 0.5 + 0.5) * body_crop_size

        # Transform the projected points back to the HD image
        if self.apply_head_network_on_body:
            hd_proj_joints = torch.einsum(
                'bij,bkj->bki',
                [head_inv_crop_transforms[:, :2, :2],
                 proj_joints_in_body_crop]) + head_inv_crop_transforms[
                     :, :2, 2].unsqueeze(dim=1)
            out_params['hd_proj_joints'] = hd_proj_joints.detach()
        elif self.apply_hand_network_on_body:
            hd_proj_joints = torch.einsum(
                'bij,bkj->bki',
                [left_hand_inv_crop_transforms[:, :2, :2],
                 proj_joints_in_body_crop]) + left_hand_inv_crop_transforms[
                     :, :2, 2].unsqueeze(dim=1)
            out_params['hd_proj_joints'] = hd_proj_joints.detach()

        if self.apply_head_network_on_body:
            inv_head_crop_transf = torch.inverse(head_crop_transform)
            head_img_keypoints = torch.einsum(
                'bij,bkj->bki',
                [inv_head_crop_transf[:, :2, :2],
                 hd_proj_joints]) + inv_head_crop_transf[:, :2, 2].unsqueeze(
                     dim=1)
            out_params['head_proj_joints'] = (
                head_img_keypoints.detach() * self.head_crop_size)

        if self.apply_hand_network_on_body:
            inv_left_hand_crop_transf = torch.inverse(left_hand_crop_transform)
            left_hand_img_keypoints = torch.einsum(
                'bij,bkj->bki',
                [inv_left_hand_crop_transf[:, :2, :2],
                 hd_proj_joints]) + inv_left_hand_crop_transf[
                     :, :2, 2].unsqueeze(dim=1)
            out_params['left_hand_proj_joints'] = (
                left_hand_img_keypoints.detach() * self.hand_crop_size)

            inv_right_hand_crop_transf = torch.inverse(
                right_hand_crop_transform)
            right_hand_img_keypoints = torch.einsum(
                'bij,bkj->bki',
                [inv_right_hand_crop_transf[:, :2, :2],
                 hd_proj_joints]) + inv_right_hand_crop_transf[
                     :, :2, 2].unsqueeze(dim=1)
            out_params['right_hand_proj_joints'] = (
                right_hand_img_keypoints.detach() * self.hand_crop_size)

        if self.training:
            # Create the tensor of ground-truth HD keypoints
            gt_hd_keypoints = []
            for t in targets:
                gt_hd_keypoints.append(t.get_field('keypoints_hd'))

            gt_hd_keypoints_with_conf = torch.tensor(
                gt_hd_keypoints, dtype=dtype, device=device)
            gt_hd_keypoints_conf = gt_hd_keypoints_with_conf[:, :, -1]
            gt_hd_keypoints = gt_hd_keypoints_with_conf[:, :, :-1]
            out_params['gt_conf'] = gt_hd_keypoints_conf.detach()

            if self.apply_head_network_on_body:
                # Convert the ground-truth HD keypoints to the head image space
                gt_head_keypoints = torch.einsum(
                    'bij,bkj->bki',
                    [inv_head_crop_transf[:, :2, :2],
                     gt_hd_keypoints]) + inv_head_crop_transf[
                         :, :2, 2].unsqueeze(dim=1)
                out_params['gt_head_keypoints'] = (
                    gt_head_keypoints.detach() * self.head_crop_size)

            # Convert the ground-truth HD keypoints to the left and right hand
            # image space
            if self.apply_hand_network_on_body:
                gt_right_hand_keypoints = (
                    torch.einsum(
                        'bij,bkj->bki',
                        [inv_right_hand_crop_transf[:, :2, :2],
                         gt_hd_keypoints]) +
                    inv_right_hand_crop_transf[:, :2, 2].unsqueeze(dim=1))
                gt_left_hand_keypoints = (
                    torch.einsum(
                        'bij,bkj->bki',
                        [inv_left_hand_crop_transf[:, :2, :2],
                         gt_hd_keypoints]) +
                    inv_left_hand_crop_transf[:, :2, 2].unsqueeze(dim=1))

                out_params['gt_right_hand_keypoints'] = (
                    gt_right_hand_keypoints.detach() * self.hand_crop_size)
                out_params['gt_left_hand_keypoints'] = (
                    gt_left_hand_keypoints.detach() * self.hand_crop_size)

        output = {
            'body': out_params,
            'losses': losses
        }

        return output
