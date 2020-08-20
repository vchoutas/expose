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

import numpy as np

import torch
import torch.nn as nn

from loguru import logger

from ...losses import build_loss

from ...data.targets.keypoints import (
    get_part_idxs, KEYPOINT_NAMES,
    BODY_CONNECTIONS, HAND_CONNECTIONS, FACE_CONNECTIONS)


class KeypointLoss(nn.Module):
    def __init__(self, exp_cfg):
        super(KeypointLoss, self).__init__()
        self.left_hip_idx = KEYPOINT_NAMES.index('left_hip')
        self.right_hip_idx = KEYPOINT_NAMES.index('right_hip')

        self.body_joints_2d_weight = exp_cfg.losses.body_joints_2d.weight
        if self.body_joints_2d_weight > 0:
            self.body_joints_2d_loss = build_loss(
                **exp_cfg.losses.body_joints_2d)
            logger.debug('2D body joints loss: {}', self.body_joints_2d_loss)

        hand_joints2d_cfg = exp_cfg.losses.hand_joints_2d
        self.hand_joints_2d_weight = hand_joints2d_cfg.weight
        self.hand_joints_2d_enable_at = hand_joints2d_cfg.enable
        self.hand_joints_2d_active = False
        if self.hand_joints_2d_weight > 0:
            hand_joints2d_cfg = exp_cfg.losses.hand_joints_2d
            self.hand_joints_2d_loss = build_loss(**hand_joints2d_cfg)
            logger.debug('2D hand joints loss: {}', self.hand_joints_2d_loss)

        face_joints2d_cfg = exp_cfg.losses.face_joints_2d
        self.face_joints_2d_weight = face_joints2d_cfg.weight
        self.face_joints_2d_enable_at = face_joints2d_cfg.enable
        self.face_joints_2d_active = False
        if self.face_joints_2d_weight > 0:
            self.face_joints_2d_loss = build_loss(**face_joints2d_cfg)
            logger.debug('2D face joints loss: {}', self.face_joints_2d_loss)

        use_face_contour = exp_cfg.datasets.use_face_contour
        idxs_dict = get_part_idxs()
        body_idxs = idxs_dict['body']
        hand_idxs = idxs_dict['hand']
        face_idxs = idxs_dict['face']
        if not use_face_contour:
            face_idxs = face_idxs[:-17]

        self.register_buffer('body_idxs', torch.tensor(body_idxs))
        self.register_buffer('hand_idxs', torch.tensor(hand_idxs))
        self.register_buffer('face_idxs', torch.tensor(face_idxs))

        self.body_joints_3d_weight = exp_cfg.losses.body_joints_3d.weight
        if self.body_joints_3d_weight > 0:
            self.body_joints_3d_loss = build_loss(
                **exp_cfg.losses.body_joints_3d)
            logger.debug('3D body_joints loss: {}', self.body_joints_3d_loss)

        hand_joints3d_cfg = exp_cfg.losses.hand_joints_3d
        self.hand_joints_3d_weight = hand_joints3d_cfg.weight
        self.hand_joints_3d_enable_at = hand_joints3d_cfg.enable
        if self.hand_joints_3d_weight > 0:
            self.hand_joints_3d_loss = build_loss(**hand_joints3d_cfg)
            logger.debug('3D hand joints loss: {}', self.hand_joints_3d_loss)
        self.hand_joints_3d_active = False

        face_joints3d_cfg = exp_cfg.losses.face_joints_3d
        self.face_joints_3d_weight = face_joints3d_cfg.weight
        self.face_joints_3d_enable_at = face_joints3d_cfg.enable
        if self.face_joints_3d_weight > 0:
            face_joints3d_cfg = exp_cfg.losses.face_joints_3d
            self.face_joints_3d_loss = build_loss(**face_joints3d_cfg)
            logger.debug('3D face joints loss: {}', self.face_joints_3d_loss)
        self.face_joints_3d_active = False

        body_edge_2d_cfg = exp_cfg.losses.get('body_edge_2d', {})
        self.body_edge_2d_weight = body_edge_2d_cfg.weight
        self.body_edge_2d_enable_at = body_edge_2d_cfg.enable
        if self.body_edge_2d_weight > 0:
            self.body_edge_2d_loss = build_loss(type='keypoint-edge',
                                                connections=BODY_CONNECTIONS,
                                                **body_edge_2d_cfg)
            logger.debug('2D body edge loss: {}', self.body_edge_2d_loss)
        self.body_edge_2d_active = False

        hand_edge_2d_cfg = exp_cfg.losses.get('hand_edge_2d', {})
        self.hand_edge_2d_weight = hand_edge_2d_cfg.get('weight', 0.0)
        self.hand_edge_2d_enable_at = hand_edge_2d_cfg.get('enable', 0)
        if self.hand_edge_2d_weight > 0:
            self.hand_edge_2d_loss = build_loss(type='keypoint-edge',
                                                connections=HAND_CONNECTIONS,
                                                **hand_edge_2d_cfg)
            logger.debug('2D hand edge loss: {}', self.hand_edge_2d_loss)
        self.hand_edge_2d_active = False

        face_edge_2d_cfg = exp_cfg.losses.get('face_edge_2d', {})
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
                type='keypoint-edge', connections=face_connections,
                **face_edge_2d_cfg)
            logger.debug('2D face edge loss: {}', self.face_edge_2d_loss)
        self.face_edge_2d_active = False

    def extra_repr(self):
        msg = []
        msg.append(f'Body joints 2D: {self.body_joints_2d_weight}')
        msg.append(f'Hand joints 2D: {self.hand_joints_2d_weight}')
        msg.append(f'Face joints 2D: {self.face_joints_2d_weight}')

        msg.append(f'Body joints 3D: {self.body_joints_3d_weight}')
        msg.append(f'Hand joints 3D: {self.hand_joints_3d_weight}')
        msg.append(f'Face joints 3D: {self.face_joints_3d_weight}')

        msg.append(f'Body edge 2D: {self.body_edge_2d_weight}')
        msg.append(f'Hand edge 2D: {self.hand_edge_2d_weight}')
        msg.append(f'Face edge 2D: {self.face_edge_2d_weight}')

        return '\n'.join(msg)

    def toggle_losses(self, iteration: int) -> None:
        if hasattr(self, 'hand_joints_2d_enable_at'):
            self.hand_joints_2d_active = (
                iteration >= self.hand_joints_2d_enable_at)
        if hasattr(self, 'face_joints_2d_enable_at'):
            self.face_joints_2d_active = (iteration >=
                                          self.face_joints_2d_enable_at)
        if hasattr(self, 'hand_joints_3d_enable_at'):
            self.hand_joints_3d_active = (iteration >=
                                          self.hand_joints_3d_enable_at)
        if hasattr(self, 'face_joints_3d_enable_at'):
            self.face_joints_3d_active = (
                iteration >= self.face_joints_3d_enable_at)
        if hasattr(self, 'body_edge_2d_enable_at'):
            self.body_edge_2d_active = (
                iteration >= self.body_edge_2d_enable_at)
        if hasattr(self, 'hand_edge_2d_enable_at'):
            self.hand_edge_2d_active = (
                iteration >= self.hand_edge_2d_enable_at)
        if hasattr(self, 'face_edge_2d_enable_at'):
            self.face_edge_2d_active = (
                iteration >= self.face_edge_2d_enable_at)

    def forward(self, proj_joints, joints3d, targets, device=None):
        if device is None:
            device = torch.device('cpu')

        losses = {}
        # If training calculate 2D projection loss
        if self.training and proj_joints is not None:
            target_keypoints2d = torch.stack(
                [target.smplx_keypoints
                 for target in targets])
            target_conf = torch.stack(
                [target.conf for target in targets])

            if self.body_joints_2d_weight > 0:
                body_joints_2d_loss = (
                    self.body_joints_2d_weight * self.body_joints_2d_loss(
                        proj_joints[:, self.body_idxs],
                        target_keypoints2d[:, self.body_idxs],
                        weights=target_conf[:, self.body_idxs]))
                losses.update(body_joints_2d_loss=body_joints_2d_loss)

            if self.hand_joints_2d_active and self.hand_joints_2d_weight > 0:
                hand_joints_2d_loss = (
                    self.hand_joints_2d_weight * self.hand_joints_2d_loss(
                        proj_joints[:, self.hand_idxs],
                        target_keypoints2d[:, self.hand_idxs],
                        weights=target_conf[:, self.hand_idxs]))
                losses.update(hand_joints_2d_loss=hand_joints_2d_loss)

            if self.face_joints_2d_active and self.face_joints_2d_weight > 0:
                face_joints_2d_loss = (
                    self.face_joints_2d_weight * self.face_joints_2d_loss(
                        proj_joints[:, self.face_idxs],
                        target_keypoints2d[:, self.face_idxs],
                        weights=target_conf[:, self.face_idxs]))
                losses.update(face_joints_2d_loss=face_joints_2d_loss)

            if self.body_edge_2d_weight > 0 and self.body_edge_2d_active:
                body_edge_2d_loss = (
                    self.body_edge_2d_weight * self.body_edge_2d_loss(
                        proj_joints, target_keypoints2d, weights=target_conf))
                losses.update(body_edge_2d_loss=body_edge_2d_loss)

            if self.hand_edge_2d_weight > 0 and self.hand_edge_2d_active:
                hand_edge_2d_loss = (
                    self.hand_edge_2d_weight * self.hand_edge_2d_loss(
                        proj_joints, target_keypoints2d, weights=target_conf))
                losses.update(hand_edge_2d_loss=hand_edge_2d_loss)

            if self.face_edge_2d_weight > 0 and self.face_edge_2d_active:
                face_edge_2d_loss = (
                    self.face_edge_2d_weight * self.face_edge_2d_loss(
                        proj_joints, target_keypoints2d, weights=target_conf))
                losses.update(face_edge_2d_loss=face_edge_2d_loss)

        #  If training calculate 3D joints loss
        if (self.training and self.body_joints_3d_weight > 0 and
                joints3d is not None):
            # Get the indices of the targets that have 3D keypoint annotations
            target_idxs = []
            start_idx = 0
            for idx, target in enumerate(targets):
                # If there are no 3D annotations, skip and add to the starting
                # index the number of bounding boxes
                if len(target) < 1:
                    continue
                if not target.has_field('keypoints3d'):
                    start_idx += 1
                    continue
                #  keyp3d_field = target.get_field('keypoints3d')
                end_idx = start_idx + 1
                target_idxs += list(range(start_idx, end_idx))
                start_idx += 1

            # TODO: Add flag for procrustes alignment between keypoints
            if len(target_idxs) > 0:
                target_idxs = torch.tensor(np.asarray(target_idxs),
                                           device=device,
                                           dtype=torch.long)

                target_keypoints3d = torch.stack(
                    [target.get_field('keypoints3d').smplx_keypoints
                     for target in targets
                     if target.has_field('keypoints3d') and
                     len(target) > 0])
                target_conf = torch.stack(
                    [target.get_field('keypoints3d')['conf']
                     for target in targets
                     if target.has_field('keypoints3d') and
                     len(target) > 0])

                # Center the predictions using the pelvis
                pred_pelvis = joints3d[target_idxs][
                    :, [self.left_hip_idx, self.right_hip_idx], :].mean(
                    dim=1, keepdim=True)
                centered_pred_joints = joints3d[target_idxs] - pred_pelvis

                gt_pelvis = target_keypoints3d[
                    :, [self.left_hip_idx, self.right_hip_idx], :].mean(
                        dim=1, keepdim=True)
                centered_gt_joints = target_keypoints3d - gt_pelvis

                if self.body_joints_3d_weight > 0:
                    body_joints_3d_loss = (
                        self.body_joints_3d_weight * self.body_joints_3d_loss(
                            centered_pred_joints[:, self.body_idxs],
                            centered_gt_joints[:, self.body_idxs],
                            weights=target_conf[:, self.body_idxs]))
                    losses.update(body_joints_3d_loss=body_joints_3d_loss)

                if (self.hand_joints_3d_active and
                        self.hand_joints_3d_weight > 0):
                    hand_joints_3d_loss = (
                        self.hand_joints_3d_weight * self.hand_joints_3d_loss(
                            joints3d[target_idxs][:, self.hand_idxs],
                            target_keypoints3d[:, self.hand_idxs],
                            weights=target_conf[:, self.hand_idxs]))
                    losses.update(hand_joints_3d_loss=hand_joints_3d_loss)

                if (self.face_joints_3d_active and
                        self.face_joints_3d_weight > 0):
                    face_joints_3d_loss = (
                        self.face_joints_3d_weight * self.face_joints_3d_loss(
                            joints3d[target_idxs][:, self.face_idxs],
                            target_keypoints3d[:, self.face_idxs],
                            weights=target_conf[:, self.face_idxs]))
                    losses.update(face_joints_3d_loss=face_joints_3d_loss)

        return losses
