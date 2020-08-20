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
from .generic_target import GenericTarget

from expose.utils.rotation_utils import batch_rodrigues

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1

SIGN_FLIP = np.array([6, 7, 8, 3, 4, 5, 9, 10, 11, 15, 16, 17,
                      12, 13, 14, 18, 19, 20, 24, 25, 26, 21, 22, 23, 27,
                      28, 29, 33, 34, 35, 30, 31, 32,
                      36, 37, 38, 42, 43, 44, 39, 40, 41, 45, 46, 47, 51,
                      52, 53, 48, 49, 50, 57, 58, 59, 54, 55, 56, 63, 64,
                      65, 60, 61, 62],
                     dtype=np.int32) - 3


class HandPose(GenericTarget):
    """ Contains the hand pose parameters
    """

    def __init__(self, left_hand_pose, right_hand_pose, **kwargs):
        super(HandPose, self).__init__()
        self.left_hand_pose = left_hand_pose
        self.right_hand_pose = right_hand_pose

    def to_tensor(self, to_rot=True, *args, **kwargs):
        if not torch.is_tensor(self.left_hand_pose):
            if self.left_hand_pose is not None:
                self.left_hand_pose = torch.from_numpy(self.left_hand_pose)
        if not torch.is_tensor(self.right_hand_pose):
            if self.right_hand_pose is not None:
                self.right_hand_pose = torch.from_numpy(
                    self.right_hand_pose)
        if to_rot:
            if self.left_hand_pose is not None:
                self.left_hand_pose = batch_rodrigues(
                    self.left_hand_pose.view(-1, 3)).view(-1, 3, 3)
            if self.right_hand_pose is not None:
                self.right_hand_pose = batch_rodrigues(
                    self.right_hand_pose.view(-1, 3)).view(-1, 3, 3)

        for k, v in self.extra_fields.items():
            if isinstance(v, GenericTarget):
                v.to_tensor(*args, **kwargs)

    def transpose(self, method):

        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        if torch.is_tensor(self.left_hand_pose):
            dim_flip = torch.tensor(
                [1, -1, -1], dtype=self.left_hand_pose.dtype)
        else:
            dim_flip = np.array([1, -1, -1], dtype=self.left_hand_pose.dtype)

        left_hand_pose = (self.right_hand_pose.reshape(15, 3) *
                          dim_flip).reshape(45)
        right_hand_pose = (self.left_hand_pose.reshape(15, 3) *
                           dim_flip).reshape(45)
        field = type(self)(left_hand_pose=left_hand_pose,
                           right_hand_pose=right_hand_pose)

        for k, v in self.extra_fields.items():
            if isinstance(v, GenericTarget):
                v = v.transpose(method)
            field.add_field(k, v)
        self.add_field('is_flipped', True)
        return field

    def to(self, *args, **kwargs):
        left_hand_pose = self.left_hand_pose
        right_hand_pose = self.right_hand_pose
        if left_hand_pose is not None:
            left_hand_pose = left_hand_pose.to(*args, **kwargs)
        if right_hand_pose is not None:
            right_hand_pose = right_hand_pose.to(*args, **kwargs)
        field = type(self)(
            left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            field.add_field(k, v)
        return field
