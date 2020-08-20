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
import cv2

from loguru import logger
from .generic_target import GenericTarget
from expose.utils.rotation_utils import batch_rodrigues

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1

sign_flip = np.array(
    [1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1,
        -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1,
        -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1,
        1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1,
        -1, 1, -1, -1])

SIGN_FLIP = torch.tensor([6, 7, 8, 3, 4, 5, 9, 10, 11, 15, 16, 17,
                          12, 13, 14, 18, 19, 20, 24, 25, 26, 21, 22, 23, 27,
                          28, 29, 33, 34, 35, 30, 31, 32,
                          36, 37, 38, 42, 43, 44, 39, 40, 41, 45, 46, 47, 51,
                          52, 53, 48, 49, 50, 57, 58, 59, 54, 55, 56, 63, 64,
                          65, 60, 61, 62],
                         dtype=torch.long) - 3


class JawPose(GenericTarget):
    """ Contains the jaw pose parameters
    """

    def __init__(self, jaw_pose, dtype=torch.float32, **kwargs):
        super(JawPose, self).__init__()
        self.jaw_pose = jaw_pose

    def to_tensor(self, to_rot=True, *args, **kwargs):
        if not torch.is_tensor(self.jaw_pose):
            self.jaw_pose = torch.from_numpy(self.jaw_pose)

        if to_rot:
            self.jaw_pose = batch_rodrigues(
                self.jaw_pose.view(-1, 3)).view(-1, 3, 3)

        for k, v in self.extra_fields.items():
            if isinstance(v, GenericTarget):
                v.to_tensor(*args, **kwargs)

    def transpose(self, method):

        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        dim_flip = np.array([1, -1, -1], dtype=self.jaw_pose.dtype)
        jaw_pose = self.jaw_pose.copy() * dim_flip

        field = type(self)(jaw_pose=jaw_pose)
        for k, v in self.extra_fields.items():
            if isinstance(v, GenericTarget):
                v = v.transpose(method)
            field.add_field(k, v)
        self.add_field('is_flipped', True)
        return field

    def to(self, *args, **kwargs):
        field = type(self)(jaw_pose=self.jaw_pose.to(*args, **kwargs))
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            field.add_field(k, v)
        return field
