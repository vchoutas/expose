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


class GlobalPose(GenericTarget):

    def __init__(self, global_pose, **kwargs):
        super(GlobalPose, self).__init__()
        self.global_pose = global_pose

    def to_tensor(self, to_rot=True, *args, **kwargs):
        if not torch.is_tensor(self.global_pose):
            self.global_pose = torch.from_numpy(self.global_pose)

        if to_rot:
            self.global_pose = batch_rodrigues(
                self.global_pose.view(-1, 3)).view(1, 3, 3)

        for k, v in self.extra_fields.items():
            if isinstance(v, GenericTarget):
                v.to_tensor(*args, **kwargs)

    def transpose(self, method):

        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        if torch.is_tensor(self.global_pose):
            dim_flip = torch.tensor([1, -1, -1], dtype=self.global_pose.dtype)
            global_pose = self.global_pose.clone().squeeze() * dim_flip
        else:
            dim_flip = np.array([1, -1, -1], dtype=self.global_pose.dtype)
            global_pose = self.global_pose.copy().squeeze() * dim_flip

        field = type(self)(global_pose=global_pose)

        for k, v in self.extra_fields.items():
            if isinstance(v, GenericTarget):
                v = v.transpose(method)
            field.add_field(k, v)
        self.add_field('is_flipped', True)
        return field

    def rotate(self, rot=0, *args, **kwargs):
        global_pose = self.global_pose.copy()
        if rot != 0:
            R = np.array([[np.cos(np.deg2rad(-rot)),
                           -np.sin(np.deg2rad(-rot)), 0],
                          [np.sin(np.deg2rad(-rot)),
                           np.cos(np.deg2rad(-rot)), 0],
                          [0, 0, 1]], dtype=np.float32)

            # find the rotation of the body in camera frame
            per_rdg, _ = cv2.Rodrigues(global_pose)
            # apply the global rotation to the global orientation
            resrot, _ = cv2.Rodrigues(np.dot(R, per_rdg))
            global_pose = (resrot.T)[0].reshape(3)
        field = type(self)(global_pose=global_pose)
        for k, v in self.extra_fields.items():
            if isinstance(v, GenericTarget):
                v = v.crop(rot=rot, *args, **kwargs)
            field.add_field(k, v)

        self.add_field('rot', rot)
        return field

    def to(self, *args, **kwargs):
        field = type(self)(global_pose=self.global_pose.to(*args, **kwargs))
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            field.add_field(k, v)
        return field
