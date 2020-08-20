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

import numpy as np
import torch

DIM_FLIP = np.array([1, -1, -1], dtype=np.float32)
DIM_FLIP_TENSOR = torch.tensor([1, -1, -1], dtype=torch.float32)


def flip_pose(pose_vector, pose_format='aa'):
    if pose_format == 'aa':
        if torch.is_tensor(pose_vector):
            dim_flip = DIM_FLIP_TENSOR
        else:
            dim_flip = DIM_FLIP
        return (pose_vector.reshape(-1, 3) * dim_flip).reshape(-1)
    elif pose_format == 'rot-mat':
        rot_mats = pose_vector.reshape(-1, 9).clone()

        rot_mats[:, [1, 2, 3, 6]] *= -1
        return rot_mats.view_as(pose_vector)
    else:
        raise ValueError(f'Unknown rotation format: {pose_format}')
