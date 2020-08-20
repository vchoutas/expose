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
from typing import Union

import torch

from .typing_utils import Tensor, Array


def no_reduction(arg):
    return arg


def to_tensor(
        tensor: Union[Tensor, Array],
        device=None,
        dtype=torch.float32
) -> Tensor:
    if torch.is_tensor(tensor):
        return tensor
    else:
        return torch.tensor(tensor, dtype=dtype, device=device)


def get_reduction_method(reduction='mean'):
    if reduction == 'mean':
        reduction = torch.mean
    elif reduction == 'sum':
        reduction = torch.sum
    elif reduction == 'none':
        reduction = no_reduction
    else:
        raise ValueError('Unknown reduction type: {}'.format(reduction))
    return reduction


def tensor_to_numpy(tensor: Tensor, default=None) -> Array:
    if tensor is None:
        return default
    else:
        return tensor.detach().cpu().numpy()


def rot_mat_to_euler(rot_mats: Tensor) -> Tensor:
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)
