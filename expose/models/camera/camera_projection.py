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
from dataclasses import dataclass

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger
from expose.utils.typing_utils import Tensor, Array

DEFAULT_FOCAL_LENGTH = 5000


@dataclass
class CameraParams:
    translation: Tensor = None
    rotation: Tensor = None
    scale: Tensor = None
    focal_length: Tensor = None

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __iter__(self):
        return self.keys()

    def keys(self):
        keys = [t.name for t in fields(self)]
        return iter(keys)

    def values(self):
        values = [getattr(self, t.name) for t in fields(self)]
        return iter(values)

    def items(self):
        data = [(t.name, getattr(self, t.name)) for t in fields(self)]
        return iter(data)


def build_cam_proj(camera_cfg, dtype=torch.float32):
    camera_type = camera_cfg.get('type', 'weak-persp')
    camera_pos_scale = camera_cfg.get('pos_func')
    if camera_pos_scale == 'softplus':
        camera_scale_func = F.softplus
    elif camera_pos_scale == 'exp':
        camera_scale_func = torch.exp
    elif camera_pos_scale == 'none' or camera_pos_scale == 'None':
        def func(x):
            return x
        camera_scale_func = func
    else:
        raise ValueError(
            f'Unknown positive scaling function: {camera_pos_scale}')

    if camera_type.lower() == 'persp':
        if camera_pos_scale == 'softplus':
            mean_flength = np.log(np.exp(DEFAULT_FOCAL_LENGTH) - 1)
        elif camera_pos_scale == 'exp':
            mean_flength = np.log(DEFAULT_FOCAL_LENGTH)
        elif camera_pos_scale == 'none':
            mean_flength = DEFAULT_FOCAL_LENGTH
        camera = PerspectiveCamera(dtype=dtype)
        camera_mean = torch.tensor(
            [mean_flength, 0.0, 0.0], dtype=torch.float32)
        camera_param_dim = 4
    elif camera_type.lower() == 'weak-persp':
        weak_persp_cfg = camera_cfg.get('weak_persp', {})
        mean_scale = weak_persp_cfg.get('mean_scale', 0.9)
        if camera_pos_scale == 'softplus':
            mean_scale = np.log(np.exp(mean_scale) - 1)
        elif camera_pos_scale == 'exp':
            mean_scale = np.log(mean_scale)
        camera_mean = torch.tensor([mean_scale, 0.0, 0.0], dtype=torch.float32)
        camera = WeakPerspectiveCamera(dtype=dtype)
        camera_param_dim = 3
    else:
        raise ValueError(f'Unknown camera type: {camera_type}')

    return {
        'camera': camera,
        'mean': camera_mean,
        'scale_func': camera_scale_func,
        'dim': camera_param_dim
    }


class PerspectiveCamera(nn.Module):
    ''' Module that implements a perspective camera
    '''

    FOCAL_LENGTH = DEFAULT_FOCAL_LENGTH

    def __init__(self, dtype=torch.float32, focal_length=None, **kwargs):
        super(PerspectiveCamera, self).__init__()
        self.dtype = dtype

        if focal_length is None:
            focal_length = self.FOCAL_LENGTH
        # Make a buffer so that PyTorch does not complain when creating
        # the camera matrix
        self.register_buffer(
            'focal_length', torch.tensor(focal_length, dtype=dtype))

    def forward(
        self,
        points: Tensor,
        focal_length: Tensor = None,
        translation: Tensor = None,
        rotation: Tensor = None,
        camera_center: Tensor = None,
        **kwargs
    ) -> Tensor:
        ''' Forward pass for the perspective camera

            Parameters
            ----------
                points: torch.tensor, BxNx3
                    The tensor that contains the points that will be projected.
                    If not in homogeneous coordinates, then
                focal_length: torch.tensor, BxNx3, optional
                    The predicted focal length of the camera. If not given,
                    then the default value of 5000 is assigned
                translation: torch.tensor, Bx3, optional
                    The translation predicted for each element in the batch. If
                    not given  then a zero translation vector is assumed
                rotation: torch.tensor, Bx3x3, optional
                    The rotation predicted for each element in the batch. If
                    not given  then an identity rotation matrix is assumed
                camera_center: torch.tensor, Bx2, optional
                    The center of each image for the projection. If not given,
                    then a zero vector is used
            Returns
            -------
                Returns a torch.tensor object with size BxNx2 with the
                location of the projected points on the image plane
        '''

        device = points.device
        batch_size = points.shape[0]

        if rotation is None:
            rotation = torch.eye(
                3, dtype=points.dtype, device=device).unsqueeze(dim=0).expand(
                    batch_size, -1, -1)
        if translation is None:
            translation = torch.zeros(
                [3], dtype=points.dtype,
                device=device).unsqueeze(dim=0).expand(batch_size, -11)

        if camera_center is None:
            camera_center = torch.zeros([batch_size, 2], dtype=points.dtype,
                                        device=device)

        with torch.no_grad():
            camera_mat = torch.zeros([batch_size, 2, 2],
                                     dtype=self.dtype, device=points.device)
            if focal_length is None:
                focal_length = self.focal_length

            camera_mat[:, 0, 0] = focal_length
            camera_mat[:, 1, 1] = focal_length

        points_transf = torch.einsum(
            'bji,bmi->bmj',
            rotation, points) + translation.unsqueeze(dim=1)

        img_points = torch.div(points_transf[:, :, :2],
                               points_transf[:, :, 2].unsqueeze(dim=-1))
        img_points = torch.einsum(
            'bmi,bji->bjm',
            camera_mat, img_points) + camera_center.reshape(-1, 1, 2)
        return img_points


class WeakPerspectiveCamera(nn.Module):
    ''' Scaled Orthographic / Weak-Perspective Camera
    '''

    def __init__(self, **kwargs):
        super(WeakPerspectiveCamera, self).__init__()

    def forward(
        self,
        points: Tensor,
        scale: Tensor,
        translation: Tensor,
        **kwargs
    ) -> Tensor:
        ''' Implements the forward pass for a Scaled Orthographic Camera

            Parameters
            ----------
                points: torch.tensor, BxNx3
                    The tensor that contains the points that will be projected.
                    If not in homogeneous coordinates, then
                scale: torch.tensor, Bx1
                    The predicted scaling parameters
                translation: torch.tensor, Bx2
                    The translation applied on the image plane to the points
            Returns
            -------
                projected_points: torch.tensor, BxNx2
                    The points projected on the image plane, according to the
                    given scale and translation
        '''
        assert translation.shape[-1] == 2, 'Translation shape must be -1x2'
        assert scale.shape[-1] == 1, 'Scale shape must be -1x1'

        projected_points = scale.view(-1, 1, 1) * (
            points[:, :, :2] + translation.view(-1, 1, 2))
        return projected_points
