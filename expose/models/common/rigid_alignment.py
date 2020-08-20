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

from typing import NewType

import torch
import torch.nn as nn

from loguru import logger


Tensor = NewType('Tensor', torch.tensor)


class RotationTranslationAlignment(nn.Module):
    def __init__(self) -> None:
        ''' Implements rotation and translation alignment with least squares

            For more information see:

            Least-Squares Rigid Motion Using SVD
            Olga Sorkine-Hornung and Michael Rabinovich

        '''
        super(RotationTranslationAlignment, self).__init__()

    def forward(
            self,
            p: Tensor,
            q: Tensor) -> Tensor:
        ''' Aligns two point clouds using the optimal R, T

            Parameters
            ----------
                p: BxNx3, torch.Tensor
                    The first of points
                q: BxNx3, torch.Tensor

            Returns
            -------
                p_hat: BxNx3, torch.Tensor
                    The points p after least squares alignment to q
        '''
        batch_size = p.shape[0]
        dtype = p.dtype
        device = p.device

        p_transpose = p.transpose(1, 2)
        q_transpose = q.transpose(1, 2)

        # 1. Remove mean.
        p_mean = torch.mean(p_transpose, dim=-1, keepdim=True)
        q_mean = torch.mean(q_transpose, dim=-1, keepdim=True)

        p_centered = p_transpose - p_mean
        q_centered = q_transpose - q_mean

        # 2. Compute variance of X1 used for scale.
        var_p = torch.sum(p_centered.pow(2), dim=(1, 2), keepdim=True)
        #  var_q = torch.sum(q_centered.pow(2), dim=(1, 2), keepdim=True)

        # Compute the outer product of the two point sets
        # Should be Bx3x3
        K = torch.bmm(p_centered, q_centered.transpose(1, 2))
        # Apply SVD on the outer product matrix to recover the rotation
        U, S, V = torch.svd(K)

        # Make sure that the computed rotation does not contain a reflection
        Z = torch.eye(3, dtype=dtype, device=device).view(
            1, 3, 3).expand(batch_size, -1, -1).contiguous()

        raw_product = torch.bmm(U, V.transpose(1, 2))
        Z[:, -1, -1] *= torch.sign(torch.det(raw_product))

        # Compute the final rotation matrix
        rotation = torch.bmm(V, torch.bmm(Z, U.transpose(1, 2)))

        scale = torch.einsum('bii->b', [torch.bmm(rotation, K)]) / var_p.view(
            -1)

        # Compute the translation vector
        translation = q_mean - scale.reshape(batch_size, 1, 1) * torch.bmm(
            rotation, p_mean)

        return (
            scale.reshape(batch_size, 1, 1) *
            torch.bmm(rotation, p_transpose) +
            translation).transpose(1, 2)
