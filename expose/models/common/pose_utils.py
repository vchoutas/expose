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

from typing import Dict, Tuple

import sys
import os.path as osp
import pickle

import math
import numpy as np

from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from expose.utils.rotation_utils import batch_rodrigues, batch_rot2aa

from .networks import MLP


class PoseParameterization(object):
    KEYS = ['regressor', 'decoder', 'dim', 'mean', 'ind_dim']

    def __init__(self, regressor=None, decoder=None, dim=0, ind_dim=0,
                 mean=None):
        super(PoseParameterization, self).__init__()

        self.regressor = regressor
        self.decoder = decoder
        self.dim = dim
        self.mean = mean
        self.ind_dim = ind_dim

    def keys(self):
        return [key for key in self.KEYS
                if getattr(self, key) is not None]

    def __getitem__(self, key):
        return getattr(self, key)


def build_pose_regressor(input_dim: int,
                         num_angles: int,
                         pose_cfg: Dict,
                         network_cfg: Dict,
                         mean_pose: np.array = None,
                         pca_basis: np.array = None,
                         append_params=True) -> Tuple[nn.Module, nn.Module]:
    pose_decoder = build_pose_decoder(
        pose_cfg, num_angles, mean_pose=mean_pose,
        pca_basis=pca_basis)

    pose_dim_size = pose_decoder.get_dim_size()
    reg_input_dim = input_dim + append_params * pose_dim_size

    regressor = MLP(reg_input_dim, pose_dim_size, **network_cfg)

    return pose_decoder, regressor


def create_pose_parameterization(input_dim, num_angles, param_type='aa',
                                 num_pca_comps=12,
                                 latent_dim_size=32,
                                 append_params=True,
                                 create_regressor=True,
                                 **kwargs):

    logger.debug('Creating {} for {} joints', param_type, num_angles)

    regressor = None

    if param_type == 'aa':
        input_dim += append_params * num_angles * 3
        if create_regressor:
            regressor = MLP(input_dim, num_angles * 3, **kwargs)
        decoder = AADecoder(num_angles=num_angles, **kwargs)
        dim = decoder.get_dim_size()
        ind_dim = 3
        mean = decoder.get_mean()
    elif param_type == 'pca':
        input_dim += append_params * num_pca_comps
        if create_regressor:
            regressor = MLP(input_dim, num_pca_comps, **kwargs)
        decoder = PCADecoder(num_pca_comps=num_pca_comps, **kwargs)
        ind_dim = num_pca_comps
        dim = decoder.get_dim_size()
        mean = decoder.get_mean()
    elif param_type == 'cont_rot_repr':
        input_dim += append_params * num_angles * 6
        if create_regressor:
            regressor = MLP(input_dim, num_angles * 6, **kwargs)
        decoder = ContinuousRotReprDecoder(num_angles, **kwargs)
        dim = decoder.get_dim_size()
        ind_dim = 6
        mean = decoder.get_mean()
    elif param_type == 'rot_mats':
        input_dim += append_params * num_angles * 9
        if create_regressor:
            regressor = MLP(input_dim, num_angles * 9, **kwargs)
        decoder = SVDRotationProjection()
        dim = decoder.get_dim_size()
        mean = decoder.get_mean()
        ind_dim = 9
    else:
        raise ValueError(f'Unknown pose parameterization: {param_type}')

    return PoseParameterization(regressor=regressor,
                                decoder=decoder,
                                dim=dim,
                                ind_dim=ind_dim,
                                mean=mean)


def build_pose_decoder(cfg, num_angles, mean_pose=None, pca_basis=None):
    param_type = cfg.get('param_type', 'aa')
    logger.debug('Creating {} for {} joints', param_type, num_angles)
    if param_type == 'aa':
        decoder = AADecoder(num_angles=num_angles, mean=mean_pose, **cfg)
    elif param_type == 'pca':
        decoder = PCADecoder(pca_basis=pca_basis, mean=mean_pose, **cfg)
    elif param_type == 'cont_rot_repr':
        decoder = ContinuousRotReprDecoder(num_angles, mean=mean_pose, **cfg)
    elif param_type == 'rot_mats':
        decoder = SVDRotationProjection()
    else:
        raise ValueError(f'Unknown pose decoder: {param_type}')
    return decoder


def build_all_pose_params(body_model_cfg,
                          feat_extract_depth,
                          body_model,
                          append_params=True,
                          dtype=torch.float32):
    mean_pose_path = osp.expandvars(body_model_cfg.mean_pose_path)
    mean_poses_dict = {}
    if osp.exists(mean_pose_path):
        logger.debug('Loading mean pose from: {} ', mean_pose_path)
        with open(mean_pose_path, 'rb') as f:
            mean_poses_dict = pickle.load(f)

    global_orient_desc = create_pose_parameterization(
        feat_extract_depth, 1, dtype=dtype,
        append_params=append_params,
        create_regressor=False, **body_model_cfg.global_orient)

    global_orient_type = body_model_cfg.get('global_orient', {}).get(
        'param_type', 'cont_rot_repr')
    logger.debug('Global pose parameterization, decoder: {}, {}',
                 global_orient_type, global_orient_desc.decoder)
    # Rotate the model 180 degrees around the x-axis
    if global_orient_type == 'aa':
        global_orient_desc.decoder.mean[0] = math.pi
    elif global_orient_type == 'cont_rot_repr':
        global_orient_desc.decoder.mean[3] = -1

    body_pose_desc = create_pose_parameterization(
        feat_extract_depth, num_angles=body_model.NUM_BODY_JOINTS,
        ignore_hands=True, dtype=dtype,
        append_params=append_params, create_regressor=False,
        mean=mean_poses_dict.get('body_pose', None),
        **body_model_cfg.body_pose)
    logger.debug('Body pose decoder: {}', body_pose_desc.decoder)

    left_hand_cfg = body_model_cfg.left_hand_pose
    right_hand_cfg = body_model_cfg.right_hand_pose
    left_hand_pose_desc = create_pose_parameterization(
        feat_extract_depth, num_angles=15, dtype=dtype,
        append_params=append_params,
        pca_basis=body_model.left_hand_components,
        mean=mean_poses_dict.get('left_hand_pose', None),
        create_regressor=False, **left_hand_cfg)
    logger.debug('Left hand pose decoder: {}', left_hand_pose_desc.decoder)

    right_hand_pose_desc = create_pose_parameterization(
        feat_extract_depth, num_angles=15, dtype=dtype,
        append_params=append_params,
        mean=mean_poses_dict.get('right_hand_pose', None),
        pca_basis=body_model.right_hand_components,
        create_regressor=False, **right_hand_cfg)
    logger.debug('Right hand pose decoder: {}', right_hand_pose_desc.decoder)

    jaw_pose_desc = create_pose_parameterization(
        feat_extract_depth, 1, dtype=dtype,
        append_params=append_params,
        create_regressor=False, **body_model_cfg.jaw_pose)

    logger.debug('Jaw pose decoder: {}', jaw_pose_desc.decoder)

    return {
        'global_orient': global_orient_desc,
        'body_pose': body_pose_desc,
        'left_hand_pose': left_hand_pose_desc,
        'right_hand_pose': right_hand_pose_desc,
        'jaw_pose': jaw_pose_desc,
    }


class RotationMatrixRegressor(nn.Linear):

    def __init__(self, input_dim, num_angles, dtype=torch.float32,
                 append_params=True, **kwargs):
        super(RotationMatrixRegressor, self).__init__(
            input_dim + append_params * num_angles * 3,
            num_angles * 9)
        self.num_angles = num_angles
        self.dtype = dtype
        self.svd_projector = SVDRotationProjection()

    def get_param_dim(self):
        return 9

    def get_dim_size(self):
        return self.num_angles * 9

    def get_mean(self):
        return torch.eye(3, dtype=self.dtype).unsqueeze(dim=0).expand(
            self.num_angles, -1, -1)

    def forward(self, module_input):
        rot_mats = super(RotationMatrixRegressor, self).forward(
            module_input).view(-1, 3, 3)

        # Project the matrices on the manifold of rotation matrices using SVD
        rot_mats = self.svd_projector(rot_mats).view(
            -1, self.num_angles, 3, 3)

        return rot_mats


class ContinuousRotReprDecoder(nn.Module):
    ''' Decoder for transforming a latent representation to rotation matrices

        Implements the decoding method described in:
        "On the Continuity of Rotation Representations in Neural Networks"
    '''

    def __init__(self, num_angles, dtype=torch.float32, mean=None,
                 **kwargs):
        super(ContinuousRotReprDecoder, self).__init__()
        self.num_angles = num_angles
        self.dtype = dtype

        if isinstance(mean, dict):
            mean = mean.get('cont_rot_repr', None)
        if mean is None:
            mean = torch.tensor(
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                dtype=self.dtype).unsqueeze(dim=0).expand(
                    self.num_angles, -1).contiguous().view(-1)

        if not torch.is_tensor(mean):
            mean = torch.tensor(mean)
        mean = mean.reshape(-1, 6)

        if mean.shape[0] < self.num_angles:
            logger.debug(mean.shape)
            mean = mean.repeat(
                self.num_angles // mean.shape[0] + 1, 1).contiguous()
            mean = mean[:self.num_angles]
        elif mean.shape[0] > self.num_angles:
            mean = mean[:self.num_angles]

        mean = mean.reshape(-1)
        self.register_buffer('mean', mean)

    def get_type(self):
        return 'cont_rot_repr'

    def extra_repr(self):
        msg = 'Num angles: {}\n'.format(self.num_angles)
        msg += 'Mean: {}'.format(self.mean.shape)
        return msg

    def get_param_dim(self):
        return 6

    def get_dim_size(self):
        return self.num_angles * 6

    def get_mean(self):
        return self.mean.clone()

    def to_offsets(self, x):
        latent = x.reshape(-1, 3, 3)[:, :3, :2].reshape(-1, 6)
        return (latent - self.mean).reshape(x.shape[0], -1, 6)

    def encode(self, x, subtract_mean=False):
        orig_shape = x.shape
        if subtract_mean:
            raise NotImplementedError
        output = x.reshape(-1, 3, 3)[:, :3, :2].contiguous()
        return output.reshape(
            orig_shape[0], orig_shape[1], 3, 2)

    def forward(self, module_input):
        batch_size = module_input.shape[0]
        reshaped_input = module_input.view(-1, 3, 2)

        # Normalize the first vector
        b1 = F.normalize(reshaped_input[:, :, 0].clone(), dim=1)

        dot_prod = torch.sum(
            b1 * reshaped_input[:, :, 1].clone(), dim=1, keepdim=True)
        # Compute the second vector by finding the orthogonal complement to it
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=1)
        # Finish building the basis by taking the cross product
        b3 = torch.cross(b1, b2, dim=1)
        rot_mats = torch.stack([b1, b2, b3], dim=-1)

        return rot_mats.view(batch_size, -1, 3, 3)


class ContinuousRotReprRegressor(nn.Linear):
    def __init__(self, input_dim, num_angles, dtype=torch.float32,
                 append_params=True, **kwargs):
        super(ContinuousRotReprRegressor, self).__init__(
            input_dim + append_params * num_angles * 6, num_angles * 6)
        self.append_params = append_params
        self.num_angles = num_angles
        self.repr_decoder = ContinuousRotReprDecoder(num_angles)

    def get_dim_size(self):
        return self.num_angles * 9

    def get_mean(self):
        if self.to_aa:
            return torch.zeros([1, self.num_angles * 3], dtype=self.dtype)
        else:
            return torch.zeros([1, self.num_angles, 3, 3], dtype=self.dtype)

    def forward(self, module_input, prev_val):
        if self.append_params:
            if self.to_aa:
                prev_val = batch_rodrigues(prev_val)
            prev_val = prev_val[:, :, :2].contiguous().view(
                -1, self.num_angles * 6)

            module_input = torch.cat([module_input, prev_val], dim=-1)

        cont_repr = super(ContinuousRotReprRegressor,
                          self).forward(module_input)

        output = self.repr_decoder(cont_repr).view(-1, self.num_angles, 3, 3)
        return output


class SVDRotationProjection(nn.Module):
    def __init__(self, **kwargs):
        super(SVDRotationProjection, self).__init__()

    def forward(self, module_input):
        # Before converting the output rotation matrices of the VAE to
        # axis-angle representation, we first need to make them in to valid
        # rotation matrices
        with torch.no_grad():
            # TODO: Replace with Batch SVD once merged
            # Iterate over the batch dimension and compute the SVD
            svd_input = module_input.detach().cpu()
            #  svd_input = output
            norm_rotation = torch.zeros_like(svd_input)
            for bidx in range(module_input.shape[0]):
                U, _, V = torch.svd(svd_input[bidx])

                # Multiply the U, V matrices to get the closest orthonormal
                # matrix
                norm_rotation[bidx] = torch.matmul(U, V.t())
            norm_rotation = norm_rotation.to(module_input.device)

        # torch.svd supports backprop only for full-rank matrices.
        # The output is calculated as the valid rotation matrix plus the
        # output minus the detached output. If one writes down the
        # computational graph for this operation, it will become clear the
        # output is the desired valid rotation matrix, while for the
        # backward pass gradients are propagated only to the original
        # matrix
        # Source: PyTorch Gumbel-Softmax hard sampling
        # https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax
        correct_rot = norm_rotation - module_input.detach() + module_input
        return correct_rot


class AARegressor(nn.Linear):
    def __init__(self, input_dim, num_angles, dtype=torch.float32,
                 append_params=True, to_aa=True, **kwargs):
        super(AARegressor, self).__init__(
            input_dim + append_params * num_angles * 3, num_angles * 3)
        self.num_angles = num_angles
        self.to_aa = to_aa
        self.dtype = dtype

    def get_param_dim(self):
        return 3

    def get_dim_size(self):
        return self.num_angles * 3

    def get_mean(self):
        return torch.zeros([self.num_angles * 3], dtype=self.dtype)

    def forward(self, features):
        aa_vectors = super(AARegressor, self).forward(features).view(
            -1, self.num_angles, 3)

        return batch_rodrigues(aa_vectors.view(-1, 3)).view(
            -1, self.num_angles, 3, 3)


class AADecoder(nn.Module):
    def __init__(self, num_angles, dtype=torch.float32, mean=None, **kwargs):
        super(AADecoder, self).__init__()
        self.num_angles = num_angles
        self.dtype = dtype

        if isinstance(mean, dict):
            mean = mean.get('aa', None)
        if mean is None:
            mean = torch.zeros([num_angles * 3], dtype=dtype)

        if not torch.is_tensor(mean):
            mean = torch.tensor(mean, dtype=dtype)
        mean = mean.reshape(-1)
        self.register_buffer('mean', mean)

    def get_dim_size(self):
        return self.num_angles * 3

    def get_mean(self):
        return torch.zeros([self.get_dim_size()], dtype=self.dtype)

    def forward(self, module_input):
        output = batch_rodrigues(module_input.view(-1, 3)).view(
            -1, self.num_angles, 3, 3)
        return output


class PCADecoder(nn.Module):
    def __init__(self, num_pca_comps=12, pca_basis=None, dtype=torch.float32,
                 mean=None,
                 **kwargs):
        super(PCADecoder, self).__init__()
        self.num_pca_comps = num_pca_comps
        self.dtype = dtype
        pca_basis_tensor = torch.tensor(pca_basis, dtype=dtype)
        self.register_buffer('pca_basis',
                             pca_basis_tensor[:self.num_pca_comps])
        inv_basis = torch.inverse(
            pca_basis_tensor.t()).unsqueeze(dim=0)
        self.register_buffer('inv_pca_basis', inv_basis)

        if isinstance(mean, dict):
            mean = mean.get('aa', None)

        if mean is None:
            mean = torch.zeros([45], dtype=dtype)

        if not torch.is_tensor(mean):
            mean = torch.tensor(mean, dtype=dtype)
        mean = mean.reshape(-1).reshape(1, -1)
        self.register_buffer('mean', mean)

    def get_param_dim(self):
        return self.num_pca_comps

    def extra_repr(self):
        msg = 'PCA Components = {}'.format(self.num_pca_comps)
        return msg

    def get_mean(self):
        return self.mean.clone()

    def get_dim_size(self):
        return self.num_pca_comps

    def to_offsets(self, x):
        batch_size = x.shape[0]
        # Convert the rotation matrices to axis angle
        aa = batch_rot2aa(x.reshape(-1, 3, 3)).reshape(batch_size, 45, 1)

        # Project to the PCA space
        offsets = torch.matmul(
            self.inv_pca_basis, aa
        ).reshape(batch_size, -1)[:, :self.num_pca_comps]

        return offsets - self.mean

    def encode(self, x, subtract_mean=False):
        batch_size = x.shape[0]
        # Convert the rotation matrices to axis angle
        aa = batch_rot2aa(x.reshape(-1, 3, 3)).reshape(batch_size, 45, 1)

        # Project to the PCA space
        output = torch.matmul(
            self.inv_pca_basis, aa
        ).reshape(batch_size, -1)[:, :self.num_pca_comps]
        if subtract_mean:
            # Remove the mean offset
            output -= self.mean

        return output

    def forward(self, pca_coeffs):
        batch_size = pca_coeffs.shape[0]
        decoded = torch.einsum(
            'bi,ij->bj', [pca_coeffs, self.pca_basis]) + self.mean

        return batch_rodrigues(decoded.view(-1, 3)).view(
            batch_size, -1, 3, 3)
