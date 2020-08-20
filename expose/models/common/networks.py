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

from typing import Optional, Tuple, List
import sys

import math

import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger

from ..nnutils import init_weights
from expose.utils.typing_utils import Tensor


def create_activation(activ_type='relu', lrelu_slope=0.2,
                      inplace=True, **kwargs):
    if activ_type == 'relu':
        return nn.ReLU(inplace=inplace)
    elif activ_type == 'leaky-relu':
        return nn.LeakyReLU(negative_slope=lrelu_slope, inplace=inplace)
    elif activ_type == 'none':
        return None
    else:
        raise ValueError(f'Unknown activation type: {activ_type}')


def create_norm_layer(input_dim, norm_type='none', num_groups=32, dim=1,
                      **kwargs):
    if norm_type == 'bn':
        if dim == 1:
            return nn.BatchNorm1d(input_dim)
        elif dim == 2:
            return nn.BatchNorm2d(input_dim)
        else:
            raise ValueError(f'Wrong dimension for BN: {dim}')
    if norm_type == 'ln':
        return nn.LayerNorm(input_dim)
    elif norm_type == 'gn':
        return nn.GroupNorm(num_groups, input_dim)
    elif norm_type.lower() == 'none':
        return None
    else:
        raise ValueError(f'Unknown normalization type: {norm_type}')


def create_adapt_pooling(name='avg', dim='2d', ksize=1):
    if dim == '2d':
        if name == 'avg':
            return nn.AdaptiveAvgPool2d(ksize)
        elif name == 'max':
            return nn.AdaptiveMaxPool2d(ksize)
        else:
            raise ValueError(f'Unknown pooling type: {name}')
    else:
        raise ValueError('Unknown pooling dimensionality: {dim}')


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    @staticmethod
    def from_bn(module: nn.BatchNorm2d):
        ''' Initializes a frozen batch norm module from a batch norm module
        '''
        dim = len(module.weight.data)

        frozen_module = FrozenBatchNorm2d(dim)
        frozen_module.weight.data = module.weight.data

        missing, not_found = frozen_module.load_state_dict(
            module.state_dict(), strict=False)
        return frozen_module

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """
        Convert BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.

        Args:
            module (torch.nn.Module):

        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.

        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res

    def forward(self, x):
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        return F.batch_norm(
            x, self.running_mean, self.running_var, self.weight, self.bias,
            False)


class ConvNormActiv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=1,
                 activation='relu',
                 norm_type='bn',
                 padding=0,
                 **kwargs):
        super(ConvNormActiv, self).__init__()
        layers = []

        norm_layer = create_norm_layer(output_dim, norm_type,
                                       dim=2,
                                       **kwargs)
        bias = norm_layer is None

        layers.append(
            nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size,
                      padding=padding,
                      bias=bias))
        if norm_layer is not None:
            layers.append(norm_layer)

        activ = create_activation(**kwargs)
        if activ is not None:
            layers.append(activ)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layers: Optional[List[int]] = None,
        activation: str = 'relu',
        norm_type: str = 'bn',
        dropout: float = 0.0,
        gain: float = 0.01,
        preactivated: bool = False,
        flatten: bool = True,
        **kwargs
    ):
        ''' Simple MLP module
        '''
        super(MLP, self).__init__()
        if layers is None:
            layers = []
        self.flatten = flatten

        curr_input_dim = input_dim
        self.num_layers = len(layers)

        self.blocks = []
        for layer_idx, layer_dim in enumerate(layers):
            activ = create_activation(**kwargs)
            norm_layer = create_norm_layer(layer_dim, norm_type, **kwargs)
            bias = norm_layer is None

            linear = nn.Linear(curr_input_dim, layer_dim, bias=bias)
            curr_input_dim = layer_dim

            layer = []
            if preactivated:
                if norm_layer is not None:
                    layer.append(norm_layer)

                if activ is not None:
                    layer.append(activ)

                layer.append(linear)

                if dropout > 0.0:
                    layer.append(nn.Dropout(dropout))
            else:
                layer.append(linear)

                if activ is not None:
                    layer.append(activ)

                if norm_layer is not None:
                    layer.append(norm_layer)

                if dropout > 0.0:
                    layer.append(nn.Dropout(dropout))

            block = nn.Sequential(*layer)
            self.add_module('layer_{:03d}'.format(layer_idx), block)
            self.blocks.append(block)

        self.output_layer = nn.Linear(curr_input_dim, output_dim)
        init_weights(self.output_layer, gain=gain,
                     init_type='xavier',
                     distr='uniform')

    def extra_repr(self):
        msg = []
        msg.append('Flatten: {}'.format(self.flatten))
        return '\n'.join(msg)

    def forward(self, module_input):
        batch_size = module_input.shape[0]
        # Flatten all dimensions
        curr_input = module_input
        if self.flatten:
            curr_input = curr_input.view(batch_size, -1)
        for block in self.blocks:
            curr_input = block(curr_input)
        return self.output_layer(curr_input)


class IterativeRegression(nn.Module):
    def __init__(self, module, mean_param, num_stages=1,
                 append_params=True, learn_mean=False,
                 detach_mean=False, dim=1,
                 **kwargs):
        super(IterativeRegression, self).__init__()
        logger.info(f'Building iterative regressor with {num_stages} stages')

        self.module = module
        self._num_stages = num_stages
        self.dim = dim

        if learn_mean:
            self.register_parameter('mean_param',
                                    nn.Parameter(mean_param,
                                                 requires_grad=True))
        else:
            self.register_buffer('mean_param', mean_param)

        self.append_params = append_params
        self.detach_mean = detach_mean
        logger.info(f'Detach mean: {self.detach_mean}')

    def get_mean(self):
        return self.mean_param.clone()

    @property
    def num_stages(self):
        return self._num_stages

    def extra_repr(self):
        msg = [
            f'Num stages = {self.num_stages}',
            f'Concatenation dimension: {self.dim}',
            f'Detach mean: {self.detach_mean}',
        ]
        return '\n'.join(msg)

    def forward(
        self,
        features: Tensor,
        cond: Optional[Tensor] = None
    ) -> Tuple[List[Tensor], List[Tensor]]:
        ''' Computes deltas on top of condition iteratively

            Parameters
            ----------
                features: torch.Tensor
                    Input features
        '''
        batch_size = features.shape[0]
        expand_shape = [batch_size] + [-1] * len(features.shape[1:])

        parameters = []
        deltas = []
        module_input = features

        if cond is None:
            cond = self.mean_param.expand(*expand_shape).clone()

        # Detach mean
        if self.detach_mean:
            cond = cond.detach()

        if self.append_params:
            assert features is not None, (
                'Features are none even though append_params is True')

            module_input = torch.cat([
                module_input,
                cond],
                dim=self.dim)
        deltas.append(self.module(module_input))
        num_params = deltas[-1].shape[1]
        parameters.append(cond[:, :num_params].clone() + deltas[-1])

        for stage_idx in range(1, self.num_stages):
            module_input = torch.cat(
                [features, parameters[stage_idx - 1]], dim=-1)
            params_upd = self.module(module_input)
            deltas.append(params_upd)
            parameters.append(parameters[stage_idx - 1] + params_upd)

        return parameters, deltas
