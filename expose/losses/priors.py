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

import os
import os.path as osp

import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger

from .utils import get_reduction_method


__all__ = ['build_prior',
           'MeanPrior',
           'IdentityPrior',
           'PenaltyPrior',
           'BarrierPrior',
           'L1Prior',
           'L2Prior',
           'GMMPrior',
           'ThresholdPrior']


def build_prior(prior_type, rho=100, reduction='mean', size_average=True,
                **kwargs):
    logger.debug('Building prior: {}', prior_type)
    if prior_type == 'l2':
        return L2Prior(reduction=reduction, **kwargs)
    elif prior_type == 'l1':
        return L1Prior(reduction=reduction, **kwargs)
    elif prior_type == 'identity':
        return IdentityPrior(reduction=reduction, **kwargs)
    elif prior_type == 'mean':
        return MeanPrior(reduction=reduction, **kwargs)
    elif prior_type == 'penalty':
        return PenaltyPrior(reduction=reduction, **kwargs)
    elif prior_type == 'barrier':
        return BarrierPrior(reduction=reduction, **kwargs)
    elif prior_type == 'threshold':
        return ThresholdPrior(reduction=reduction, **kwargs)
    elif prior_type == 'gmm':
        return GMMPrior(reduction=reduction, **kwargs)
    else:
        raise ValueError('Unknown prior type: {}'.format(prior_type))


class MeanPrior(nn.Module):
    def __init__(self, mean=None, reduction='mean', **kwargs):
        super(MeanPrior, self).__init__()
        assert mean is not None, 'Request MeanPrior, but mean was not given!'
        if type(mean) is not torch.Tensor:
            mean = torch.tensor(mean)
        self.register_buffer('mean', mean.view(1, *list(mean.shape)))
        self.reduction_str = reduction
        self.reduction = get_reduction_method(reduction)

    def extra_repr(self):
        return f'Mean: {self.mean.shape}'

    def forward(self, module_input, *args, **kwargs):
        return (module_input - self.mean).pow(2).sum() / module_input.shape[0]


class IdentityPrior(nn.Module):
    def __init__(self, reduction='mean', **kwargs):
        ''' Penalizes inputs to be close to identity matrix
        '''
        super(IdentityPrior, self).__init__()
        self.reduction_str = reduction
        self.reduction = get_reduction_method(reduction)

        self.register_buffer(
            'identity', torch.eye(3, dtype=torch.float32).unsqueeze(dim=0))

    def forward(self, module_input, *args, **kwargs):
        x = module_input.view(-1, 3, 3)
        batch_size = module_input.shape[0]

        return (x - self.identity).pow(2).sum() / batch_size


class ThresholdPrior(nn.Module):
    def __init__(self, reduction='mean', margin=1, norm='l2', epsilon=1e-7,
                 **kwargs):
        super(ThresholdPrior, self).__init__()
        self.reduction_str = reduction
        self.reduction = get_reduction_method(reduction)
        self.margin = margin
        assert norm in ['l1', 'l2'], 'Norm variable must me l1 or l2'
        self.norm = norm
        self.epsilon = epsilon

    def extra_repr(self):
        msg = 'Reduction: {}\n'.format(self.reduction_str)
        msg += 'Margin: {}\n'.format(self.margin)
        msg += 'Norm: {}'.format(self.norm)
        return msg

    def forward(self, module_input, *args, **kwargs):
        batch_size = module_input.shape[0]

        abs_values = module_input.abs()
        mask = abs_values.gt(self.margin)

        invalid_values = torch.masked_select(module_input, mask)

        if self.norm == 'l1':
            return invalid_values.abs().sum() / (
                mask.to(dtype=module_input.dtype).sum() + self.epsilon
            )
        elif self.norm == 'l2':
            return invalid_values.pow(2).sum() / (
                mask.to(dtype=module_input.dtype).sum() + self.epsilon
            )


class PenaltyPrior(nn.Module):
    def __init__(self, reduction='mean', margin=1, norm='l2', epsilon=1e-7,
                 use_vector=True,
                 **kwargs):
        ''' Soft constraint to prevent parameters for leaving feasible set

            Implements a penalty constraint that encourages the parameters to
            stay in the feasible set of solutions. Assumes that the initial
            estimate is already in this set
        '''
        super(PenaltyPrior, self).__init__()
        self.reduction_str = reduction
        self.reduction = get_reduction_method(reduction)
        self.margin = margin
        assert norm in ['l1', 'l2'], 'Norm variable must me l1 or l2'
        self.norm = norm
        self.epsilon = epsilon
        self.use_vector = use_vector

    def extra_repr(self):
        msg = 'Reduction: {}\n'.format(self.reduction_str)
        msg += 'Margin: {}\n'.format(self.margin)
        msg += 'Norm: {}'.format(self.norm)
        return msg

    def forward(self, module_input, *args, **kwargs):
        batch_size = module_input.shape[0]
        if self.use_vector:

            if self.norm == 'l1':
                param_norm = module_input.abs().view(
                    batch_size, -1).sum(dim=-1)
                margin = self.margin
            elif self.norm == 'l2':
                param_norm = module_input.pow(2).view(
                    batch_size, -1).sum(dim=-1)
                margin = self.margin ** 2

            thresholded_vals = F.relu(param_norm - margin)
            non_zeros = (
                thresholded_vals.gt(0).to(torch.float32).sum() + self.epsilon)
            return (thresholded_vals.sum() / non_zeros)
        else:
            upper_margin = F.relu(module_input - self.margin)
            lower_margin = F.relu(-(module_input + self.margin))
            with torch.no_grad():
                upper_non_zeros = (
                    upper_margin.gt(0).to(torch.float32).sum() + self.epsilon)
                lower_non_zeros = (
                    lower_margin.gt(0).to(torch.float32).sum() + self.epsilon)

            if self.norm == 'l1':
                return (upper_margin.abs().sum() / upper_non_zeros +
                        lower_margin.abs().sum() / lower_non_zeros)
            elif self.norm == 'l2':
                return (upper_margin.pow(2).sum() / upper_non_zeros +
                        lower_margin.pow(2).sum() / lower_non_zeros)


class BarrierPrior(nn.Module):
    def __init__(self, reduction='mean', margin=1, barrier='log',
                 epsilon=1e-7, symmetric=True, **kwargs):
        ''' Soft constraint that pushes parameters away from the border

            Implements a barrier constraint that encourages the parameters to
            stay away from the border of the feasible set. Assumes that the initial
            estimate is already in this set
        '''
        super(BarrierPrior, self).__init__()
        self.reduction_str = reduction
        self.reduction = get_reduction_method(reduction)
        assert barrier in ['log', 'inv'], 'Norm variable must me inv or log'
        self.barrier = barrier
        self.epsilon = epsilon
        self.symmetric = symmetric
        self.register_buffer('margin', torch.tensor(margin))

    def extra_repr(self):
        msg = 'Reduction: {}\n'.format(self.reduction_str)
        msg += 'Margin: {}\n'.format(self.margin)
        msg += 'Barrier: {}'.format(self.barrier)
        msg += 'Symmetric: {}'.format(self.symmetric)
        return msg

    def forward(self, module_input, *args, **kwargs):
        if self.barrier == 'log':
            loss = -torch.log(self.margin) - torch.log(
                -(module_input - self.margin) + self.epsilon).mean()
            if self.symmetric:
                loss += -torch.log(self.margin) - torch.log(
                    (module_input + self.margin) + self.epsilon).mean()
        elif self.barrier == 'inv':
            loss = - 1 / (module_input - self.margin + self.epsilon).mean()
            if self.symmetric:
                loss += 1 / (module_input + self.margin)
                # Compensate for the minimum to make it zero
                loss -= 1
        return loss


class L1Prior(nn.Module):
    def __init__(self, dtype=torch.float32, reduction='mean', **kwargs):
        super(L1Prior, self).__init__()
        self.reduction = get_reduction_method(reduction)

    def forward(self, module_input, *args):
        return self.reduction(module_input.abs().sum(dim=-1))


class L2Prior(nn.Module):
    def __init__(self, dtype=torch.float32, reduction='mean', **kwargs):
        super(L2Prior, self).__init__()
        self.reduction = get_reduction_method(reduction)

    def forward(self, module_input, *args):
        return self.reduction(module_input.pow(2))


class GMMPrior(nn.Module):

    def __init__(self, path,
                 num_gaussians=6, dtype=torch.float32, epsilon=1e-16,
                 reduction='mean',
                 use_max=False,
                 **kwargs):
        super(GMMPrior, self).__init__()

        logger.debug('Loading GMMPrior from {}', path)
        if dtype == torch.float32:
            np_dtype = np.float32
        elif dtype == torch.float64:
            np_dtype = np.float64
        else:
            raise ValueError(
                'Unknown float type {}.format(exiting)!'.format(dtype))

        self.num_gaussians = num_gaussians
        self.epsilon = epsilon
        self.reduction = get_reduction_method(reduction)
        self.use_max = use_max
        self.dtype = dtype

        path = osp.expanduser(osp.expandvars(path))
        with open(path, 'rb') as f:
            gmm = pickle.load(f, encoding='latin1')

        if type(gmm) == dict:
            means = gmm['means']
            covs = gmm['covars']
            weights = gmm['weights']
        elif 'sklearn.mixture.gmm.GMM' in str(type(gmm)):
            means = gmm.means_
            covs = gmm.covars_
            weights = gmm.weights_
        else:
            msg = 'Unknown type for the prior: {}, exiting!'.format(type(gmm))
            raise ValueError(msg)

        self.register_buffer('means', torch.tensor(means, dtype=dtype))
        self.register_buffer('covs', torch.tensor(covs, dtype=dtype))

        precisions = [np.linalg.inv(cov) for cov in covs]
        precisions = np.stack(precisions)

        self.register_buffer('precisions',
                             torch.tensor(precisions, dtype=dtype))

        nll_weights = np.asarray(gmm['weights'])
        nll_weights = torch.tensor(nll_weights, dtype=dtype).unsqueeze(dim=0)

        nll_weights = torch.log(nll_weights)
        self.register_buffer('nll_weights', nll_weights)

        weights = torch.tensor(gmm['weights'], dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('weights', weights)

        self.register_buffer('pi_term',
                             torch.log(torch.tensor(2 * np.pi, dtype=dtype)))

        cov_dets = [np.log(np.linalg.det(covs[idx]))
                    for idx in range(covs.shape[0])]

        self.register_buffer('cov_dets',
                             torch.tensor(cov_dets, dtype=dtype))

        # The dimensionality of the random variable
        self.random_var_dim = self.means.shape[1]

    def extra_repr(self):
        msg = []
        msg.append(f'Mean: {self.means.shape}')
        msg.append(f'Covariance: {self.covs.shape}')
        return '\n'.join(msg)

    def get_mean(self):
        ''' Returns the mean of the mixture '''
        mean_pose = torch.matmul(self.weights, self.means)
        return mean_pose

    def max_log_likelihood(self, pose, *args):
        diff_from_mean = pose.unsqueeze(dim=1) - self.means

        prec_diff_prod = torch.einsum('mij,bmj->bmi',
                                      [self.precisions, diff_from_mean])
        diff_prec_quadratic = (prec_diff_prod * diff_from_mean).sum(dim=-1)

        curr_loglikelihood = -0.5 * (diff_prec_quadratic +
                                     self.cov_dets +
                                     self.random_var_dim * self.pi_term)
        curr_loglikelihood += (-self.nll_weights)
        #  curr_loglikelihood = 0.5 * diff_prec_quadratic - \
        #  torch.log(self.nll_weights)

        min_likelihood, _ = torch.min(curr_loglikelihood, dim=1)
        return self.reduction(min_likelihood)

    def logsumexp_likelihood(self, pose, *args, **kwargs):
        diff_from_mean = pose.unsqueeze(dim=1) - self.means

        prec_diff_prod = torch.einsum('mij,bmj->bmi',
                                      [self.precisions, diff_from_mean])
        diff_prec_quadratic = (prec_diff_prod * diff_from_mean).sum(dim=-1)

        exponent = (self.nll_weights -
                    0.5 * self.random_var_dim * self.pi_term -
                    0.5 * self.cov_dets -
                    0.5 * diff_prec_quadratic)
        logsumexp = -torch.logsumexp(exponent, dim=-1)

        return self.reduction(logsumexp)

    def forward(self, pose, *args):
        if len(pose.shape) == 4:
            raise NotImplementedError

        if self.use_max:
            return self.max_log_likelihood(pose, *args)
        else:
            return self.logsumexp_likelihood(pose, *args)
