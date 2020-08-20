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
from loguru import logger

from . import transforms as T


def build_transforms(transf_cfg, is_train):
    if is_train:
        flip_prob = transf_cfg.get('flip_prob', 0)
        downsample_dist = transf_cfg.get('downsample_dist', 'categorical')
        downsample_cat_factors = transf_cfg.get(
            'downsample_cat_factors', (1.0, ))
        downsample_factor_min = transf_cfg.get('downsample_factor_min', 1.0)
        downsample_factor_max = transf_cfg.get('downsample_factor_max', 1.0)
        scale_factor = transf_cfg.get('scale_factor', 0.0)
        scale_factor_min = transf_cfg.get('scale_factor_min', 0.0)
        scale_factor_max = transf_cfg.get('scale_factor_max', 0.0)
        scale_dist = transf_cfg.get('scale_dist', 'uniform')
        rotation_factor = transf_cfg.get('rotation_factor', 0.0)
        noise_scale = transf_cfg.get('noise_scale', 0.0)
        center_jitter_factor = transf_cfg.get('center_jitter_factor', 0.0)
        center_jitter_dist = transf_cfg.get('center_jitter_dist', 'normal')
    else:
        flip_prob = 0.0
        downsample_dist = 'categorical'
        downsample_cat_factors = (1.0,)
        downsample_factor_min = 1.0
        downsample_factor_max = 1.0
        scale_factor = 0.0
        scale_factor_min = 1.0
        scale_factor_max = 1.0
        rotation_factor = 0.0
        noise_scale = 0.0
        center_jitter_factor = 0.0
        center_jitter_dist = transf_cfg.get('center_jitter_dist', 'normal')
        scale_dist = transf_cfg.get('scale_dist', 'uniform')

    normalize_transform = T.Normalize(
        transf_cfg.get('mean'), transf_cfg.get('std'))
    logger.debug('Normalize {}', normalize_transform)

    crop_size = transf_cfg.get('crop_size')
    crop = T.Crop(crop_size=crop_size, is_train=is_train,
                  scale_factor_max=scale_factor_max,
                  scale_factor_min=scale_factor_min,
                  scale_factor=scale_factor,
                  scale_dist=scale_dist)
    pixel_noise = T.ChannelNoise(noise_scale=noise_scale)
    logger.debug('Crop {}', crop)

    downsample = T.SimulateLowRes(
        dist=downsample_dist,
        cat_factors=downsample_cat_factors,
        factor_min=downsample_factor_min,
        factor_max=downsample_factor_max)

    transform = T.Compose(
        [
            T.BBoxCenterJitter(center_jitter_factor, dist=center_jitter_dist),
            T.RandomHorizontalFlip(flip_prob),
            T.RandomRotation(
                is_train=is_train, rotation_factor=rotation_factor),
            crop,
            pixel_noise,
            downsample,
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform
