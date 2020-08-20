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

from typing import NewType, Union, Tuple

import numpy as np
import random

import time
from copy import deepcopy
from loguru import logger
import cv2

import PIL.Image as pil_img
import torch
import torchvision
from torchvision.transforms import functional as F

from ..targets import GenericTarget
from ..targets.keypoints import get_part_idxs
from ...utils.transf_utils import crop, get_transform


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        self.timers = {}

    def __call__(self, image, target, **kwargs):
        next_input = (image, target)
        for t in self.transforms:
            output = t(*next_input, **kwargs)
            next_input = output
        return next_input

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __str__(self):
        return 'RandomHorizontalFlip({:.03f})'.format(self.prob)

    def _flip(self, img):
        if img is None:
            return None
        if 'numpy.ndarray' in str(type(img)):
            return np.ascontiguousarray(img[:, ::-1, :]).copy()
        else:
            return F.hflip(img)

    def __call__(self, image, target, force_flip=False, **kwargs):
        flip = random.random() < self.prob
        target.add_field('is_flipped', flip)
        if flip or force_flip:
            output_image = self._flip(image)
            flipped_target = target.transpose(0)

            _, W, _ = output_image.shape

            left_hand_bbox, right_hand_bbox = None, None
            if flipped_target.has_field('left_hand_bbox'):
                left_hand_bbox = flipped_target.get_field('left_hand_bbox')
            if target.has_field('right_hand_bbox'):
                right_hand_bbox = flipped_target.get_field('right_hand_bbox')
            if left_hand_bbox is not None:
                flipped_target.add_field('right_hand_bbox', left_hand_bbox)
            if right_hand_bbox is not None:
                flipped_target.add_field('left_hand_bbox', right_hand_bbox)

            width = target.size[1]
            center = target.get_field('center')
            TO_REMOVE = 1
            center[0] = width - center[0] - TO_REMOVE

            if target.has_field('keypoints_hd'):
                keypoints_hd = target.get_field('keypoints_hd')
                flipped_keypoints_hd = keypoints_hd.copy()
                flipped_keypoints_hd[:, 0] = (
                    width - flipped_keypoints_hd[:, 0] - TO_REMOVE)
                flipped_keypoints_hd = flipped_keypoints_hd[target.FLIP_INDS]
                flipped_target.add_field('keypoints_hd', flipped_keypoints_hd)

            # Update the center
            flipped_target.add_field('center', center)
            if target.has_field('orig_center'):
                orig_center = target.get_field('orig_center').copy()
                orig_center[0] = width - orig_center[0] - TO_REMOVE
                flipped_target.add_field('orig_center', orig_center)

            if target.has_field('intrinsics'):
                intrinsics = target.get_field('intrinsics')
                cam_center = intrinsics[:2, 2].copy()
                cam_center[0] = width - cam_center[0] - TO_REMOVE
                intrinsics[:2, 2] = cam_center
                flipped_target.add_field('intrinsics', intrinsics)
            # Expressions are not symmetric, so we remove them from the targets
            # when the image is flipped
            if flipped_target.has_field('expression'):
                flipped_target.delete_field('expression')

            return output_image, flipped_target
        else:
            return image, target


class BBoxCenterJitter(object):
    def __init__(self, factor=0.0, dist='normal'):
        super(BBoxCenterJitter, self).__init__()
        self.factor = factor
        self.dist = dist
        assert self.dist in ['normal', 'uniform'], (
            f'Distribution must be normal or uniform, not {self.dist}')

    def __str__(self):
        return f'BBoxCenterJitter({self.factor:0.2f})'

    def __call__(self, image, target, **kwargs):
        if self.factor <= 1e-3:
            return image, target

        bbox_size = target.get_field('bbox_size')

        jitter = bbox_size * self.factor

        if self.dist == 'normal':
            center_jitter = np.random.randn(2) * jitter
        elif self.dist == 'uniform':
            center_jitter = np.random.rand(2) * 2 * jitter - jitter

        center = target.get_field('center')
        H, W, _ = target.size
        new_center = center + center_jitter
        new_center[0] = np.clip(new_center[0], 0, W)
        new_center[1] = np.clip(new_center[1], 0, H)

        target.add_field('center', new_center)

        return image, target


class SimulateLowRes(object):
    def __init__(
        self,
        dist: str = 'categorical',
        factor: float = 1.0,
        cat_factors: Tuple[float] = (1.0,),
        factor_min: float = 1.0,
        factor_max: float = 1.0
    ) -> None:
        self.factor_min = factor_min
        self.factor_max = factor_max
        self.dist = dist
        self.cat_factors = cat_factors
        assert dist in ['uniform', 'categorical']

    def __str__(self) -> str:
        if self.dist == 'uniform':
            dist_str = (
                f'{self.dist.title()}: [{self.factor_min}, {self.factor_max}]')
        else:
            dist_str = (
                f'{self.dist.title()}: [{self.cat_factors}]')
        return f'SimulateLowResolution({dist_str})'

    def _sample_low_res(
        self,
        image: Union[np.ndarray, pil_img.Image]
    ) -> np.ndarray:
        '''
        '''
        if self.dist == 'uniform':
            downsample = self.factor_min != self.factor_max
            if not downsample:
                return image
            factor = np.random.rand() * (
                self.factor_max - self.factor_min) + self.factor_min
        elif self.dist == 'categorical':
            if len(self.cat_factors) < 2:
                return image
            idx = np.random.randint(0, len(self.cat_factors))
            factor = self.cat_factors[idx]

        H, W, _ = image.shape
        downsampled_image = cv2.resize(
            image, (int(W // factor), int(H // factor)), cv2.INTER_NEAREST
        )
        resized_image = cv2.resize(
            downsampled_image, (W, H), cv2.INTER_LINEAR_EXACT)
        return resized_image

    def __call__(
        self,
        image: Union[np.ndarray, pil_img.Image],
        cropped_image: Union[np.ndarray, pil_img.Image],
        target: GenericTarget,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, GenericTarget]:
        '''
        '''
        if torch.is_tensor(cropped_image):
            raise NotImplementedError
        elif isinstance(cropped_image, (pil_img.Image, np.ndarray)):
            resized_image = self._sample_low_res(cropped_image)

        return image, resized_image, target


class ChannelNoise(object):
    def __init__(self, noise_scale=0.0):
        self.noise_scale = noise_scale

    def __str__(self):
        return 'ChannelNoise: {:.4f}'.format(self.noise_scale)

    def __call__(
        self,
        image: Union[np.ndarray, pil_img.Image],
        cropped_image: Union[np.ndarray, pil_img.Image],
        target: GenericTarget,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, GenericTarget]:
        '''
        '''
        if self.noise_scale > 0:
            if image.dtype == np.float32:
                img_max = 1.0
            elif image.dtype == np.uint8:
                img_max = 255
            # Each channel is multiplied with a number
            # in the area [1 - self.noise_scale,1 + self.noise_scale]
            pn = np.random.uniform(1 - self.noise_scale,
                                   1 + self.noise_scale, 3)
            if not isinstance(image, (np.ndarray, )):
                image = np.asarray(image)
            if not isinstance(cropped_image, (np.ndarray,)):
                cropped_image = np.asarray(cropped_image)
            output_image = np.clip(
                image * pn[np.newaxis, np.newaxis], 0,
                img_max).astype(image.dtype)
            output_cropped_image = np.clip(
                cropped_image * pn[np.newaxis, np.newaxis], 0,
                img_max).astype(image.dtype)

            return output_image, output_cropped_image, target
        else:
            return image, cropped_image, target


class RandomRotation(object):
    def __init__(self, is_train: bool = True,
                 rotation_factor: float = 0):
        self.is_train = is_train
        self.rotation_factor = rotation_factor

    def __str__(self):
        return f'RandomRotation(rotation_factor={self.rotation_factor})'

    def __repr__(self):
        msg = [
            f'Training: {self.is_training}',
            f'Rotation factor: {self.rotation_factor}'
        ]
        return '\n'.join(msg)

    def __call__(self, image, target, **kwargs):
        rot = 0.0
        if not self.is_train:
            return image, target
        if self.is_train:
            rot = min(2 * self.rotation_factor,
                      max(-2 * self.rotation_factor,
                          np.random.randn() * self.rotation_factor))
            if np.random.uniform() <= 0.6:
                rot = 0
        if rot == 0.0:
            return image, target

        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), rot, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        # perform the actual rotation and return the image
        rotated_image = cv2.warpAffine(image, M, (nW, nH))

        new_target = target.rotate(rot=rot)

        center = target.get_field('center')
        center = np.dot(M[:2, :2], center) + M[:2, 2]
        new_target.add_field('center', center)

        if target.has_field('keypoints_hd'):
            keypoints_hd = target.get_field('keypoints_hd')
            rotated_keyps = (
                np.dot(keypoints_hd[:, :2], M[:2, :2].T) + M[:2, 2] +
                1).astype(np.int)
            rotated_keyps = np.concatenate(
                [rotated_keyps, keypoints_hd[:, [2]]], axis=-1)
            new_target.add_field('keypoints_hd', rotated_keyps)

        if target.has_field('intrinsics'):
            intrinsics = target.get_field('intrinsics').copy()

            cam_center = intrinsics[:2, 2]
            intrinsics[:2, 2] = (
                np.dot(M[:2, :2], cam_center) + M[:2, 2])
            new_target.add_field('intrinsics', intrinsics)

        return rotated_image, new_target


class Crop(object):
    def __init__(self, is_train=True,
                 crop_size=224,
                 scale_factor_min=0.00,
                 scale_factor_max=0.00,
                 scale_factor=0.0,
                 scale_dist='uniform',
                 rotation_factor=0,
                 min_hand_bbox_dim=20,
                 min_head_bbox_dim=20,
                 ):
        super(Crop, self).__init__()
        self.crop_size = crop_size

        self.is_train = is_train
        self.scale_factor_min = scale_factor_min
        self.scale_factor_max = scale_factor_max
        self.scale_factor = scale_factor
        self.scale_dist = scale_dist

        self.rotation_factor = rotation_factor
        self.min_hand_bbox_dim = min_hand_bbox_dim
        self.min_head_bbox_dim = min_head_bbox_dim

        part_idxs = get_part_idxs()
        self.left_hand_idxs = part_idxs['left_hand']
        self.right_hand_idxs = part_idxs['right_hand']
        self.head_idxs = part_idxs['head']

    def __str__(self):
        return 'Crop(size={}, scale={}, rotation_factor={})'.format(
            self.crop_size, self.scale_factor, self.rotation_factor)

    def __repr__(self):
        msg = 'Training: {}\n'.format(self.is_train)
        msg += 'Crop size: {}\n'.format(self.crop_size)
        msg += 'Scale factor augm: {}\n'.format(self.scale_factor)
        msg += 'Rotation factor augm: {}'.format(self.rotation_factor)
        return msg

    def __call__(self, image, target, **kwargs):
        sc = 1.0
        if self.is_train:
            if self.scale_dist == 'normal':
                sc = min(1 + self.scale_factor,
                         max(1 - self.scale_factor,
                             np.random.randn() * self.scale_factor + 1))
            elif self.scale_dist == 'uniform':
                if self.scale_factor_max == 0.0 and self.scale_factor_min == 0:
                    sc = 1.0
                else:
                    sc = (np.random.rand() *
                          (self.scale_factor_max - self.scale_factor_min) +
                          self.scale_factor_min)

        scale = target.get_field('scale') * sc
        center = target.get_field('center')
        orig_bbox_size = target.get_field('bbox_size')
        bbox_size = orig_bbox_size * sc

        np_image = np.asarray(image)
        cropped_image = crop(
            np_image, center, scale, [self.crop_size, self.crop_size])
        cropped_target = target.crop(
            center, scale, crop_size=self.crop_size)

        transf = get_transform(
            center, scale, [self.crop_size, self.crop_size])

        cropped_target.add_field('crop_transform', transf)
        cropped_target.add_field('bbox_size', bbox_size)

        if target.has_field('intrinsics'):
            intrinsics = target.get_field('intrinsics').copy()
            fscale = cropped_image.shape[0] / orig_bbox_size
            intrinsics[0, 0] *= (fscale / sc)
            intrinsics[1, 1] *= (fscale / sc)

            cam_center = intrinsics[:2, 2]
            intrinsics[:2, 2] = (
                np.dot(transf[:2, :2], cam_center) + transf[:2, 2])
            cropped_target.add_field('intrinsics', intrinsics)

        return np_image, cropped_image, cropped_target


class ColorJitter(object):
    def __init__(self, brightness=0.0, contrast=0, saturation=0, hue=0):
        super(ColorJitter, self).__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

        self.transform = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast,
            saturation=saturation, hue=hue)

    def __repr__(self):
        name = 'ColorJitter(\n'
        name += f'brightness={self.brightness:.2f}\n'
        name += f'contrast={self.contrast:.2f}\n'
        name += f'saturation={self.saturation:.2f}\n'
        name += f'hue={self.hue:.2f}'
        return name

    def __call__(self, image, target, **kwargs):
        return self.transform(image), target


class ToTensor(object):
    def __init__(self):
        super(ToTensor, self).__init__()

    def __repr__(self):
        return 'ToTensor()'

    def __str__(self):
        return 'ToTensor()'

    def __call__(self, image, cropped_image, target, **kwargs):
        target.to_tensor()
        return F.to_tensor(image), F.to_tensor(cropped_image), target


class Normalize(object):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def __str__(self):
        msg = 'Mean: {}, '.format(self.mean)
        msg += 'Std: {}\n'.format(self.std)
        return msg

    def __repr__(self):
        msg = 'Mean: {}\n'.format(self.mean)
        msg += 'Std: {}\n'.format(self.std)
        return msg

    def __call__(self, image, cropped_image, target, **kwargs):
        output_image = F.normalize(
            image, mean=self.mean, std=self.std)
        output_cropped_image = F.normalize(
            cropped_image, mean=self.mean, std=self.std)
        return output_image, output_cropped_image, target
