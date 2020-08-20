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
import os
import os.path as osp
import pickle

import time

import torch
import torch.utils.data as dutils
import numpy as np

from loguru import logger

from ..targets import BoundingBox
from ..utils import bbox_to_center_scale

from ...utils.img_utils import read_img

FOLDER_MAP_FNAME = 'folder_map.pkl'


class Stirling3D(dutils.Dataset):
    def __init__(self, data_path='data/stirling/HQ',
                 head_only=True,
                 split='train',
                 dtype=torch.float32,
                 metrics=None,
                 transforms=None,
                 **kwargs):
        super(Stirling3D, self).__init__()
        assert head_only, 'Stirling3D can only be used as a head only dataset'

        self.split = split
        assert 'test' in split, (
            f'Stirling3D can only be used for testing, but got split: {split}'
        )
        if metrics is None:
            metrics = []
        self.metrics = metrics

        self.data_path = osp.expandvars(osp.expanduser(data_path))
        self.transforms = transforms
        self.dtype = dtype

        self.img_paths = np.array(
            [osp.join(self.data_path, fname)
             for fname in sorted(os.listdir(self.data_path))]
        )
        self.num_items = len(self.img_paths)

    def get_elements_per_index(self):
        return 1

    def __repr__(self):
        return 'Stirling3D( \n\t Split: {self.split}\n)'

    def name(self):
        return f'Stirling3D/{self.split}'

    def get_num_joints(self):
        return 0

    def only_2d(self):
        return False

    def __len__(self):
        return self.num_items

    def __getitem__(self, index):
        img = read_img(self.img_paths[index])

        H, W, _ = img.shape
        bbox = np.array([0, 0, W - 1, H - 1], dtype=np.float32)
        target = BoundingBox(bbox, size=img.shape)

        center = np.array([W, H], dtype=np.float32) * 0.5
        target.add_field('center', center)

        center, scale, bbox_size = bbox_to_center_scale(bbox)
        target.add_field('scale', scale)
        target.add_field('bbox_size', bbox_size)
        target.add_field('image_size', img.shape)

        if self.transforms is not None:
            img, cropped_image, target = self.transforms(img, target)

        target.add_field('name', self.name())
        target.add_field('fname', osp.split(self.img_paths[index])[1])
        return img, cropped_image, target, index
