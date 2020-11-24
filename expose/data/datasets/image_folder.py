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
import glob

import time

import numpy as np
import torch
import torch.utils.data as dutils

from loguru import logger

from ..utils import bbox_to_center_scale

from expose.utils.img_utils import read_img
from expose.data.targets import BoundingBox
from mmd.utils.MServiceUtils import sort_by_numeric

EXTS = ['.jpg', '.jpeg', '.png']


class ImageFolder(dutils.Dataset):
    def __init__(self,
                 data_folder='data/images',
                 transforms=None,
                 **kwargs):
        super(ImageFolder, self).__init__()

        paths = []
        self.transforms = transforms
        data_folder = osp.expandvars(data_folder)
        for fname in sorted(glob.glob(data_folder), key=sort_by_numeric):
            if not any(fname.endswith(ext) for ext in EXTS):
                continue
            paths.append(osp.join(data_folder, fname))

        self.paths = np.stack(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img, idx_dir = read_img(self.paths[index])

        if self.transforms is not None:
            img = self.transforms(img)

        return {
            'images': img,
            'paths': self.paths[index], 
            'idx_dir': idx_dir
        }


class ImageFolderWithBoxes(dutils.Dataset):
    def __init__(self,
                 img_paths,
                 bboxes,
                 transforms=None,
                 scale_factor=1.2,
                 **kwargs):
        super(ImageFolderWithBoxes, self).__init__()

        self.transforms = transforms

        self.paths = np.stack(img_paths)
        self.bboxes = np.stack(bboxes)
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img, idx_dir = read_img(self.paths[index])

        bbox = self.bboxes[index]

        target = BoundingBox(bbox, size=img.shape)

        center, scale, bbox_size = bbox_to_center_scale(
            bbox, dset_scale_factor=self.scale_factor)
        target.add_field('bbox_size', bbox_size)
        target.add_field('orig_bbox_size', bbox_size)
        target.add_field('orig_center', center)
        target.add_field('center', center)
        target.add_field('scale', scale)

        _, fname = osp.split(self.paths[index])
        target.add_field('fname', f'{fname}_{index:03d}')
        target.add_field('idx_dir', idx_dir)

        if self.transforms is not None:
            full_img, cropped_image, target = self.transforms(img, target)

        return full_img, cropped_image, target, index
