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

from ..targets import (Keypoints2D,
                       Betas, Expression, GlobalPose,
                       #  WeakPerspectiveCamera,
                       JawPose, Vertices, Joints)
from ..targets.keypoints import dset_to_body_model
#  from ..utils.bbox import keyps_to_bbox, bbox_to_center_scale

from ...utils.img_utils import read_img

FOLDER_MAP_FNAME = 'folder_map.pkl'

IMAGE_SIZE = 1024
REF_BOX_SIZE = 200
DEFAULT_FOCAL_LENGTH = 4754.97941935


class FFHQ(dutils.Dataset):
    def __init__(self, data_path='data/ffhq',
                 img_folder='images',
                 param_fname='ffhq_parameters.npz',
                 head_only=True,
                 split='train',
                 dtype=torch.float32,
                 joints_to_ign=None,
                 metrics=None,
                 transforms=None,
                 return_params=True,
                 return_shape=False,
                 return_vertices=False,
                 vertex_folder='vertices',
                 use_face_contour=False,
                 split_size=0.8,
                 vertex_flip_correspondences='',
                 **kwargs):
        super(FFHQ, self).__init__()
        assert head_only, 'FFHQ can only be used as a head only dataset'

        if metrics is None:
            metrics = []
        self.metrics = metrics

        self.split = split
        self.is_train = 'train' in split
        self.return_params = return_params
        self.return_vertices = return_vertices
        self.use_face_contour = use_face_contour

        self.return_shape = return_shape
        self.data_path = osp.expandvars(osp.expanduser(data_path))
        self.img_folder = osp.join(self.data_path, img_folder)

        self.transforms = transforms
        self.dtype = dtype

        param_path = osp.join(self.data_path, param_fname)
        self.vertex_path = osp.join(self.data_path, vertex_folder)

        vertex_flip_correspondences = osp.expandvars(
            vertex_flip_correspondences)
        err_msg = (
            'Vertex flip correspondences path does not exist:' +
            f' {vertex_flip_correspondences}'
        )
        assert osp.exists(vertex_flip_correspondences), err_msg
        flip_data = np.load(vertex_flip_correspondences)
        self.bc = flip_data['bc']
        self.closest_faces = flip_data['closest_faces']

        params = np.load(param_path)
        params_dict = {key: params[key] for key in params.keys()}

        self.global_pose = params_dict['global_pose'].astype(np.float32).copy()
        self.jaw_pose = params_dict['jaw_pose'].astype(np.float32).copy()
        self.betas = params_dict['betas'].astype(np.float32).copy()
        self.expression = params_dict['expression'].astype(np.float32).copy()
        self.keypoints2d = params_dict['keypoints2D'].astype(np.float32).copy()
        self.img_fnames = np.asarray(params_dict['img_fnames'])

        self.return_vertices = return_vertices
        #  if return_vertices:
            #  assert 'vertices' in params_dict, (
                #  'Requested vertices but these are not in the npz file')
            #  self.vertices = params_dict['vertices'].astype(np.float32).copy()

        num_items = len(self.betas)
        idxs = np.arange(num_items)
        if self.is_train:
            self.idxs = idxs[:int(num_items * split_size)]
        else:
            self.idxs = idxs[int(num_items * split_size):]
        self.num_items = len(self.idxs)

        folder_map_fname = osp.expandvars(
            osp.join(self.data_path, img_folder, split, FOLDER_MAP_FNAME))
        self.use_folder_split = osp.exists(folder_map_fname)
        if self.use_folder_split:
            self.img_folder = osp.join(self.data_path, img_folder, split)
            with open(folder_map_fname, 'rb') as f:
                data_dict = pickle.load(f)
            self.items_per_folder = max(data_dict.values())

        source_idxs, target_idxs = dset_to_body_model(
            dset='ffhq', use_face_contour=self.use_face_contour)
        self.source_idxs = np.asarray(source_idxs, dtype=np.int64)
        self.target_idxs = np.asarray(target_idxs, dtype=np.int64)

    def get_elements_per_index(self):
        return 1

    def __repr__(self):
        return 'FFHQ( \n\t Split: {self.split}\n)'

    def name(self):
        return f'FFHQ/{self.split}'

    def get_num_joints(self):
        return 51 + self.use_face_contour * 17

    def only_2d(self):
        return False

    def __len__(self):
        return self.num_items

    def __getitem__(self, index):
        data_idx = self.idxs[index]

        if self.use_folder_split:
            folder_idx = index // self.items_per_folder
            file_idx = index

        global_pose = self.global_pose[data_idx]
        jaw_pose = self.jaw_pose[data_idx]
        expression = self.expression[data_idx]
        keypoints2d = self.keypoints2d[data_idx]

        if self.use_folder_split:
            img_fn = osp.join(
                self.img_folder, f'folder_{folder_idx:010d}',
                f'{file_idx:010d}.jpg')
        else:
            img_fn = osp.join(self.img_folder,
                              str(self.img_fnames[data_idx]))

        img = read_img(img_fn.replace('.png', '.jpg'))

        output_keypoints2d = np.zeros(
            [127 + 17 * self.use_face_contour, 3], dtype=np.float32)
        output_keypoints2d[self.target_idxs, :-1] = keypoints2d[
            self.source_idxs]
        output_keypoints2d[self.target_idxs, -1] = 1.0
        target = Keypoints2D(
            output_keypoints2d, img.shape, flip_axis=0, dtype=self.dtype)

        center = np.array([512, 512], dtype=np.float32)
        scale = IMAGE_SIZE / REF_BOX_SIZE
        target.add_field('orig_center', center)
        target.add_field('center', center)
        target.add_field('scale', scale)
        target.add_field('bbox_size', IMAGE_SIZE)
        H, W, _ = img.shape
        fscale = img.shape[0] / 256
        intrinsics = np.array(
            [[DEFAULT_FOCAL_LENGTH * fscale, 0.0, W * 0.5],
             [0.0, DEFAULT_FOCAL_LENGTH * fscale, H * 0.5],
             [0.0, 0.0, 1.0]]
        )
        target.add_field('intrinsics', intrinsics)
        if self.return_params:
            global_pose_field = GlobalPose(global_pose=global_pose)
            target.add_field('global_pose', global_pose_field)
            jaw_pose_field = JawPose(jaw_pose=jaw_pose)
            target.add_field('jaw_pose', jaw_pose_field)
            expression_field = Expression(expression=expression)
            target.add_field('expression', expression_field)
        if self.return_vertices:
            fname, _ = osp.splitext(self.img_fnames[data_idx])
            vertex_fname = osp.join(self.vertex_path, f'{fname}.npy')
            vertices = np.load(vertex_fname)
            vertex_field = Vertices(
                vertices, bc=self.bc, closest_faces=self.closest_faces)
            target.add_field('vertices', vertex_field)
        if self.return_shape:
            target.add_field('betas', Betas(self.betas[data_idx]))

        if self.transforms is not None:
            img, cropped_image, target = self.transforms(
                img, target, dset_scale_factor=2.0)
        target.add_field('name', self.name())
        return img, cropped_image, target, index
