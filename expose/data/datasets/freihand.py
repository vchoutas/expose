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

import json
import time

import torch
import torch.utils.data as dutils
import numpy as np

from loguru import logger

from ..targets import (Keypoints2D, Keypoints3D,
                       Betas, Expression, GlobalPose, BodyPose,
                       #  WeakPerspectiveCamera,
                       BoundingBox,
                       HandPose, JawPose, Vertices, Joints)
from ..targets.keypoints import dset_to_body_model, get_part_idxs
from ..utils.bbox import keyps_to_bbox, bbox_to_center_scale

from ...utils.img_utils import read_img

FOLDER_MAP_FNAME = 'folder_map.pkl'

IMG_SIZE = 224
REF_BOX_SIZE = 200


class FreiHand(dutils.Dataset):
    def __init__(self, data_path='data/freihand',
                 hand_only=True,
                 split='train',
                 dtype=torch.float32,
                 joints_to_ign=None,
                 metrics=None,
                 transforms=None,
                 return_params=True,
                 return_vertices=True,
                 use_face_contour=False,
                 return_shape=False,
                 file_format='json',
                 **kwargs):

        super(FreiHand, self).__init__()

        assert hand_only, 'FreiHand can only be used as a hand dataset'

        if metrics is None:
            metrics = []
        self.metrics = metrics

        self.split = split
        self.is_train = 'train' in split
        self.return_params = return_params
        self.return_vertices = return_vertices
        self.use_face_contour = use_face_contour

        self.return_shape = return_shape
        key = ('training' if 'val' in split or 'train' in split else
               'evaluation')
        self.data_path = osp.expandvars(osp.expanduser(data_path))
        self.img_folder = osp.join(self.data_path, key, 'rgb')
        self.transforms = transforms
        self.dtype = dtype

        intrinsics_path = osp.join(self.data_path, f'{key}_K.json')
        param_path = osp.join(self.data_path, f'{key}_mano.json')
        xyz_path = osp.join(self.data_path, f'{key}_xyz.json')
        vertices_path = osp.join(self.data_path, f'{key}_verts.json')

        start = time.perf_counter()
        if file_format == 'json':
            with open(intrinsics_path, 'r') as f:
                intrinsics = json.load(f)
            if self.split != 'test':
                with open(param_path, 'r') as f:
                    param = json.load(f)
                with open(xyz_path, 'r') as f:
                    xyz = json.load(f)
                if self.return_vertices:
                    with open(vertices_path, 'r') as f:
                        vertices = json.load(f)
        elif file_format == 'npz':
            param_path = osp.join(self.data_path, f'{key}.npz')
            data = np.load(param_path)
            intrinsics = data['intrinsics']
            param = data['param']
            xyz = data['xyz']
            if self.return_vertices:
                vertices = data['vertices']
            self.translation = np.asarray(data['translation'])

            data.close()
        elapsed = time.perf_counter() - start
        logger.info(f'Loading parameters: {elapsed}')

        mean_pose_path = osp.expandvars(
            '$CLUSTER_HOME/SMPL_HF_Regressor_data/data/all_means.pkl')
        mean_poses_dict = {}
        if osp.exists(mean_pose_path):
            logger.info('Loading mean pose from: {} ', mean_pose_path)
            with open(mean_pose_path, 'rb') as f:
                mean_poses_dict = pickle.load(f)

        if self.split != 'test':
            split_size = 0.8
            #  num_items = len(xyz) * 4
            num_green_bg = len(xyz)
            # For green background images
            train_idxs = np.arange(0, int(split_size * num_green_bg))
            val_idxs = np.arange(int(split_size * num_green_bg), num_green_bg)

            all_train_idxs = []
            all_val_idxs = []
            for idx in range(4):
                all_val_idxs.append(val_idxs + num_green_bg * idx)
                all_train_idxs.append(train_idxs + num_green_bg * idx)
            self.train_idxs = np.concatenate(all_train_idxs)
            self.val_idxs = np.concatenate(all_val_idxs)

        if split == 'train':
            self.img_idxs = self.train_idxs
            self.param_idxs = self.train_idxs % num_green_bg
            self.start = 0
        elif split == 'val':
            self.img_idxs = self.val_idxs
            self.param_idxs = self.val_idxs % num_green_bg
            #  self.start = len(self.train_idxs)
        elif 'test' in split:
            self.img_idxs = np.arange(len(intrinsics))
            self.param_idxs = np.arange(len(intrinsics))

        self.num_items = len(self.img_idxs)

        self.intrinsics = intrinsics
        if 'test' not in split:
            xyz = np.asarray(xyz, dtype=np.float32)
            param = np.asarray(param, dtype=np.float32).reshape(len(xyz), -1)
            if self.return_vertices:
                vertices = np.asarray(vertices, dtype=np.float32)

            right_hand_mean = mean_poses_dict['right_hand_pose']['aa'].squeeze()
            self.poses = param[:, :48].reshape(num_green_bg, -1, 3)
            self.poses[:, 1:] += right_hand_mean[np.newaxis]
            self.betas = param[:, 48:58].copy()

            intrinsics = np.asarray(intrinsics, dtype=np.float32)

            if self.return_vertices:
                self.vertices = vertices
            self.xyz = xyz

        folder_map_fname = osp.expandvars(
            osp.join(self.data_path, split, FOLDER_MAP_FNAME))
        self.use_folder_split = osp.exists(folder_map_fname)
        if self.use_folder_split:
            self.img_folder = osp.join(self.data_path, split)
            logger.info(self.img_folder)
            with open(folder_map_fname, 'rb') as f:
                data_dict = pickle.load(f)
            self.items_per_folder = max(data_dict.values())

        if joints_to_ign is None:
            joints_to_ign = []
            self.joints_to_ign = np.array(joints_to_ign, dtype=np.int32)

        source_idxs, target_idxs = dset_to_body_model(dset='freihand')
        self.source_idxs = np.asarray(source_idxs, dtype=np.int64)
        self.target_idxs = np.asarray(target_idxs, dtype=np.int64)

    def get_elements_per_index(self):
        return 1

    def __repr__(self):
        return 'FreiHand( \n\t Split: {}\n)'.format(self.split)

    def name(self):
        return 'FreiHand/{}'.format(self.split)

    def get_num_joints(self):
        return 21

    def __len__(self):
        return self.num_items

    def only_2d(self):
        return False

    def project_points(self, K, xyz):
        uv = np.matmul(K, xyz.T).T
        return uv[:, :2] / uv[:, -1:]

    def __getitem__(self, index):
        img_idx = self.img_idxs[index]
        param_idx = self.param_idxs[index]

        if self.use_folder_split:
            folder_idx = index // self.items_per_folder
            file_idx = index

        K = self.intrinsics[param_idx].copy()
        if 'test' not in self.split:
            pose = self.poses[param_idx].copy()

            global_pose = pose[0].reshape(-1)
            right_hand_pose = pose[1:].reshape(-1)

            scale = 0.5 * (K[0, 0] + K[1, 1])
            #  focal = scale * 2 / IMG_SIZE
            #  pp = K[:2, 2] / scale - IMG_SIZE / (2 * scale)

            keypoints3d = self.xyz[param_idx].copy()
            keypoints2d = self.project_points(K, keypoints3d)
            #  pp -= keypoints3d[0, :2]

            keypoints3d -= keypoints3d[0]

            keypoints2d = np.concatenate(
                [keypoints2d, np.ones_like(keypoints2d[:, [-1]])], axis=-1
            )
            keypoints3d = np.concatenate(
                [keypoints3d, np.ones_like(keypoints2d[:, [-1]])], axis=-1
            )

        #  logger.info('Reading keypoints: {}', time.perf_counter() - start)

        if self.use_folder_split:
            img_fn = osp.join(
                self.img_folder, f'folder_{folder_idx:010d}',
                f'{file_idx:010d}.jpg')
        else:
            img_fn = osp.join(self.img_folder, f'{img_idx:08d}.jpg')

        #  start = time.perf_counter()
        img = read_img(img_fn)
        #  logger.info('Reading image: {}'.format(time.perf_counter() - start))

        if 'test' in self.split:
            bbox = np.array([0, 0, 224, 224], dtype=np.float32)
            target = BoundingBox(bbox, size=img.shape)
        else:
            # Pad to compensate for extra keypoints
            output_keypoints2d = np.zeros(
                [127 + 17 * self.use_face_contour, 3], dtype=np.float32)
            output_keypoints3d = np.zeros(
                [127 + 17 * self.use_face_contour, 4], dtype=np.float32)

            output_keypoints2d[self.target_idxs] = keypoints2d[self.source_idxs]
            output_keypoints3d[self.target_idxs] = keypoints3d[self.source_idxs]

            target = Keypoints2D(
                output_keypoints2d, img.shape, flip_axis=0, dtype=self.dtype)
            #  _, scale, _ = bbox_to_center_scale(
                #  keyps_to_bbox(output_keypoints2d[:, :-1],
                              #  output_keypoints2d[:, -1], img_size=img.shape),
                #  dset_scale_factor=2.0, ref_bbox_size=224,
            #  )
            keyp3d_target = Keypoints3D(
                output_keypoints3d, img.shape[:-1], flip_axis=0, dtype=self.dtype)
            target.add_field('keypoints3d', keyp3d_target)
            target.add_field('intrinsics', K)

        target.add_field('bbox_size', IMG_SIZE / 2)
        center = np.array([IMG_SIZE, IMG_SIZE], dtype=np.float32) * 0.5
        target.add_field('orig_center', np.asarray(img.shape[:-1]) * 0.5)
        target.add_field('center', center)
        scale = IMG_SIZE / REF_BOX_SIZE
        target.add_field('scale', scale)
        #  target.bbox = np.asarray([0, 0, IMG_SIZE, IMG_SIZE], dtype=np.float32)

        #  target.add_field('camera', WeakPerspectiveCamera(focal, pp))

        #  start = time.perf_counter()
        if self.return_params:
            global_pose_field = GlobalPose(global_pose=global_pose)
            target.add_field('global_pose', global_pose_field)
            hand_pose_field = HandPose(right_hand_pose=right_hand_pose,
                                       left_hand_pose=None)
            target.add_field('hand_pose', hand_pose_field)

        if hasattr(self, 'translation'):
            translation = self.translation[param_idx]
        else:
            translation = np.zeros([3], dtype=np.float32)
        target.add_field('translation', translation)

        if self.return_vertices:
            vertices = self.vertices[param_idx]
            hand_vertices_field = Vertices(vertices)
            target.add_field('vertices', hand_vertices_field)
        if self.return_shape:
            target.add_field('betas', Betas(self.betas[param_idx]))

        #  print('SMPL-HF Field {}'.format(time.perf_counter() - start))

        #  start = time.perf_counter()
        if self.transforms is not None:
            full_img, cropped_image, target = self.transforms(
                img, target, dset_scale_factor=2.0)
        #  logger.info('Transforms: {}'.format(time.perf_counter() - start))

        target.add_field('name', self.name())
        # Key used to access the fit dict
        #  img_fn = osp.split(self.img_fns[index])[1].decode('utf-8')

        #  dict_key = ['curated_fits', img_fn, index]

        #  dict_key = tuple(dict_key)
        #  target.add_field('dict_key', dict_key)

        return full_img, cropped_image, target, index
