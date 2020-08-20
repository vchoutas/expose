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

from ..utils.bbox import (
    keyps_to_bbox, bbox_to_center_scale, scale_to_bbox_size)
from ..targets import (Keypoints2D, BodyPose, GlobalPose, Betas, Vertices,
                       HandPose, JawPose, Expression, BoundingBox)
from ..targets.keypoints import (KEYPOINT_NAMES,
                                 get_part_idxs,
                                 dset_to_body_model)
from ...utils import nand

from ...utils.img_utils import read_img
FOLDER_MAP_FNAME = 'folder_map.pkl'


class SPIN(dutils.Dataset):
    def __init__(self, img_folder, npz_files=[], dtype=torch.float32,
                 use_face_contour=False,
                 binarization=True,
                 body_thresh=0.1,
                 hand_thresh=0.2,
                 face_thresh=0.4,
                 min_hand_keypoints=8,
                 min_head_keypoints=8,
                 transforms=None,
                 split='train',
                 return_shape=False,
                 return_full_pose=False,
                 return_params=True,
                 return_gender=False,
                 vertex_folder='vertices',
                 return_vertices=True,
                 vertex_flip_correspondences='',
                 **kwargs):
        super(SPIN, self).__init__()

        self.img_folder = osp.expandvars(img_folder)
        self.transforms = transforms
        self.use_face_contour = use_face_contour
        self.body_thresh = body_thresh
        self.hand_thresh = hand_thresh
        self.face_thresh = face_thresh
        self.binarization = binarization
        self.dtype = dtype
        self.split = split

        self.min_hand_keypoints = min_hand_keypoints
        self.min_head_keypoints = min_head_keypoints

        self.return_vertices = return_vertices
        self.return_gender = return_gender
        self.return_params = return_params
        self.return_shape = return_shape
        self.return_full_pose = return_full_pose

        self.vertex_folder = osp.join(
            osp.split(self.img_folder)[0], vertex_folder)

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

        self.spin_data = {}
        start = 0
        for npz_fn in npz_files:
            npz_fn = osp.expandvars(npz_fn)
            dset = osp.splitext(osp.split(npz_fn)[1])[0]

            data = np.load(npz_fn)
            has_smpl = np.asarray(data['has_smpl']).astype(np.bool)
            data = {key: data[key][has_smpl] for key in data.keys()}

            logger.info(start)
            data['dset'] = [dset] * data['pose'].shape[0]
            start += data['pose'].shape[0]
            if 'genders' not in data and self.return_gender:
                data['genders'] = [''] * len(data['pose'])
            data['indices'] = np.arange(data['pose'].shape[0])
            if dset == 'lsp':
                data['part'][26, [9, 11], :] = data['part'][26, [11, 9], :]
            self.spin_data[dset] = data

        folder_map_fname = osp.expandvars(
            osp.join(img_folder, FOLDER_MAP_FNAME))
        with open(folder_map_fname, 'rb') as f:
            data_dict = pickle.load(f)
        self.items_per_folder = max(data_dict.values())

        self.indices = np.concatenate(
            [self.spin_data[dset]['indices'] for dset in self.spin_data],
            axis=0).astype(np.int32)
        self.centers = np.concatenate(
            [self.spin_data[dset]['center'] for dset in self.spin_data],
            axis=0).astype(np.float32)
        self.scales = np.concatenate(
            [self.spin_data[dset]['scale'] for dset in self.spin_data],
            axis=0).astype(np.float32)
        self.poses = np.concatenate(
            [self.spin_data[dset]['pose']
             for dset in self.spin_data], axis=0).astype(np.float32)
        self.keypoints2d = np.concatenate(
            [self.spin_data[dset]['part'] for dset in self.spin_data],
            axis=0).astype(np.float32)
        self.imgname = np.concatenate(
            [self.spin_data[dset]['imgname']
             for dset in self.spin_data],
            axis=0).astype(np.string_)
        self.dset = np.concatenate([self.spin_data[dset]['dset']
                                    for dset in self.spin_data],
                                   axis=0).astype(np.string_)
        if self.return_gender:
            gender = []
            for dset in self.spin_data:
                gender.append(self.spin_data[dset]['genders'])
            self.gender = np.concatenate(gender).astype(np.string_)

        if self.return_shape:
            self.betas = np.concatenate(
                [self.spin_data[dset]['betas']
                 for dset in self.spin_data], axis=0).astype(np.float32)

        #  self.dset_names = list(self.spin_data.keys())
        dset_sizes = list(
            map(lambda x: x['pose'].shape[0], self.spin_data.values()))
        #  logger.info(self.dset_sizes)

        self.num_items = sum(dset_sizes)
        #  logger.info(self.num_items)

        source_idxs, target_idxs = dset_to_body_model(
            model_type='smplx', use_hands=True, use_face=True,
            dset='spin', use_face_contour=self.use_face_contour)
        self.source_idxs = np.asarray(source_idxs, dtype=np.int64)
        self.target_idxs = np.asarray(target_idxs, dtype=np.int64)

        idxs_dict = get_part_idxs()
        body_idxs = idxs_dict['body']
        hand_idxs = idxs_dict['hand']
        face_idxs = idxs_dict['face']
        if not self.use_face_contour:
            face_idxs = face_idxs[:-17]
        self.body_idxs = np.asarray(body_idxs)
        self.hand_idxs = np.asarray(hand_idxs)
        self.face_idxs = np.asarray(face_idxs)

    def get_elements_per_index(self):
        return 1

    def name(self):
        return 'SPIN/{}'.format(self.split)

    def only_2d(self):
        return False

    def __len__(self):
        return self.num_items

    def __getitem__(self, index):
        folder_idx = index // self.items_per_folder
        file_idx = index

        img_fn = osp.join(self.img_folder,
                          'folder_{:010d}'.format(folder_idx),
                          '{:010d}.jpg'.format(file_idx))
        img = read_img(img_fn)
        keypoints2d = self.keypoints2d[index]

        output_keypoints2d = np.zeros([127 + 17 * self.use_face_contour,
                                       3], dtype=np.float32)

        output_keypoints2d[self.target_idxs] = keypoints2d[self.source_idxs]

        # Remove joints with negative confidence
        output_keypoints2d[output_keypoints2d[:, -1] < 0, -1] = 0
        if self.body_thresh > 0:
            # Only keep the points with confidence above a threshold
            body_conf = output_keypoints2d[self.body_idxs, -1]
            hand_conf = output_keypoints2d[self.hand_idxs, -1]
            face_conf = output_keypoints2d[self.face_idxs, -1]

            body_conf[body_conf < self.body_thresh] = 0.0
            hand_conf[hand_conf < self.hand_thresh] = 0.0
            face_conf[face_conf < self.face_thresh] = 0.0
            if self.binarization:
                body_conf = (
                    body_conf >= self.body_thresh).astype(
                        output_keypoints2d.dtype)
                hand_conf = (
                    hand_conf >= self.hand_thresh).astype(
                        output_keypoints2d.dtype)
                face_conf = (
                    face_conf >= self.face_thresh).astype(
                        output_keypoints2d.dtype)

            output_keypoints2d[self.body_idxs, -1] = body_conf
            output_keypoints2d[self.hand_idxs, -1] = hand_conf
            output_keypoints2d[self.face_idxs, -1] = face_conf

        target = Keypoints2D(
            output_keypoints2d, img.shape, flip_axis=0, dtype=self.dtype)

        keypoints = output_keypoints2d[:, :-1]
        conf = output_keypoints2d[:, -1]
        _, _, bbox_size = bbox_to_center_scale(
            keyps_to_bbox(keypoints, conf, img_size=img.shape),
            dset_scale_factor=1.2
        )
        center = self.centers[index]
        scale = self.scales[index]
        target.add_field('center', center)
        target.add_field('scale', scale)
        target.add_field('bbox_size', bbox_size)
        target.add_field('keypoints_hd', output_keypoints2d)

        if self.return_params:
            pose = self.poses[index].reshape(-1, 3)

            global_pose_target = GlobalPose(pose[0].reshape(-1))
            target.add_field('global_pose', global_pose_target)
            if self.return_full_pose:
                body_pose = pose[1:]
            else:
                body_pose = pose[1:22]
            body_pose_target = BodyPose(body_pose.reshape(-1))
            target.add_field('body_pose', body_pose_target)

        if self.return_shape:
            betas = self.betas[index]
            target.add_field('betas', Betas(betas))
        if self.return_vertices:
            fname = osp.join(self.vertex_folder, f'{index:06d}.npy')
            H, W, _ = img.shape

            fscale = H / bbox_size
            intrinsics = np.array([[5000 * fscale, 0, 0],
                                   [0, 5000 * fscale, 0],
                                   [0, 0, 1]], dtype=np.float32)

            target.add_field('intrinsics', intrinsics)
            vertices = np.load(fname)
            vertex_field = Vertices(
                vertices, bc=self.bc, closest_faces=self.closest_faces)
            target.add_field('vertices', vertex_field)

        if self.transforms is not None:
            force_flip = False
            full_img, cropped_image, cropped_target = self.transforms(
                img, target, dset_scale_factor=1.2, force_flip=force_flip)
        target.add_field('name', self.name())

        dict_key = [f'spin/{self.dset[index].decode("utf-8")}',
                    self.imgname[index].decode('utf-8'), index]
        if hasattr(self, 'gender') and self.return_gender:
            gender = self.gender[index].decode('utf-8')
            if gender == 'F' or gender == 'M':
                target.add_field('gender', gender)
            dict_key.append(gender)

        # Add the key used to access the fit dict
        dict_key = tuple(dict_key)
        target.add_field('dict_key', dict_key)

        return full_img, cropped_image, cropped_target, index


class SPINX(SPIN):
    def __init__(self, return_params=True,
                 head_only=False,
                 hand_only=False,
                 return_expression=True,
                 *args, **kwargs):
        super(SPINX, self).__init__(return_params=return_params,
                                    *args, **kwargs)
        assert nand(head_only, hand_only), (
            'Hand only and head only can\'t be True at the same time')

        self.return_expression = return_expression
        self.head_only = head_only
        self.hand_only = hand_only

        self.keypoints2d = np.concatenate(
            [self.spin_data[dset]['body_keypoints']
                for dset in self.spin_data],
            axis=0).astype(np.float32)
        self.left_hand_keypoints = np.concatenate(
            [self.spin_data[dset]['left_hand_keypoints']
             for dset in self.spin_data], axis=0)
        self.right_hand_keypoints = np.concatenate(
            [self.spin_data[dset]['right_hand_keypoints']
             for dset in self.spin_data], axis=0)
        self.face_keypoints = np.concatenate(
            [self.spin_data[dset]['face_keypoints']
             for dset in self.spin_data], axis=0)

        self.spin_keypoints = np.concatenate(
            [self.spin_data[dset]['spin_keypoints']
             for dset in self.spin_data], axis=0)

        if self.return_expression:
            self.expression = np.concatenate(
                [self.spin_data[dset]['expression']
                 for dset in self.spin_data], axis=0).astype(np.float32)

        self.translation = np.concatenate(
            [self.spin_data[dset]['translation']
             for dset in self.spin_data], axis=0).astype(np.float32)

        source_idxs, target_idxs = dset_to_body_model(
            model_type='smplx', use_hands=True, use_face=True,
            dset='openpose25+hands+face',
            #  dset='spinx',
            use_face_contour=self.use_face_contour)
        self.source_idxs = np.asarray(source_idxs, dtype=np.int64)
        self.target_idxs = np.asarray(target_idxs, dtype=np.int64)

        idxs_dict = get_part_idxs()
        body_idxs = idxs_dict['body']
        hand_idxs = idxs_dict['hand']
        left_hand_idxs = idxs_dict['left_hand']
        right_hand_idxs = idxs_dict['right_hand']
        face_idxs = idxs_dict['face']
        head_idxs = idxs_dict['head']
        if not self.use_face_contour:
            face_idxs = face_idxs[:-17]
            head_idxs = head_idxs[:-17]

        self.body_idxs = np.asarray(body_idxs)
        self.hand_idxs = np.asarray(hand_idxs)
        self.left_hand_idxs = np.asarray(left_hand_idxs)
        self.right_hand_idxs = np.asarray(right_hand_idxs)
        self.face_idxs = np.asarray(face_idxs)
        self.head_idxs = np.asarray(head_idxs)

    def get_elements_per_index(self):
        return 1

    def name(self):
        return 'SPINX/{}'.format(self.split)

    def only_2d(self):
        return False

    def __len__(self):
        return self.num_items

    def __getitem__(self, index):
        folder_idx = index // self.items_per_folder
        file_idx = index

        img_fn = osp.join(self.img_folder,
                          'folder_{:010d}'.format(folder_idx),
                          '{:010d}.jpg'.format(file_idx))
        img = read_img(img_fn)

        body_keypoints = self.keypoints2d[index]
        left_hand_keypoints = self.left_hand_keypoints[index]
        right_hand_keypoints = self.right_hand_keypoints[index]
        face_keypoints = self.face_keypoints[index]

        keypoints2d = np.concatenate(
            [body_keypoints, left_hand_keypoints, right_hand_keypoints,
             face_keypoints], axis=0)

        output_keypoints2d = np.zeros([127 + 17 * self.use_face_contour,
                                       3], dtype=np.float32)

        output_keypoints2d[self.target_idxs] = keypoints2d[self.source_idxs]

        # Remove joints with negative confidence
        output_keypoints2d[output_keypoints2d[:, -1] < 0, -1] = 0
        if self.body_thresh > 0:
            # Only keep the points with confidence above a threshold
            body_conf = output_keypoints2d[self.body_idxs, -1]
            hand_conf = output_keypoints2d[self.hand_idxs, -1]
            face_conf = output_keypoints2d[self.face_idxs, -1]

            body_conf[body_conf < self.body_thresh] = 0.0
            hand_conf[hand_conf < self.hand_thresh] = 0.0
            face_conf[face_conf < self.face_thresh] = 0.0
            if self.binarization:
                body_conf = (
                    body_conf >= self.body_thresh).astype(
                        output_keypoints2d.dtype)
                hand_conf = (
                    hand_conf >= self.hand_thresh).astype(
                        output_keypoints2d.dtype)
                face_conf = (
                    face_conf >= self.face_thresh).astype(
                        output_keypoints2d.dtype)

            output_keypoints2d[self.body_idxs, -1] = body_conf
            output_keypoints2d[self.hand_idxs, -1] = hand_conf
            output_keypoints2d[self.face_idxs, -1] = face_conf

        if self.head_only:
            keypoints = output_keypoints2d[self.head_idxs, :-1]
            conf = output_keypoints2d[self.head_idxs, -1]
        elif self.hand_only:
            keypoints = output_keypoints2d[self.hand_idxs, :-1]
            conf = output_keypoints2d[self.hand_idxs, -1]
        else:
            keypoints = output_keypoints2d[:, :-1]
            conf = output_keypoints2d[:, -1]
        center, scale, bbox_size = bbox_to_center_scale(
            keyps_to_bbox(keypoints, conf, img_size=img.shape),
            dset_scale_factor=1.2,
        )

        target = Keypoints2D(
            output_keypoints2d, img.shape[:-1], flip_axis=0, dtype=self.dtype)
        _, _, bbox_size = bbox_to_center_scale(
            keyps_to_bbox(keypoints, conf, img_size=img.shape),
            dset_scale_factor=1.2)

        center = self.centers[index]
        scale = self.scales[index]
        target.add_field('center', center)
        target.add_field('scale', scale)
        target.add_field('bbox_size', bbox_size)

        target.add_field('keypoints_hd', output_keypoints2d)

        target.add_field('orig_center', center)
        target.add_field('orig_bbox_size', scale * 200)

        left_hand_bbox = keyps_to_bbox(
            output_keypoints2d[self.left_hand_idxs, :-1],
            output_keypoints2d[self.left_hand_idxs, -1],
            img_size=img.shape, scale=1.5)
        left_hand_bbox_target = BoundingBox(left_hand_bbox, img.shape)
        has_left_hand = (output_keypoints2d[self.left_hand_idxs, -1].sum() >
                         self.min_hand_keypoints)
        if has_left_hand:
            target.add_field('left_hand_bbox', left_hand_bbox_target)
            target.add_field(
                'orig_left_hand_bbox',
                BoundingBox(left_hand_bbox, img.shape, transform=False))

        right_hand_bbox = keyps_to_bbox(
            output_keypoints2d[self.right_hand_idxs, :-1],
            output_keypoints2d[self.right_hand_idxs, -1],
            img_size=img.shape, scale=1.5)
        right_hand_bbox_target = BoundingBox(right_hand_bbox, img.shape)
        has_right_hand = (output_keypoints2d[self.right_hand_idxs, -1].sum() >
                          self.min_hand_keypoints)
        if has_right_hand:
            target.add_field('right_hand_bbox', right_hand_bbox_target)
            target.add_field(
                'orig_right_hand_bbox',
                BoundingBox(right_hand_bbox, img.shape, transform=False))

        head_bbox = keyps_to_bbox(
            output_keypoints2d[self.head_idxs, :-1],
            output_keypoints2d[self.head_idxs, -1],
            img_size=img.shape, scale=1.2)
        head_bbox_target = BoundingBox(head_bbox, img.shape)
        has_head = (output_keypoints2d[self.head_idxs, -1].sum() >
                    self.min_head_keypoints)
        if has_head:
            target.add_field('head_bbox', head_bbox_target)
            target.add_field(
                'orig_head_bbox',
                BoundingBox(head_bbox, img.shape, transform=False))

        if self.return_params:
            pose = self.poses[index].reshape(-1, 3)

            global_pose_target = GlobalPose(pose[0].reshape(-1))
            target.add_field('global_pose', global_pose_target)
            body_pose = pose[1:22]
            body_pose_target = BodyPose(body_pose.reshape(-1))
            target.add_field('body_pose', body_pose_target)

            jaw_pose = pose[22]
            jaw_pose_target = JawPose(jaw_pose.reshape(-1))
            target.add_field('jaw_pose', jaw_pose_target)

            left_hand_pose = pose[25:25 + 15]
            right_hand_pose = pose[-15:]
            hand_pose_target = HandPose(left_hand_pose.reshape(-1),
                                        right_hand_pose.reshape(-1))
            target.add_field('hand_pose', hand_pose_target)

            if self.return_shape:
                betas = self.betas[index]
                target.add_field('betas', Betas(betas))

            expression = self.expression[index]
            target.add_field('expression', Expression(expression))

        if self.transforms is not None:
            force_flip = False
            full_img, cropped_image, cropped_target = self.transforms(
                img, target, force_flip=force_flip)

        target.add_field('name', self.name())

        dict_key = [f'spinx/{self.dset[index].decode("utf-8")}',
                    self.imgname[index].decode('utf-8'),
                    self.indices[index]]

        if hasattr(self, 'gender') and self.return_gender:
            gender = self.gender[index].decode('utf-8')
            if gender == 'F' or gender == 'M':
                target.add_field('gender', gender)
            dict_key.append(gender)

        # Add the key used to access the fit dict
        dict_key = tuple(dict_key)
        target.add_field('dict_key', dict_key)

        return full_img, cropped_image, cropped_target, index


class LSPTest(dutils.Dataset):
    def __init__(self, data_path,
                 return_full_pose=False,
                 return_params=True,
                 transforms=None,
                 use_face_contour=True,
                 dtype=torch.float32,
                 **kwargs,
                 ):
        super(LSPTest, self).__init__()

        self.img_folder = osp.expandvars(
            '/ps/project/handsproject/SMPL_HF/lsp/lsp_dataset_original/images')
        self.data_path = osp.expandvars(data_path)
        self.transforms = transforms
        self.use_face_contour = use_face_contour
        self.dtype = dtype
        self.return_vertices = False

        data = np.load(self.data_path)
        #  has_smpl = np.asarray(data['has_smpl']).astype(np.bool)
        self.centers = data['center'].astype(np.float32)
        self.scales = data['scale'].astype(np.float32)
        self.keypoints2d = data['part'].astype(np.float32)
        logger.info(self.keypoints2d.shape)
        self.imgname = data['imgname'].astype(np.string_)

        logger.info(self.scales.shape)
        self.num_items = len(self.scales)
        data.close()

        source_idxs, target_idxs = dset_to_body_model(
            model_type='smplx', use_hands=True, use_face=True,
            dset='lsp', use_face_contour=self.use_face_contour)
        self.source_idxs = np.asarray(source_idxs, dtype=np.int64)
        self.target_idxs = np.asarray(target_idxs, dtype=np.int64)

        idxs_dict = get_part_idxs()
        body_idxs = idxs_dict['body']
        hand_idxs = idxs_dict['hand']
        face_idxs = idxs_dict['face']
        if not self.use_face_contour:
            face_idxs = face_idxs[:-17]
        self.body_idxs = np.asarray(body_idxs)
        self.hand_idxs = np.asarray(hand_idxs)
        self.face_idxs = np.asarray(face_idxs)

    def __len__(self):
        return self.num_items

    def name(self):
        return 'LSP/{Test}'

    def __getitem__(self, index):
        img_name = self.imgname[index].decode('utf-8')
        img_path = osp.join(self.img_folder, img_name)

        img = read_img(img_path)
        keypoints2d = self.keypoints2d[index]

        output_keypoints2d = np.zeros([127 + 17 * self.use_face_contour,
                                       3], dtype=np.float32)

        output_keypoints2d[self.target_idxs, :-1] = keypoints2d[
            self.source_idxs]
        output_keypoints2d[self.target_idxs, -1] = 1.0

        target = Keypoints2D(
            output_keypoints2d, img.shape, flip_axis=0, dtype=self.dtype)

        center = self.centers[index]
        scale = self.scales[index]
        bbox_size = scale * 200

        target.add_field('center', center)
        target.add_field('scale', scale)
        target.add_field('bbox_size', bbox_size)
        target.add_field('keypoints_hd', output_keypoints2d)
        target.add_field('name', self.name())
        target.add_field('fname', img_name)

        target.add_field('orig_center', center)
        target.add_field('orig_bbox_size', scale * 200)

        if self.return_vertices:
            H, W, _ = img.shape

            intrinsics = np.array([[5000, 0, 0.5 * W],
                                   [0, 5000, 0.5 * H],
                                   [0, 0, 1]], dtype=np.float32)
            target.add_field('intrinsics', intrinsics)

            fname = osp.join(self.vertex_folder, f'{index:06d}.npy')
            vertices = np.load(fname) + self.translation[index]
            vertex_field = Vertices(
                vertices, bc=self.bc, closest_faces=self.closest_faces)
            target.add_field('vertices', vertex_field)

        if self.transforms is not None:
            force_flip = False
            full_img, cropped_image, cropped_target = self.transforms(
                img, target, force_flip=force_flip)

        return full_img, cropped_image, cropped_target, index
