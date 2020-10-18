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

from copy import deepcopy

import numpy as np

import cv2

import torch
from loguru import logger

from .generic_target import GenericTarget
from expose.utils.transf_utils import get_transform

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class Keypoints2D(GenericTarget):
    def __init__(self, keypoints, size,
                 flip_axis=0,
                 use_face_contour=False,
                 bbox=None,
                 center=None,
                 scale=1.0,
                 source='',
                 **kwargs):
        super(Keypoints2D, self).__init__()
        self.size = size
        self.source = source
        self.bbox = bbox
        self.center = center
        self.scale = scale

        self.flip_axis = flip_axis

        self.smplx_keypoints = keypoints[:, :-1]
        self.conf = keypoints[:, -1]

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'Number of keypoints={}, '.format(self.smplx_keypoints.shape[0])
        s += 'image_width={}, '.format(self.size[1])
        s += 'image_height={})'.format(self.size[0])
        return s

    def to_tensor(self, *args, **kwargs):
        if not torch.is_tensor(self.smplx_keypoints):
            self.smplx_keypoints = torch.from_numpy(self.smplx_keypoints)
            self.conf = torch.from_numpy(self.conf)

        for k, v in self.extra_fields.items():
            if isinstance(v, GenericTarget):
                v.to_tensor(*args, **kwargs)

    def normalize(self, bboxes):
        center = (bboxes[:, 2:] + bboxes[:, :2]) * 0.5
        bbox_width = bboxes[:, 2] - bboxes[:, 0]
        bbox_height = bboxes[:, 3] - bboxes[:, 1]

        if center.shape[0] < 1:
            return
        if self.smplx_keypoints.shape[0] < 1:
            return
        self.smplx_keypoints[:, :, :2] -= center.unsqueeze(dim=1)

        self.smplx_keypoints[:, :, 0] = (
            self.smplx_keypoints[:, :, 0] / bbox_width[:, np.newaxis]) * 2
        self.smplx_keypoints[:, :, 1] = (
            self.smplx_keypoints[:, :, 1] / bbox_height[:, np.newaxis]) * 2

    def rotate(self, rot=0, *args, **kwargs):
        (h, w) = self.size[:2]
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
        kp = self.smplx_keypoints.copy()
        kp = (np.dot(kp, M[:2, :2].T) + M[:2, 2] + 1).astype(np.int)

        conf = self.conf.copy().reshape(-1, 1)
        kp = np.concatenate([kp, conf], axis=1).astype(np.float32)
        keypoints = type(self)(kp, size=(nH, nW, 3))
        for k, v in self.extra_fields.items():
            if isinstance(v, GenericTarget):
                v = v.rotate(rot=rot, *args, **kwargs)
            keypoints.add_field(k, v)

        self.add_field('rot', rot)
        return keypoints

    def crop(self, center, scale, crop_size=224, *args, **kwargs):
        kp = self.smplx_keypoints.copy()
        transf = get_transform(center, scale, (crop_size, crop_size))
        kp = (np.dot(kp, transf[:2, :2].T) + transf[:2, 2] + 1).astype(np.int)

        kp = 2.0 * kp / crop_size - 1.0

        conf = self.conf.copy().reshape(-1, 1)
        kp = np.concatenate([kp, conf], axis=1).astype(np.float32)
        keypoints = type(self)(kp, size=(crop_size, crop_size, 3))
        keypoints.source = self.source
        for k, v in self.extra_fields.items():
            if isinstance(v, GenericTarget):
                v = v.crop(center=center, scale=scale,
                           crop_size=crop_size, *args, **kwargs)
            keypoints.add_field(k, v)

        return keypoints

    def get_keypoints_and_conf(self, key='all'):
        if key == 'all':
            keyp_data = [self.smplx_keypoints, self.conf]
        elif key == 'body':
            keyp_data = [self.smplx_keypoints[BODY_IDXS],
                         self.conf[BODY_IDXS]]
        elif key == 'left_hand':
            keyp_data = [self.smplx_keypoints[LEFT_HAND_IDXS],
                         self.conf[LEFT_HAND_IDXS]]
        elif key == 'right_hand':
            keyp_data = [self.smplx_keypoints[RIGHT_HAND_IDXS],
                         self.conf[RIGHT_HAND_IDXS]]
        elif key == 'head':
            keyp_data = [self.smplx_keypoints[HEAD_IDXS],
                         self.conf[HEAD_IDXS]]
        else:
            raise ValueError(f'Unknown key: {key}')
        if torch.is_tensor(keyp_data[0]):
            return torch.cat(
                [keyp_data[0], keyp_data[1][..., None]], dim=-1)
        else:
            return np.concatenate(
                [keyp_data[0], keyp_data[1][..., None]], axis=-1)

    def resize(self, size, *args, **kwargs):
        ratios = tuple(float(s) / float(s_orig)
                       for s, s_orig in zip(size, self.size))
        ratio_w, ratio_h = ratios
        resized_data = self.smplx_keypoints.copy()

        resized_data[..., 0] *= ratio_w
        resized_data[..., 1] *= ratio_h

        resized_keyps = np.concatenate([resized_data,
                                        self.conf.unsqueeze(dim=-1)], axis=-1)

        keypoints = type(self)(resized_keyps, size=size)
        keypoints.source = self.source
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.resize(size, *args, **kwargs)
            keypoints.add_field(k, v)

        return keypoints

    def __getitem__(self, key):
        if key == 'keypoints':
            return self.smplx_keypoints
        elif key == 'conf':
            return self.conf
        else:
            raise ValueError('Unknown key: {}'.format(key))

    def __len__(self):
        return 1

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT,):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT implemented")

        width = self.size[1]
        TO_REMOVE = 1
        flip_inds = type(self).FLIP_INDS
        if torch.is_tensor(self.smplx_keypoints):
            flipped_data = torch.cat([self.smplx_keypoints,
                                      self.conf.unsqueeze(dim=-1)],
                                     dim=-1)

            num_joints = flipped_data.shape[0]
            #  flipped_data[torch.arange(num_joints)] = torch.index_select(
            #  flipped_data, 0, flip_inds[:num_joints])
            flipped_data[np.arange(num_joints)] = flipped_data[
                flip_inds[:num_joints]]
            #  width = self.size[0]
            #  TO_REMOVE = 1
            # Flip x coordinates
            #  flipped_data[..., 0] = width - flipped_data[..., 0] - TO_REMOVE
            flipped_data[..., :, self.flip_axis] = width - flipped_data[
                ..., :, self.flip_axis] - TO_REMOVE

            #  Maintain COCO convention that if visibility == 0, then x, y = 0
            #  inds = flipped_data[..., 2] == 0
            #  flipped_data[inds] = 0
        else:
            flipped_data = np.concatenate(
                [self.smplx_keypoints, self.conf[..., np.newaxis]], axis=-1)

            num_joints = flipped_data.shape[0]
            flipped_data[np.arange(num_joints)] = flipped_data[
                flip_inds[:num_joints]]
            # Flip x coordinates
            flipped_data[..., 0] = width - flipped_data[..., 0] - TO_REMOVE

            #  Maintain COCO convention that if visibility == 0, then x, y = 0
            #  inds = flipped_data[..., 2] == 0
            #  flipped_data[inds] = 0

        keypoints = type(self)(flipped_data, self.size)
        keypoints.source = self.source
        if self.bbox is not None:
            keypoints.bbox = self.bbox.copy()

        for k, v in self.extra_fields.items():
            if isinstance(v, GenericTarget):
                v = v.transpose(method)
            keypoints.add_field(k, v)

        self.add_field('is_flipped', True)
        return keypoints

    def to(self, *args, **kwargs):
        keyp_tensor = torch.cat([self.smplx_keypoints,
                                 self.conf.unsqueeze(dim=-1)], dim=-1)
        keypoints = type(self)(keyp_tensor.to(*args, **kwargs), self.size)
        keypoints.source = self.source
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            keypoints.add_field(k, v)
        return keypoints


KEYPOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'jaw',
    'left_eye_smplx',
    'right_eye_smplx',
    'left_index1',
    'left_index2',
    'left_index3',
    'left_middle1',
    'left_middle2',
    'left_middle3',
    'left_pinky1',
    'left_pinky2',
    'left_pinky3',
    'left_ring1',
    'left_ring2',
    'left_ring3',
    'left_thumb1',
    'left_thumb2',
    'left_thumb3',
    'right_index1',
    'right_index2',
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3',
    'nose',
    'right_eye',
    'left_eye',
    'right_ear',
    'left_ear',
    'left_big_toe',
    'left_small_toe',
    'left_heel',
    'right_big_toe',
    'right_small_toe',
    'right_heel',
    'left_thumb',
    'left_index',
    'left_middle',
    'left_ring',
    'left_pinky',
    'right_thumb',
    'right_index',
    'right_middle',
    'right_ring',
    'right_pinky',
    'right_eye_brow1',
    'right_eye_brow2',
    'right_eye_brow3',
    'right_eye_brow4',
    'right_eye_brow5',
    'left_eye_brow5',
    'left_eye_brow4',
    'left_eye_brow3',
    'left_eye_brow2',
    'left_eye_brow1',
    'nose1',
    'nose2',
    'nose3',
    'nose4',
    'right_nose_2',
    'right_nose_1',
    'nose_middle',
    'left_nose_1',
    'left_nose_2',
    'right_eye1',
    'right_eye2',
    'right_eye3',
    'right_eye4',
    'right_eye5',
    'right_eye6',
    'left_eye4',
    'left_eye3',
    'left_eye2',
    'left_eye1',
    'left_eye6',
    'left_eye5',
    'right_mouth_1',
    'right_mouth_2',
    'right_mouth_3',
    'mouth_top',
    'left_mouth_3',
    'left_mouth_2',
    'left_mouth_1',
    'left_mouth_5',  # 59 in OpenPose output
    'left_mouth_4',  # 58 in OpenPose output
    'mouth_bottom',
    'right_mouth_4',
    'right_mouth_5',
    'right_lip_1',
    'right_lip_2',
    'lip_top',
    'left_lip_2',
    'left_lip_1',
    'left_lip_3',
    'lip_bottom',
    'right_lip_3',
    # Face contour
    'right_contour_1',
    'right_contour_2',
    'right_contour_3',
    'right_contour_4',
    'right_contour_5',
    'right_contour_6',
    'right_contour_7',
    'right_contour_8',
    'contour_middle',
    'left_contour_8',
    'left_contour_7',
    'left_contour_6',
    'left_contour_5',
    'left_contour_4',
    'left_contour_3',
    'left_contour_2',
    'left_contour_1',
]

MANO_NAMES = [
    'right_wrist',
    'right_index1',
    'right_index2',
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3',
    'right_thumb',
    'right_index',
    'right_middle',
    'right_ring',
    'right_pinky']


KEYPOINT_PARTS = {'pelvis': 'body',
                  'left_hip': 'body,vmd',
                  'right_hip': 'body,vmd',
                  'spine1': 'body,vmd',
                  'left_knee': 'body,vmd',
                  'right_knee': 'body,vmd',
                  'spine2': 'body',
                  'left_ankle': 'body,vmd',
                  'right_ankle': 'body,vmd',
                  'spine3': 'body,vmd',
                  'left_foot': 'body,vmd',
                  'right_foot': 'body,vmd',
                  'neck': 'body,flame,vmd',
                  'left_collar': 'body',
                  'right_collar': 'body',
                  'head': 'body,head,flame,vmd',
                  'left_shoulder': 'body,vmd',
                  'right_shoulder': 'body,vmd',
                  'left_elbow': 'body,vmd',
                  'right_elbow': 'body,vmd',
                  'left_wrist': 'body,hand,vmd',
                  'right_wrist': 'body,hand,vmd',
                  'jaw': 'body,head,flame',
                  'left_eye_smplx': 'body,head,flame',
                  'right_eye_smplx': 'body,head,flame',
                  'left_index1': 'hand',
                  'left_index2': 'hand',
                  'left_index3': 'hand',
                  'left_middle1': 'hand',
                  'left_middle2': 'hand',
                  'left_middle3': 'hand',
                  'left_pinky1': 'hand',
                  'left_pinky2': 'hand',
                  'left_pinky3': 'hand',
                  'left_ring1': 'hand',
                  'left_ring2': 'hand',
                  'left_ring3': 'hand',
                  'left_thumb1': 'hand',
                  'left_thumb2': 'hand',
                  'left_thumb3': 'hand',
                  'right_index1': 'hand',
                  'right_index2': 'hand',
                  'right_index3': 'hand',
                  'right_middle1': 'hand',
                  'right_middle2': 'hand',
                  'right_middle3': 'hand',
                  'right_pinky1': 'hand',
                  'right_pinky2': 'hand',
                  'right_pinky3': 'hand',
                  'right_ring1': 'hand',
                  'right_ring2': 'hand',
                  'right_ring3': 'hand',
                  'right_thumb1': 'hand',
                  'right_thumb2': 'hand',
                  'right_thumb3': 'hand',
                  'nose': 'body,head,vmd',
                  'right_eye': 'body,head',
                  'left_eye': 'body,head',
                  'right_ear': 'body,head',
                  'left_ear': 'body,head',
                  'left_big_toe': 'body',
                  'left_small_toe': 'body',
                  'left_heel': 'body',
                  'right_big_toe': 'body',
                  'right_small_toe': 'body',
                  'right_heel': 'body',
                  'left_thumb': 'hand',
                  'left_index': 'hand',
                  'left_middle': 'hand',
                  'left_ring': 'hand',
                  'left_pinky': 'hand',
                  'right_thumb': 'hand',
                  'right_index': 'hand',
                  'right_middle': 'hand',
                  'right_ring': 'hand',
                  'right_pinky': 'hand',
                  'right_eye_brow1': 'face,head,flame',
                  'right_eye_brow2': 'face,head,flame',
                  'right_eye_brow3': 'face,head,flame',
                  'right_eye_brow4': 'face,head,flame',
                  'right_eye_brow5': 'face,head,flame',
                  'left_eye_brow5': 'face,head,flame',
                  'left_eye_brow4': 'face,head,flame',
                  'left_eye_brow3': 'face,head,flame',
                  'left_eye_brow2': 'face,head,flame',
                  'left_eye_brow1': 'face,head,flame',
                  'nose1': 'face,head,flame',
                  'nose2': 'face,head,flame',
                  'nose3': 'face,head,flame',
                  'nose4': 'face,head,flame',
                  'right_nose_2': 'face,head,flame',
                  'right_nose_1': 'face,head,flame',
                  'nose_middle': 'face,head,flame',
                  'left_nose_1': 'face,head,flame',
                  'left_nose_2': 'face,head,flame',
                  'right_eye1': 'face,head,flame',
                  'right_eye2': 'face,head,flame',
                  'right_eye3': 'face,head,flame',
                  'right_eye4': 'face,head,flame',
                  'right_eye5': 'face,head,flame',
                  'right_eye6': 'face,head,flame',
                  'left_eye4': 'face,head,flame',
                  'left_eye3': 'face,head,flame',
                  'left_eye2': 'face,head,flame',
                  'left_eye1': 'face,head,flame',
                  'left_eye6': 'face,head,flame',
                  'left_eye5': 'face,head,flame',
                  'right_mouth_1': 'face,head,flame',
                  'right_mouth_2': 'face,head,flame',
                  'right_mouth_3': 'face,head,flame',
                  'mouth_top': 'face,head,flame',
                  'left_mouth_3': 'face,head,flame',
                  'left_mouth_2': 'face,head,flame',
                  'left_mouth_1': 'face,head,flame',
                  'left_mouth_5': 'face,head,flame',
                  'left_mouth_4': 'face,head,flame',
                  'mouth_bottom': 'face,head,flame',
                  'right_mouth_4': 'face,head,flame',
                  'right_mouth_5': 'face,head,flame',
                  'right_lip_1': 'face,head,flame',
                  'right_lip_2': 'face,head,flame',
                  'lip_top': 'face,head,flame',
                  'left_lip_2': 'face,head,flame',
                  'left_lip_1': 'face,head,flame',
                  'left_lip_3': 'face,head,flame',
                  'lip_bottom': 'face,head,flame',
                  'right_lip_3': 'face,head,flame',
                  'right_contour_1': 'face,head,flame',
                  'right_contour_2': 'face,head,flame',
                  'right_contour_3': 'face,head,flame',
                  'right_contour_4': 'face,head,flame',
                  'right_contour_5': 'face,head,flame',
                  'right_contour_6': 'face,head,flame',
                  'right_contour_7': 'face,head,flame',
                  'right_contour_8': 'face,head,flame',
                  'contour_middle': 'face,head,flame',
                  'left_contour_8': 'face,head,flame',
                  'left_contour_7': 'face,head,flame',
                  'left_contour_6': 'face,head,flame',
                  'left_contour_5': 'face,head,flame',
                  'left_contour_4': 'face,head,flame',
                  'left_contour_3': 'face,head,flame',
                  'left_contour_2': 'face,head,flame',
                  'left_contour_1': 'face,head,flame'}


def get_part_idxs():
    body_idxs = np.asarray([
        idx
        for idx, val in enumerate(KEYPOINT_PARTS.values())
        if 'body' in val])
    hand_idxs = np.asarray([
        idx
        for idx, val in enumerate(KEYPOINT_PARTS.values())
        if 'hand' in val])

    left_hand_idxs = np.asarray([
        idx
        for idx, val in enumerate(KEYPOINT_PARTS.values())
        if 'hand' in val and 'left' in KEYPOINT_NAMES[idx]])

    right_hand_idxs = np.asarray([
        idx
        for idx, val in enumerate(KEYPOINT_PARTS.values())
        if 'hand' in val and 'right' in KEYPOINT_NAMES[idx]])

    face_idxs = np.asarray([
        idx
        for idx, val in enumerate(KEYPOINT_PARTS.values())
        if 'face' in val])
    head_idxs = np.asarray([
        idx
        for idx, val in enumerate(KEYPOINT_PARTS.values())
        if 'head' in val])
    flame_idxs = np.asarray([
        idx
        for idx, val in enumerate(KEYPOINT_PARTS.values())
        if 'flame' in val])
    #  joint_weights[hand_idxs] = hand_weight
    #  joint_weights[face_idxs] = face_weight
    return {
        'body': body_idxs.astype(np.int64),
        'hand': hand_idxs.astype(np.int64),
        'face': face_idxs.astype(np.int64),
        'head': head_idxs.astype(np.int64),
        'left_hand': left_hand_idxs.astype(np.int64),
        'right_hand': right_hand_idxs.astype(np.int64),
        'flame': flame_idxs.astype(np.int64),
    }


PARTS = get_part_idxs()
BODY_IDXS = PARTS['body']
LEFT_HAND_IDXS = PARTS['left_hand']
RIGHT_HAND_IDXS = PARTS['right_hand']
FACE_IDXS = PARTS['face']
FLAME_IDXS = PARTS['flame']
HEAD_IDXS = PARTS['head']
LEFT_HAND_KEYPOINT_NAMES = [KEYPOINT_NAMES[ii] for ii in LEFT_HAND_IDXS]
RIGHT_HAND_KEYPOINT_NAMES = [KEYPOINT_NAMES[ii] for ii in RIGHT_HAND_IDXS]
HEAD_KEYPOINT_NAMES = [KEYPOINT_NAMES[ii] for ii in HEAD_IDXS]
FLAME_KEYPOINT_NAMES = [KEYPOINT_NAMES[ii] for ii in FLAME_IDXS]


CONNECTIONS = [
    ['left_eye', 'nose'],
    ['right_eye', 'nose'],
    ['right_eye', 'right_ear'],
    ['left_eye', 'left_ear'],
    ['right_shoulder', 'right_elbow'],
    ['right_elbow', 'right_wrist'],
    # Right Thumb
    ['right_wrist', 'right_thumb1'],
    ['right_thumb1', 'right_thumb2'],
    ['right_thumb2', 'right_thumb3'],
    ['right_thumb3', 'right_thumb'],
    # Right Index
    ['right_wrist', 'right_index1'],
    ['right_index1', 'right_index2'],
    ['right_index2', 'right_index3'],
    ['right_index3', 'right_index'],
    # Right Middle
    ['right_wrist', 'right_middle1'],
    ['right_middle1', 'right_middle2'],
    ['right_middle2', 'right_middle3'],
    ['right_middle3', 'right_middle'],
    # Right Ring
    ['right_wrist', 'right_ring1'],
    ['right_ring1', 'right_ring2'],
    ['right_ring2', 'right_ring3'],
    ['right_ring3', 'right_ring'],
    # Right Pinky
    ['right_wrist', 'right_pinky1'],
    ['right_pinky1', 'right_pinky2'],
    ['right_pinky2', 'right_pinky3'],
    ['right_pinky3', 'right_pinky'],
    # Left Hand
    ['left_shoulder', 'left_elbow'],
    ['left_elbow', 'left_wrist'],
    # Left Thumb
    ['left_wrist', 'left_thumb1'],
    ['left_thumb1', 'left_thumb2'],
    ['left_thumb2', 'left_thumb3'],
    ['left_thumb3', 'left_thumb'],
    # Left Index
    ['left_wrist', 'left_index1'],
    ['left_index1', 'left_index2'],
    ['left_index2', 'left_index3'],
    ['left_index3', 'left_index'],
    # Left Middle
    ['left_wrist', 'left_middle1'],
    ['left_middle1', 'left_middle2'],
    ['left_middle2', 'left_middle3'],
    ['left_middle3', 'left_middle'],
    # Left Ring
    ['left_wrist', 'left_ring1'],
    ['left_ring1', 'left_ring2'],
    ['left_ring2', 'left_ring3'],
    ['left_ring3', 'left_ring'],
    # Left Pinky
    ['left_wrist', 'left_pinky1'],
    ['left_pinky1', 'left_pinky2'],
    ['left_pinky2', 'left_pinky3'],
    ['left_pinky3', 'left_pinky'],

    # Right Foot
    ['right_hip', 'right_knee'],
    ['right_knee', 'right_ankle'],
    ['right_ankle', 'right_heel'],
    ['right_ankle', 'right_big_toe'],
    ['right_ankle', 'right_small_toe'],

    ['left_hip', 'left_knee'],
    ['left_knee', 'left_ankle'],
    ['left_ankle', 'left_heel'],
    ['left_ankle', 'left_big_toe'],
    ['left_ankle', 'left_small_toe'],

    ['neck', 'right_shoulder'],
    ['neck', 'left_shoulder'],
    ['neck', 'nose'],
    ['pelvis', 'spine1'],
    ['spine1', 'spine3'],
    ['spine3', 'neck'],
    ['pelvis', 'left_hip'],
    ['pelvis', 'right_hip'],

    # Left Eye brow
    ['left_eye_brow1', 'left_eye_brow2'],
    ['left_eye_brow2', 'left_eye_brow3'],
    ['left_eye_brow3', 'left_eye_brow4'],
    ['left_eye_brow4', 'left_eye_brow5'],

    # Right Eye brow
    ['right_eye_brow1', 'right_eye_brow2'],
    ['right_eye_brow2', 'right_eye_brow3'],
    ['right_eye_brow3', 'right_eye_brow4'],
    ['right_eye_brow4', 'right_eye_brow5'],

    # Left Eye
    ['left_eye1', 'left_eye2'],
    ['left_eye2', 'left_eye3'],
    ['left_eye3', 'left_eye4'],
    ['left_eye4', 'left_eye5'],
    ['left_eye5', 'left_eye6'],
    ['left_eye6', 'left_eye1'],

    # Right Eye
    ['right_eye1', 'right_eye2'],
    ['right_eye2', 'right_eye3'],
    ['right_eye3', 'right_eye4'],
    ['right_eye4', 'right_eye5'],
    ['right_eye5', 'right_eye6'],
    ['right_eye6', 'right_eye1'],

    # Nose Vertical
    ['nose1', 'nose2'],
    ['nose2', 'nose3'],
    ['nose3', 'nose4'],

    # Nose Horizontal
    ['nose4', 'nose_middle'],
    ['left_nose_1', 'left_nose_2'],
    ['left_nose_2', 'nose_middle'],
    ['nose_middle', 'right_nose_1'],
    ['right_nose_1', 'right_nose_2'],

    # Mouth
    ['left_mouth_1', 'left_mouth_2'],
    ['left_mouth_2', 'left_mouth_3'],
    ['left_mouth_3', 'mouth_top'],
    ['mouth_top', 'right_mouth_3'],
    ['right_mouth_3', 'right_mouth_2'],
    ['right_mouth_2', 'right_mouth_1'],
    ['right_mouth_1', 'right_mouth_5'],
    ['right_mouth_5', 'right_mouth_4'],
    ['right_mouth_4', 'mouth_bottom'],
    ['mouth_bottom', 'left_mouth_4'],
    ['left_mouth_4', 'left_mouth_5'],
    ['left_mouth_5', 'left_mouth_1'],

    # Lips
    ['left_lip_1', 'left_lip_2'],
    ['left_lip_2', 'lip_top'],
    ['lip_top', 'right_lip_2'],
    ['right_lip_2', 'right_lip_1'],
    ['right_lip_1', 'right_lip_3'],
    ['right_lip_3', 'lip_bottom'],
    ['lip_bottom', 'left_lip_3'],
    ['left_lip_3', 'left_lip_1'],

    # Contour
    ['left_contour_1', 'left_contour_2'],
    ['left_contour_2', 'left_contour_3'],
    ['left_contour_3', 'left_contour_4'],
    ['left_contour_4', 'left_contour_5'],
    ['left_contour_5', 'left_contour_6'],
    ['left_contour_6', 'left_contour_7'],
    ['left_contour_7', 'left_contour_8'],
    ['left_contour_8', 'contour_middle'],

    ['contour_middle', 'right_contour_8'],
    ['right_contour_8', 'right_contour_7'],
    ['right_contour_7', 'right_contour_6'],
    ['right_contour_6', 'right_contour_5'],
    ['right_contour_5', 'right_contour_4'],
    ['right_contour_4', 'right_contour_3'],
    ['right_contour_3', 'right_contour_2'],
    ['right_contour_2', 'right_contour_1'],
]

def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        # Right Thumb
        [keypoints.index('right_wrist'), keypoints.index('right_thumb1')],
        [keypoints.index('right_thumb1'), keypoints.index('right_thumb2')],
        [keypoints.index('right_thumb2'), keypoints.index('right_thumb3')],
        [keypoints.index('right_thumb3'), keypoints.index('right_thumb')],
        # Right Index
        [keypoints.index('right_wrist'), keypoints.index('right_index1')],
        [keypoints.index('right_index1'), keypoints.index('right_index2')],
        [keypoints.index('right_index2'), keypoints.index('right_index3')],
        [keypoints.index('right_index3'), keypoints.index('right_index')],
        # Right Middle
        [keypoints.index('right_wrist'), keypoints.index('right_middle1')],
        [keypoints.index('right_middle1'), keypoints.index('right_middle2')],
        [keypoints.index('right_middle2'), keypoints.index('right_middle3')],
        [keypoints.index('right_middle3'), keypoints.index('right_middle')],
        # Right Ring
        [keypoints.index('right_wrist'), keypoints.index('right_ring1')],
        [keypoints.index('right_ring1'), keypoints.index('right_ring2')],
        [keypoints.index('right_ring2'), keypoints.index('right_ring3')],
        [keypoints.index('right_ring3'), keypoints.index('right_ring')],
        # Right Pinky
        [keypoints.index('right_wrist'), keypoints.index('right_pinky1')],
        [keypoints.index('right_pinky1'), keypoints.index('right_pinky2')],
        [keypoints.index('right_pinky2'), keypoints.index('right_pinky3')],
        [keypoints.index('right_pinky3'), keypoints.index('right_pinky')],
        # Left Hand
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        # Left Thumb
        [keypoints.index('left_wrist'), keypoints.index('left_thumb1')],
        [keypoints.index('left_thumb1'), keypoints.index('left_thumb2')],
        [keypoints.index('left_thumb2'), keypoints.index('left_thumb3')],
        [keypoints.index('left_thumb3'), keypoints.index('left_thumb')],
        # Left Index
        [keypoints.index('left_wrist'), keypoints.index('left_index1')],
        [keypoints.index('left_index1'), keypoints.index('left_index2')],
        [keypoints.index('left_index2'), keypoints.index('left_index3')],
        [keypoints.index('left_index3'), keypoints.index('left_index')],
        # Left Middle
        [keypoints.index('left_wrist'), keypoints.index('left_middle1')],
        [keypoints.index('left_middle1'), keypoints.index('left_middle2')],
        [keypoints.index('left_middle2'), keypoints.index('left_middle3')],
        [keypoints.index('left_middle3'), keypoints.index('left_middle')],
        # Left Ring
        [keypoints.index('left_wrist'), keypoints.index('left_ring1')],
        [keypoints.index('left_ring1'), keypoints.index('left_ring2')],
        [keypoints.index('left_ring2'), keypoints.index('left_ring3')],
        [keypoints.index('left_ring3'), keypoints.index('left_ring')],
        # Left Pinky
        [keypoints.index('left_wrist'), keypoints.index('left_pinky1')],
        [keypoints.index('left_pinky1'), keypoints.index('left_pinky2')],
        [keypoints.index('left_pinky2'), keypoints.index('left_pinky3')],
        [keypoints.index('left_pinky3'), keypoints.index('left_pinky')],

        # Right Foot
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('right_ankle'), keypoints.index('right_heel')],
        [keypoints.index('right_ankle'), keypoints.index('right_big_toe')],
        [keypoints.index('right_ankle'), keypoints.index('right_small_toe')],

        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('left_ankle'), keypoints.index('left_heel')],
        [keypoints.index('left_ankle'), keypoints.index('left_big_toe')],
        [keypoints.index('left_ankle'), keypoints.index('left_small_toe')],

        [keypoints.index('neck'), keypoints.index('right_shoulder')],
        [keypoints.index('neck'), keypoints.index('left_shoulder')],
        [keypoints.index('neck'), keypoints.index('nose')],
        [keypoints.index('pelvis'), keypoints.index('spine1')],
        [keypoints.index('spine1'), keypoints.index('spine3')],
        [keypoints.index('spine3'), keypoints.index('neck')],
        [keypoints.index('pelvis'), keypoints.index('left_hip')],
        [keypoints.index('pelvis'), keypoints.index('right_hip')],

        # Left Eye brow
        [keypoints.index('left_eye_brow1'), keypoints.index('left_eye_brow2')],
        [keypoints.index('left_eye_brow2'), keypoints.index('left_eye_brow3')],
        [keypoints.index('left_eye_brow3'), keypoints.index('left_eye_brow4')],
        [keypoints.index('left_eye_brow4'), keypoints.index('left_eye_brow5')],

        # Right Eye brow
        [keypoints.index('right_eye_brow1'),
         keypoints.index('right_eye_brow2')],
        [keypoints.index('right_eye_brow2'),
         keypoints.index('right_eye_brow3')],
        [keypoints.index('right_eye_brow3'),
         keypoints.index('right_eye_brow4')],
        [keypoints.index('right_eye_brow4'),
         keypoints.index('right_eye_brow5')],

        # Left Eye
        [keypoints.index('left_eye1'), keypoints.index('left_eye2')],
        [keypoints.index('left_eye2'), keypoints.index('left_eye3')],
        [keypoints.index('left_eye3'), keypoints.index('left_eye4')],
        [keypoints.index('left_eye4'), keypoints.index('left_eye5')],
        [keypoints.index('left_eye5'), keypoints.index('left_eye6')],
        [keypoints.index('left_eye6'), keypoints.index('left_eye1')],

        # Right Eye
        [keypoints.index('right_eye1'), keypoints.index('right_eye2')],
        [keypoints.index('right_eye2'), keypoints.index('right_eye3')],
        [keypoints.index('right_eye3'), keypoints.index('right_eye4')],
        [keypoints.index('right_eye4'), keypoints.index('right_eye5')],
        [keypoints.index('right_eye5'), keypoints.index('right_eye6')],
        [keypoints.index('right_eye6'), keypoints.index('right_eye1')],

        # Nose Vertical
        [keypoints.index('nose1'), keypoints.index('nose2')],
        [keypoints.index('nose2'), keypoints.index('nose3')],
        [keypoints.index('nose3'), keypoints.index('nose4')],

        # Nose Horizontal
        [keypoints.index('nose_middle'), keypoints.index('nose4')],
        [keypoints.index('left_nose_1'), keypoints.index('left_nose_2')],
        [keypoints.index('left_nose_1'), keypoints.index('nose_middle')],
        [keypoints.index('nose_middle'), keypoints.index('right_nose_1')],
        [keypoints.index('right_nose_2'), keypoints.index('right_nose_1')],

        # Mouth
        [keypoints.index('left_mouth_1'), keypoints.index('left_mouth_2')],
        [keypoints.index('left_mouth_2'), keypoints.index('left_mouth_3')],
        [keypoints.index('left_mouth_3'), keypoints.index('mouth_top')],
        [keypoints.index('mouth_top'), keypoints.index('right_mouth_3')],
        [keypoints.index('right_mouth_3'), keypoints.index('right_mouth_2')],
        [keypoints.index('right_mouth_2'), keypoints.index('right_mouth_1')],
        [keypoints.index('right_mouth_1'), keypoints.index('right_mouth_5')],
        [keypoints.index('right_mouth_5'), keypoints.index('right_mouth_4')],
        [keypoints.index('right_mouth_4'), keypoints.index('mouth_bottom')],
        [keypoints.index('mouth_bottom'), keypoints.index('left_mouth_4')],
        [keypoints.index('left_mouth_4'), keypoints.index('left_mouth_5')],
        [keypoints.index('left_mouth_5'), keypoints.index('left_mouth_1')],

        # Lips
        [keypoints.index('left_lip_1'), keypoints.index('left_lip_2')],
        [keypoints.index('left_lip_2'), keypoints.index('lip_top')],
        [keypoints.index('lip_top'), keypoints.index('right_lip_2')],
        [keypoints.index('right_lip_2'), keypoints.index('right_lip_1')],
        [keypoints.index('right_lip_1'), keypoints.index('right_lip_3')],
        [keypoints.index('right_lip_3'), keypoints.index('lip_bottom')],
        [keypoints.index('lip_bottom'), keypoints.index('left_lip_3')],
        [keypoints.index('left_lip_3'), keypoints.index('left_lip_1')],

        # Contour
        [keypoints.index('left_contour_1'), keypoints.index('left_contour_2')],
        [keypoints.index('left_contour_2'), keypoints.index('left_contour_3')],
        [keypoints.index('left_contour_3'), keypoints.index('left_contour_4')],
        [keypoints.index('left_contour_4'), keypoints.index('left_contour_5')],
        [keypoints.index('left_contour_5'), keypoints.index('left_contour_6')],
        [keypoints.index('left_contour_6'), keypoints.index('left_contour_7')],
        [keypoints.index('left_contour_7'), keypoints.index('left_contour_8')],
        [keypoints.index('left_contour_8'), keypoints.index('contour_middle')],

        [keypoints.index('contour_middle'),
         keypoints.index('right_contour_8')],
        [keypoints.index('right_contour_8'),
         keypoints.index('right_contour_7')],
        [keypoints.index('right_contour_7'),
         keypoints.index('right_contour_6')],
        [keypoints.index('right_contour_6'),
         keypoints.index('right_contour_5')],
        [keypoints.index('right_contour_5'),
         keypoints.index('right_contour_4')],
        [keypoints.index('right_contour_4'),
         keypoints.index('right_contour_3')],
        [keypoints.index('right_contour_3'),
         keypoints.index('right_contour_2')],
        [keypoints.index('right_contour_2'),
         keypoints.index('right_contour_1')],
    ]
    return kp_lines


FLIP_MAP = {}
for keyp_name in KEYPOINT_NAMES:
    if 'left' in keyp_name:
        FLIP_MAP[keyp_name] = keyp_name.replace('left', 'right')
    elif 'right' in keyp_name:
        FLIP_MAP[keyp_name] = keyp_name.replace('right', 'left')


def _create_flip_indices(names, flip_map):
    full_flip_map = flip_map.copy()
    full_flip_map.update({v: k for k, v in flip_map.items()})
    flipped_names = [i if i not in full_flip_map else full_flip_map[i]
                     for i in names]
    flip_indices = [names.index(i) for i in flipped_names]
    return torch.tensor(flip_indices)


ALL_CONNECTIONS = kp_connections(KEYPOINT_NAMES)
BODY_CONNECTIONS = [
    (KEYPOINT_NAMES.index(conn[0]), KEYPOINT_NAMES.index(conn[1]))
    for conn in CONNECTIONS
    if KEYPOINT_PARTS[conn[0]] == 'body' and
    KEYPOINT_PARTS[conn[1]] == 'body'
]
HAND_CONNECTIONS = [
    (KEYPOINT_NAMES.index(conn[0]), KEYPOINT_NAMES.index(conn[1]))
    for conn in CONNECTIONS
    if KEYPOINT_PARTS[conn[0]] == 'hand' and
    KEYPOINT_PARTS[conn[1]] == 'hand'
]
RIGHT_HAND_CONNECTIONS = [
    (RIGHT_HAND_KEYPOINT_NAMES.index(conn[0]),
     RIGHT_HAND_KEYPOINT_NAMES.index(conn[1]))
    for conn in CONNECTIONS
    if 'hand' in KEYPOINT_PARTS[conn[0]]and
    'hand' in KEYPOINT_PARTS[conn[1]] and
    ('right' in conn[0] or 'right' in conn[1])
]
FACE_CONNECTIONS = [
    (KEYPOINT_NAMES.index(conn[0]), KEYPOINT_NAMES.index(conn[1]))
    for conn in CONNECTIONS
    if 'face' in KEYPOINT_PARTS[conn[0]] and
    'face' in KEYPOINT_PARTS[conn[1]]
]
FLAME_CONNECTIONS = [
    (FLAME_KEYPOINT_NAMES.index(conn[0]),
     FLAME_KEYPOINT_NAMES.index(conn[1]))
    for conn in CONNECTIONS
    if 'flame' in KEYPOINT_PARTS[conn[0]] and
    'flame' in KEYPOINT_PARTS[conn[1]]
]
VMD_CONNECTIONS = [
    (KEYPOINT_NAMES.index(conn[0]), KEYPOINT_NAMES.index(conn[1]))
    for conn in CONNECTIONS
    if KEYPOINT_PARTS[conn[0]] == 'vmd' and
    KEYPOINT_PARTS[conn[1]] == 'vmd'
]
HEAD_CONNECTIONS = FACE_CONNECTIONS
FLIP_INDS = np.asarray(
    _create_flip_indices(KEYPOINT_NAMES, FLIP_MAP))
Keypoints2D.FLIP_INDS = FLIP_INDS
Keypoints2D.CONNECTIONS = np.asarray(ALL_CONNECTIONS).reshape(-1, 2)


class Keypoints3D(Keypoints2D):
    def __init__(self, *args, **kwargs):
        super(Keypoints3D, self).__init__(*args, **kwargs)

    def rotate(self, rot=0, *args, **kwargs):
        kp = self.smplx_keypoints.copy()
        conf = self.conf.copy().reshape(-1, 1)

        if rot != 0:
            R = np.array([[np.cos(np.deg2rad(-rot)),
                           -np.sin(np.deg2rad(-rot)), 0],
                          [np.sin(np.deg2rad(-rot)),
                           np.cos(np.deg2rad(-rot)), 0],
                          [0, 0, 1]], dtype=np.float32)
            kp = np.dot(kp, R.T)

        kp = np.concatenate([kp, conf], axis=1).astype(np.float32)

        keypoints = type(self)(kp, size=self.size)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.rotate(rot=rot, *args, **kwargs)
            keypoints.add_field(k, v)
        self.add_field('rot', kwargs.get('rot', 0))
        return keypoints

    def crop(self, center, scale, crop_size=224, *args, **kwargs):
        for k, v in self.extra_fields.items():
            if isinstance(v, GenericTarget):
                v = v.crop(center=center, scale=scale,
                           crop_size=crop_size, *args, **kwargs)
        return self

    def center_by_keyp(self, keyp_name='pelvis'):
        keyp_idx = KEYPOINT_NAMES.index(keyp_name)
        self.smplx_keypoints -= self.smplx_keypoints[[keyp_idx]]

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT,):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT implemented")

        flip_inds = type(self).FLIP_INDS
        if torch.is_tensor(self.smplx_keypoints):
            flipped_data = torch.cat([self.smplx_keypoints,
                                      self.conf.unsqueeze(dim=-1)],
                                     dim=-1)

            num_joints = flipped_data.shape[0]
            #  flipped_data[torch.arange(num_joints)] = torch.index_select(
            #  flipped_data, 0, flip_inds[:num_joints])
            flipped_data[np.arange(num_joints)] = flipped_data[
                flip_inds[:num_joints]]
            #  width = self.size[0]
            #  TO_REMOVE = 1
            # Flip x coordinates
            #  flipped_data[..., 0] = width - flipped_data[..., 0] - TO_REMOVE
            flipped_data[..., :, self.flip_axis] *= (-1)

            #  Maintain COCO convention that if visibility == 0, then x, y = 0
            #  inds = flipped_data[..., 2] == 0
            #  flipped_data[inds] = 0
        else:
            flipped_data = np.concatenate([self.smplx_keypoints,
                                           self.conf[..., np.newaxis]], axis=-1)

            num_joints = flipped_data.shape[0]
            #  flipped_data[torch.arange(num_joints)] = torch.index_select(
            #  flipped_data, 0, flip_inds[:num_joints])
            flipped_data[np.arange(num_joints)] = flipped_data[
                flip_inds[:num_joints]]
            #  width = self.size[0]
            #  TO_REMOVE = 1
            # Flip x coordinates
            #  flipped_data[..., 0] = width - flipped_data[..., 0] - TO_REMOVE
            flipped_data[..., :, self.flip_axis] *= (-1)

        keypoints = type(self)(flipped_data, self.size)
        for k, v in self.extra_fields.items():
            if isinstance(v, GenericTarget):
                v = v.transpose(method)
            keypoints.add_field(k, v)
        self.add_field('is_flipped', True)

        return keypoints


OPENPOSE_JOINTS = [
    'nose', 'neck',
    'right_shoulder', 'right_elbow', 'right_wrist',
    'left_shoulder', 'left_elbow', 'left_wrist',
    'pelvis',
    'right_hip', 'right_knee', 'right_ankle',
    'left_hip', 'left_knee', 'left_ankle',
    'right_eye', 'left_eye', 'right_ear', 'left_ear',
    'left_wrist',
    'left_thumb1', 'left_thumb2', 'left_thumb3', 'left_thumb',
    'left_index1', 'left_index2', 'left_index3', 'left_index',
    'left_middle1', 'left_middle2', 'left_middle3', 'left_middle',
    'left_ring1', 'left_ring2', 'left_ring3', 'left_ring',
    'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_pinky',
    'right_wrist',
    'right_thumb1', 'right_thumb2', 'right_thumb3', 'right_thumb',
    'right_index1', 'right_index2', 'right_index3', 'right_index',
    'right_middle1', 'right_middle2', 'right_middle3', 'right_middle',
    'right_ring1', 'right_ring2', 'right_ring3', 'right_ring',
    'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_pinky',
    # Face contour
    'right_contour_1',
    'right_contour_2',
    'right_contour_3',
    'right_contour_4',
    'right_contour_5',
    'right_contour_6',
    'right_contour_7',
    'right_contour_8',
    'contour_middle',
    'left_contour_8',
    'left_contour_7',
    'left_contour_6',
    'left_contour_5',
    'left_contour_4',
    'left_contour_3',
    'left_contour_2',
    'left_contour_1',
    # Eye brows
    'right_eye_brow1',
    'right_eye_brow2',
    'right_eye_brow3',
    'right_eye_brow4',
    'right_eye_brow5',
    'left_eye_brow5',
    'left_eye_brow4',
    'left_eye_brow3',
    'left_eye_brow2',
    'left_eye_brow1',
    'nose1',
    'nose2',
    'nose3',
    'nose4',
    'right_nose_2',
    'right_nose_1',
    'nose_middle',
    'left_nose_1',
    'left_nose_2',
    'right_eye1',
    'right_eye2',
    'right_eye3',
    'right_eye4',
    'right_eye5',
    'right_eye6',
    'left_eye4',
    'left_eye3',
    'left_eye2',
    'left_eye1',
    'left_eye6',
    'left_eye5',
    'right_mouth_1',
    'right_mouth_2',
    'right_mouth_3',
    'mouth_top',
    'left_mouth_3',
    'left_mouth_2',
    'left_mouth_1',
    'left_mouth_5',  # 59 in OpenPose output
    'left_mouth_4',  # 58 in OpenPose output
    'mouth_bottom',
    'right_mouth_4',
    'right_mouth_5',
    'right_lip_1',
    'right_lip_2',
    'lip_top',
    'left_lip_2',
    'left_lip_1',
    'left_lip_3',
    'lip_bottom',
    'right_lip_3',
]

FEET_KEYPS_NAMES = ['left_big_toe', 'left_small_toe', 'left_heel',
                    'right_big_toe', 'right_small_toe', 'right_heel']
OPENPOSE_JOINTS25 = deepcopy(OPENPOSE_JOINTS)
start = 19
for feet_name in FEET_KEYPS_NAMES:
    OPENPOSE_JOINTS25.insert(start, feet_name)
    start += 1

MPII_JOINTS = [
    'right_ankle',
    'right_knee',
    'right_hip',
    'left_hip',
    'left_knee',
    'left_ankle',
    'pelvis',
    'thorax',
    'upper_neck',
    'head_top',
    'right_wrist',
    'right_elbow',
    'right_shoulder',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    # Hand joints
    'left_wrist',
    'left_thumb1', 'left_thumb2', 'left_thumb3', 'left_thumb',
    'left_index1', 'left_index2', 'left_index3', 'left_index',
    'left_middle1', 'left_middle2', 'left_middle3', 'left_middle',
    'left_ring1', 'left_ring2', 'left_ring3', 'left_ring',
    'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_pinky',
    'right_wrist',
    'right_thumb1', 'right_thumb2', 'right_thumb3', 'right_thumb',
    'right_index1', 'right_index2', 'right_index3', 'right_index',
    'right_middle1', 'right_middle2', 'right_middle3', 'right_middle',
    'right_ring1', 'right_ring2', 'right_ring3', 'right_ring',
    'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_pinky',
]


FFHQ_KEYPOINTS = [
    'global',
    'neck',
    'jaw',
    'left_eye',
    'right_eye',
    'right_eye_brow1',
    'right_eye_brow2',
    'right_eye_brow3',
    'right_eye_brow4',
    'right_eye_brow5',
    'left_eye_brow5',
    'left_eye_brow4',
    'left_eye_brow3',
    'left_eye_brow2',
    'left_eye_brow1',
    'nose1',
    'nose2',
    'nose3',
    'nose4',
    'right_nose_2',
    'right_nose_1',
    'nose_middle',
    'left_nose_1',
    'left_nose_2',
    'right_eye1',
    'right_eye2',
    'right_eye3',
    'right_eye4',
    'right_eye5',
    'right_eye6',
    'left_eye4',
    'left_eye3',
    'left_eye2',
    'left_eye1',
    'left_eye6',
    'left_eye5',
    'right_mouth_1',
    'right_mouth_2',
    'right_mouth_3',
    'mouth_top',
    'left_mouth_3',
    'left_mouth_2',
    'left_mouth_1',
    'left_mouth_5',  # 59 in OpenPose output
    'left_mouth_4',  # 58 in OpenPose output
    'mouth_bottom',
    'right_mouth_4',
    'right_mouth_5',
    'right_lip_1',
    'right_lip_2',
    'lip_top',
    'left_lip_2',
    'left_lip_1',
    'left_lip_3',
    'lip_bottom',
    'right_lip_3',
    # Face contour
    'right_contour_1',
    'right_contour_2',
    'right_contour_3',
    'right_contour_4',
    'right_contour_5',
    'right_contour_6',
    'right_contour_7',
    'right_contour_8',
    'contour_middle',
    'left_contour_8',
    'left_contour_7',
    'left_contour_6',
    'left_contour_5',
    'left_contour_4',
    'left_contour_3',
    'left_contour_2',
    'left_contour_1',
]


COCO_KEYPOINTS = ['nose',
                  'neck',
                  'left_eye',
                  'right_eye',
                  'left_ear',
                  'right_ear',
                  'left_shoulder',
                  'right_shoulder',
                  'left_elbow',
                  'right_elbow',
                  'left_wrist',
                  'right_wrist',
                  'left_hip',
                  'right_hip',
                  'left_knee',
                  'right_knee',
                  'left_ankle',
                  'right_ankle',
                  'pelvis']


THREEDPW_JOINTS = [
    'nose',
    'neck',
    'right_shoulder',
    'right_elbow',
    'right_wrist',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    'right_hip',
    'right_knee',
    'right_ankle',
    'left_hip',
    'left_knee',
    'left_ankle',
    'right_eye',
    'left_eye',
    'right_ear',
    'left_ear',
]


POSETRACK_KEYPOINT_NAMES = [
    'nose',
    'neck',
    'head_top',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
    'pelvis'
]

AICH_KEYPOINT_NAMES = [
    'right_shoulder',
    'right_elbow',
    'right_wrist',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    'right_hip',
    'right_knee',
    'right_ankle',
    'left_hip',
    'left_knee',
    'left_ankle',
    'head_top',
    'neck',
    'pelvis'
]

NAMES_DICT = {
    'coco': COCO_KEYPOINTS,
    'openpose19': OPENPOSE_JOINTS[:19],
    'openpose19+hands': OPENPOSE_JOINTS[19:19 + 2 * 21],
    'openpose19+hands+face': OPENPOSE_JOINTS,
    'openpose25': OPENPOSE_JOINTS25[:25],
    'openpose25+hands': OPENPOSE_JOINTS25[:25 + 2 * 21],
    'openpose25+hands+face': OPENPOSE_JOINTS25
}

SPIN_KEYPOINT_NAMES = [
    'right_ankle',
    'right_knee',
    'right_hip',
    'left_hip',
    'left_knee',
    'left_ankle',
    'right_wrist',
    'right_elbow',
    'right_shoulder',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    'neck',
    'head_top',
    'pelvis',
    'thorax',
    'spine',
    'h36m_jaw',
    'h36m_head',
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
]

SPINX_KEYPOINT_NAMES = [
    'right_ankle',
    'right_knee',
    'right_hip',
    'left_hip',
    'left_knee',
    'left_ankle',
    'right_wrist',
    'right_elbow',
    'right_shoulder',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    'neck',
    'head_top',
    'pelvis',
    'thorax',
    'spine',
    'h36m_jaw',
    'h36m_head',
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
] + OPENPOSE_JOINTS25[25:]

PANOPTIC_KEYPOINT_NAMES = [
    'neck',
    'nose',
    'pelvis',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    'left_hip',
    'left_knee',
    'left_ankle',
    'right_shoulder',
    'right_elbow',
    'right_wrist',
    'right_hip',
    'right_knee',
    'right_ankle',
    'left_eye',
    'left_ear',
    'right_eye',
    'right_ear',
]
PANOPTIC_KEYPOINT_NAMES += (OPENPOSE_JOINTS[19:19 + 2 * 21] +
                            OPENPOSE_JOINTS[19 + 2 * 21 + 17:] +
                            OPENPOSE_JOINTS[19 + 2 * 21:19 + 2 * 21 + 17]
                            )

FREIHAND_NAMES = [
    'right_wrist',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3',
    'right_thumb',
    'right_index1',
    'right_index2',
    'right_index3',
    'right_index',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_middle',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_ring',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_pinky',
]

LSP_NAMES = [
    'right_ankle',
    'right_knee',
    'right_hip',
    'left_hip',
    'left_knee',
    'left_ankle',
    'right_wrist',
    'right_elbow',
    'right_shoulder',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    'neck',
    'head_top',
]

RAW_H36M_NAMES = [
    'pelvis',
    'left_hip',
    'left_knee',
    'left_ankle',
    'right_hip',
    'right_knee',
    'right_ankle',
    'spine',
    'neck',  # 'thorax',
    'neck/nose',
    'head',  # 'head_h36m',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    'right_shoulder',
    'right_elbow',
    'right_wrist'
]

H36M_NAMES = [
    'right_ankle',
    'right_knee',
    'right_hip',
    'left_hip',
    'left_knee',
    'left_ankle',
    'right_wrist',
    'right_elbow',
    'right_shoulder',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    'neck',
    'top_of_head_(lsp)',
    'pelvis_(mpii)',
    'thorax_(mpii)',
    'spine_(h36m)',
    'jaw_(h36m)',
    'head',
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear'
]


def body_model_to_dset(model_type='smplx', dset='coco', joints_to_ign=None,
                       use_face_contour=False, **kwargs):
    if joints_to_ign is None:
        joints_to_ign = []

    mapping = {}
    if model_type == 'smplx':
        keypoint_names = KEYPOINT_NAMES
    elif model_type == 'mano':
        keypoint_names = MANO_NAMES

    if dset == 'coco':
        dset_keyp_names = COCO_KEYPOINTS
    elif dset == 'openpose19':
        dset_keyp_names = OPENPOSE_JOINTS[:19]
    elif dset == 'openpose19+hands':
        dset_keyp_names = OPENPOSE_JOINTS[19:19 + 2 * 21]
    elif dset == 'openpose19+hands+face':
        dset_keyp_names = OPENPOSE_JOINTS
    elif dset == 'openpose25':
        dset_keyp_names = OPENPOSE_JOINTS25[:25]
    elif dset == 'openpose25+hands':
        dset_keyp_names = OPENPOSE_JOINTS25[:25 + 2 * 21]
    elif dset == 'openpose25+hands+face':
        dset_keyp_names = OPENPOSE_JOINTS25
    elif dset == 'freihand':
        dset_keyp_names = FREIHAND_NAMES
    else:
        raise ValueError('Unknown dset dataset: {}'.format(dset))

    for idx, name in enumerate(dset_keyp_names):
        if 'contour' in name and not use_face_contour:
            continue
        if name in keypoint_names:
            mapping[idx] = keypoint_names.index(name)

    dset_keyp_idxs = np.array(list(mapping.keys()), dtype=np.long)
    model_keyps_idxs = np.array(list(mapping.values()), dtype=np.long)

    return dset_keyp_idxs, model_keyps_idxs


def dset_to_body_model(model_type='smplx', dset='coco', joints_to_ign=None,
                       use_face_contour=False, **kwargs):
    if joints_to_ign is None:
        joints_to_ign = []

    mapping = {}

    if dset == 'coco':
        dset_keyp_names = COCO_KEYPOINTS
    elif dset == 'openpose19':
        dset_keyp_names = OPENPOSE_JOINTS[:19]
    elif dset == 'openpose19+hands':
        dset_keyp_names = OPENPOSE_JOINTS[19:19 + 2 * 21]
    elif dset == 'openpose19+hands':
        dset_keyp_names = OPENPOSE_JOINTS[19:19 + 2 * 21]
    elif dset == 'openpose25':
        dset_keyp_names = OPENPOSE_JOINTS25[:25]
    elif dset == 'openpose25+hands':
        dset_keyp_names = OPENPOSE_JOINTS25[:25 + 2 * 21]
    elif dset == 'openpose25+hands+face':
        dset_keyp_names = OPENPOSE_JOINTS25
    elif dset == 'posetrack':
        dset_keyp_names = POSETRACK_KEYPOINT_NAMES
    elif dset == 'mpii':
        dset_keyp_names = MPII_JOINTS
    elif dset == 'left-mpii-hands':
        dset_keyp_names = MPII_JOINTS[-2 * 21:-21]
    elif dset == 'right-mpii-hands':
        dset_keyp_names = MPII_JOINTS[-21:]
    elif dset == 'aich':
        dset_keyp_names = AICH_KEYPOINT_NAMES
    elif dset == 'spin':
        dset_keyp_names = SPIN_KEYPOINT_NAMES
    elif dset == 'spinx':
        dset_keyp_names = SPINX_KEYPOINT_NAMES
    elif dset == 'panoptic':
        dset_keyp_names = PANOPTIC_KEYPOINT_NAMES
    elif dset == 'mano':
        dset_keyp_names = MANO_NAMES
    elif dset == '3dpw':
        dset_keyp_names = THREEDPW_JOINTS
    elif dset == 'freihand':
        dset_keyp_names = FREIHAND_NAMES
    elif dset == 'h36m':
        dset_keyp_names = H36M_NAMES
    elif dset == 'raw_h36m':
        dset_keyp_names = RAW_H36M_NAMES
    elif dset == 'ffhq':
        dset_keyp_names = FFHQ_KEYPOINTS
    elif dset == 'lsp':
        dset_keyp_names = LSP_NAMES
    else:
        raise ValueError('Unknown dset dataset: {}'.format(dset))

    for idx, name in enumerate(KEYPOINT_NAMES):
        if 'contour' in name and not use_face_contour:
            continue
        if name in dset_keyp_names:
            mapping[idx] = dset_keyp_names.index(name)

    model_keyps_idxs = np.array(list(mapping.keys()), dtype=np.long)
    dset_keyps_idxs = np.array(list(mapping.values()), dtype=np.long)

    return dset_keyps_idxs, model_keyps_idxs
