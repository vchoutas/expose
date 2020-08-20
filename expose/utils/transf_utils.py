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

from typing import Tuple
import numpy as np

import PIL.Image as pil_img
import time
from loguru import logger
import cv2

from .typing_utils import Array


def get_transform(
    center: Array, scale: float,
    res: Tuple[int],
    rot: float = 0
) -> Array:
    """
    General image processing functions
    """
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3), dtype=np.float32)
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3), dtype=np.float32)
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t.astype(np.float32)


# Consistent with https://github.com/bearpaw/pytorch-pose
# and the lua version of https://github.com/anewell/pose-hg-train
def transform(pt, center, scale, res, invert=0, rot=0):
    # Transform pixel location to different reference
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def crop(img, center, scale, res, rot=0, dtype=np.float32):
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(transform([res[0] + 1, res[1] + 1],
                            center, scale, res, invert=1)) - 1
    # size of cropped image
    #  crop_shape = [br[1] - ul[1], br[0] - ul[0]]
    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)

    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_shape = list(map(int, new_shape))
    new_img = np.zeros(new_shape, dtype=img.dtype)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]

    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    # Range to sample from original image
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]
            ] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    #  pixel_scale = 1.0 if new_img.max() > 1.0 else 255
    #  resample = pil_img.BILINEAR
    if not rot == 0:
        new_H, new_W, _ = new_img.shape

        rotn_center = (new_W / 2.0, new_H / 2.0)
        M = cv2.getRotationMatrix2D(rotn_center, rot, 1.0).astype(np.float32)

        new_img = cv2.warpAffine(new_img, M, tuple(new_shape[:2]),
                                 cv2.INTER_LINEAR_EXACT)
        new_img = new_img[pad:new_H - pad, pad:new_W - pad]

    output = cv2.resize(new_img, tuple(res), interpolation=cv2.INTER_LINEAR)
    return output.astype(np.float32)
