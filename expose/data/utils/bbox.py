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


from typing import Dict, Tuple

import torch
import numpy as np
from loguru import logger

from expose.utils.typing_utils import Tensor


def points_to_bbox(
        points: Tensor,
        bbox_scale_factor: float = 1.0) -> Tuple[Tensor, Tensor]:

    min_coords, _ = torch.min(points, dim=1)
    xmin, ymin = min_coords[:, 0], min_coords[:, 1]
    max_coords, _ = torch.max(points, dim=1)
    xmax, ymax = max_coords[:, 0], max_coords[:, 1]

    center = torch.stack(
        [xmax + xmin, ymax + ymin], dim=-1) * 0.5

    width = (xmax - xmin)
    height = (ymax - ymin)

    # Convert the bounding box to a square box
    size = torch.max(width, height) * bbox_scale_factor

    return center, size


def center_size_to_bbox(center: Tensor, size: Tensor) -> Tensor:
    xmin = center[:, 0] - size * 0.5
    ymin = center[:, 1] - size * 0.5

    xmax = center[:, 0] + size * 0.5
    ymax = center[:, 1] + size * 0.5

    return torch.stack([xmin, ymin, xmax, ymax], axis=-1)


def keyps_to_bbox(keypoints, conf, img_size=None, clip_to_img=False,
                  min_valid_keypoints=6, scale=1.0):
    valid_keypoints = keypoints[conf > 0]
    if len(valid_keypoints) < min_valid_keypoints:
        return None

    xmin, ymin = np.amin(valid_keypoints, axis=0)
    xmax, ymax = np.amax(valid_keypoints, axis=0)
    # Clip to the image
    if img_size is not None and clip_to_img:
        H, W, _ = img_size
        xmin = np.clip(xmin, 0, W)
        xmax = np.clip(xmax, 0, W)
        ymin = np.clip(ymin, 0, H)
        ymax = np.clip(ymax, 0, H)

    width = (xmax - xmin) * scale
    height = (ymax - ymin) * scale

    x_center = 0.5 * (xmax + xmin)
    y_center = 0.5 * (ymax + ymin)
    xmin = x_center - 0.5 * width
    xmax = x_center + 0.5 * width
    ymin = y_center - 0.5 * height
    ymax = y_center + 0.5 * height

    bbox = np.stack([xmin, ymin, xmax, ymax], axis=0).astype(np.float32)
    if bbox_area(bbox) > 0:
        return bbox
    else:
        return None


def bbox_to_center_scale(bbox, dset_scale_factor=1.0, ref_bbox_size=200):
    if bbox is None:
        return None, None, None
    bbox = bbox.reshape(-1)
    bbox_size = dset_scale_factor * max(
        bbox[2] - bbox[0], bbox[3] - bbox[1])
    scale = bbox_size / ref_bbox_size
    center = np.stack(
        [(bbox[0] + bbox[2]) * 0.5,
         (bbox[1] + bbox[3]) * 0.5]).astype(np.float32)
    return center, scale, bbox_size


def scale_to_bbox_size(scale, ref_bbox_size=200):
    return scale * ref_bbox_size


def bbox_area(bbox):
    if torch.is_tensor(bbox):
        if bbox is None:
            return 0.0
        xmin, ymin, xmax, ymax = torch.split(bbox.reshape(-1, 4), 1, dim=1)
        return torch.abs((xmax - xmin) * (ymax - ymin)).squeeze(dim=-1)
    else:
        if bbox is None:
            return 0.0
        xmin, ymin, xmax, ymax = np.split(bbox.reshape(-1, 4), 4, axis=1)
        return np.abs((xmax - xmin) * (ymax - ymin))


def bbox_to_wh(bbox):
    if bbox is None:
        return (0.0, 0.0)
    xmin, ymin, xmax, ymax = np.split(bbox.reshape(-1, 4), 4, axis=1)
    return xmax - xmin, ymax - ymin


def bbox_iou(bbox1, bbox2, epsilon=1e-9):
    ''' Computes IoU between bounding boxes

        Parameters
        ----------
            bbox1: torch.Tensor or np.ndarray
                A Nx4 array of bounding boxes in xyxy format
            bbox2: torch.Tensor or np.ndarray
                A Nx4 array of bounding boxes in xyxy format
        Returns
        -------
            ious: torch.Tensor or np.ndarray
                A N dimensional array that contains the IoUs between bounding
                box pairs
    '''
    if torch.is_tensor(bbox1):
        # B
        bbox1 = bbox1.reshape(-1, 4)
        bbox2 = bbox2.reshape(-1, 4)

        # Should be B
        left_top = torch.max(bbox1[:, :2], bbox2[:, :2])
        right_bottom = torch.min(bbox1[:, 2:], bbox2[:, 2:])

        wh = (right_bottom - left_top).clamp(min=0)

        area1, area2 = bbox_area(bbox1), bbox_area(bbox2)

        isect = wh[:, 0] * wh[:, 1].reshape(bbox1.shape[0])
        union = (area1 + area2 - isect).reshape(bbox1.shape[0])
    else:
        bbox1 = bbox1.reshape(4)
        bbox2 = bbox2.reshape(4)

        left_top = np.maximum(bbox1[:2], bbox2[:2])
        right_bottom = np.minimum(bbox1[2:], bbox2[2:])

        wh = right_bottom - left_top

        area1, area2 = bbox_area(bbox1), bbox_area(bbox2)

        isect = np.clip(wh[0] * wh[1], 0, float('inf'))
        union = (area1 + area2 - isect).squeeze()

    return isect / (union + epsilon)
