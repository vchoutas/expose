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
from typing import List, Optional
import functools
import glob
import datetime
from matplotlib.pyplot import ylabel
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
from collections import OrderedDict, defaultdict
from loguru import logger
import cv2
import argparse
import time
import open3d as o3d
from tqdm import tqdm
from threadpoolctl import threadpool_limits
import PIL.Image as pil_img
import matplotlib.pyplot as plt
import json

import torch
import torch.utils.data as dutils
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.transforms import Compose, Normalize, ToTensor

from expose.data.datasets import ImageFolder, ImageFolderWithBoxes

from expose.data.targets.image_list import to_image_list
from expose.utils.checkpointer import Checkpointer
import expose.data.utils.bbox as bboxutils

from expose.data.build import collate_batch
from expose.data.transforms import build_transforms

from expose.models.smplx_net import SMPLXNet
from expose.config import cfg
from expose.config.cmd_parser import set_face_contour
from expose.utils.plot_utils import HDRenderer

from expose.data.targets import Keypoints2D
from expose.data.targets.keypoints import body_model_to_dset, ALL_CONNECTIONS, KEYPOINT_NAMES, BODY_CONNECTIONS

# 指数表記なし、有効小数点桁数6、30を超えると省略あり、一行の文字数200
np.set_printoptions(suppress=True, precision=6, threshold=30, linewidth=200)

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    arg_formatter = argparse.ArgumentDefaultsHelpFormatter
    description = 'PyTorch SMPL-X Regressor Demo'
    parser = argparse.ArgumentParser(formatter_class=arg_formatter,
                                     description=description)

    parser.add_argument('--folder-path', type=str, dest='folder_path',
                        help='The folder with images that will be processed')

    args = parser.parse_args()

    original_width = 0
    original_height = 0

    for img_path in glob.glob(osp.join(args.folder_path, "**/hd_overlay.png")):
        im = cv2.imread(img_path)
        original_height, original_width, _ = im.shape
        break

    fmt = cv2.VideoWriter_fourcc(*'mp4v')
    
    logger.info(f"■ OpenCV -----")
    # メッシュの結合
    writer = cv2.VideoWriter(osp.join("C:/MMD/expose_mmd/samples/girl2/girl_46949_mp4/face", 'faces.mp4'), fmt, 30, (original_width, original_height))
    for img_path in glob.glob(osp.join("C:/MMD/expose_mmd/samples/girl2/girl_46949_mp4/face", "capture_*.png")):
        writer.write(cv2.imread(img_path))
    writer.release()

    logger.info(f"■ メッシュ -----")
    # メッシュの結合
    writer = cv2.VideoWriter(osp.join(args.folder_path, 'hd_overlay.mp4'), fmt, 30, (original_width, original_height))
    for img_path in glob.glob(osp.join(args.folder_path, "**/hd_overlay.png")):
        writer.write(cv2.imread(img_path))
    writer.release()

    logger.info(f"■ joint -----")
    # JOINTの結合
    writer = cv2.VideoWriter(osp.join(args.folder_path, 'joints.mp4'), fmt, 30, (1500, 1500))
    for img_path in glob.glob(osp.join(args.folder_path, "**/*_joints.png")):
        writer.write(cv2.imread(img_path))
    writer.release()

    logger.info(f"■ face(2d) -----")
    # faceの結合
    writer = cv2.VideoWriter(osp.join(args.folder_path, 'face.mp4'), fmt, 30, (1500, 1500))
    for img_path in glob.glob(osp.join(args.folder_path, "**/face.png")):
        writer.write(cv2.imread(img_path))
    writer.release()

    logger.info(f"■ face(3d) -----")
    # face3dの結合
    writer = cv2.VideoWriter(osp.join(args.folder_path, 'face3d.mp4'), fmt, 30, (1500, 1500))
    for img_path in glob.glob(osp.join(args.folder_path, "**/face3d.png")):
        writer.write(cv2.imread(img_path))
    writer.release()

    cv2.destroyAllWindows()
