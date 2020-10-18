# -*- coding: utf-8 -*-
from os import name
import numpy as np
import argparse
import csv
import cv2
import pathlib
import PIL.Image as pil_img
import matplotlib.pyplot as plt
from miu.mmd.VmdWriter import VmdWriter
import os.path as osp
import json
import glob
import datetime
from expose.data.targets.keypoints import ALL_CONNECTIONS, KEYPOINT_NAMES, FACE_CONNECTIONS

from miu.module.MMath import MMatrix4x4, MQuaternion, MVector3D
from miu.mmd.VmdData import VmdBoneFrame, VmdMotion, VmdShowIkFrame, VmdInfoIk
from miu.mmd.PmxData import PmxModel, Bone, Vertex
from miu.utils.MServiceUtils import get_file_encoding, calc_global_pos, separate_local_qq
from miu.utils.MLogger import MLogger

logger = MLogger(__name__, level=MLogger.DEBUG)
SCALE_MIKU = 0.0625

def execute(cmd_args):
    folder_path = cmd_args.folder_path

    # 動画上の関節位置
    for fno, joints_path in enumerate(glob.glob(osp.join(folder_path, '**/*_joints.json'))):

        frame_joints = {}
        with open(joints_path, 'r') as f:
            frame_joints = json.load(f)
        
        original_width = frame_joints["image"]["width"]
        original_height = frame_joints["image"]["height"]

        # 描画設定
        fig = plt.figure(figsize=(15,15),dpi=100)
        # 3DAxesを追加
        ax = fig.add_subplot(111)

        # ジョイント出力                    
        ax.set_xlim(1200, 1600)
        ax.set_ylim(-900, -500)
        ax.set(xlabel='x', ylabel='y')

        xs = []
        ys = []

        xall = []
        yall = []

        for j3d_from_idx, j3d_to_idx in FACE_CONNECTIONS:
            jfname = KEYPOINT_NAMES[j3d_from_idx]
            jtname = KEYPOINT_NAMES[j3d_to_idx]
            
            fx = frame_joints["proj_joints"][jfname]['x'] * 1.5
            fy = -frame_joints["proj_joints"][jfname]['y'] * 1.5
            tx = frame_joints["proj_joints"][jtname]['x'] * 1.5
            ty = -frame_joints["proj_joints"][jtname]['y'] * 1.5

            xs = [fx, tx]
            ys = [fy, ty]

            xall.append(fx)
            yall.append(fy)

            ax.plot(xs, ys, marker="o", ms=2, c="#0000FF")
            ax.text(fx - 10, fy + 10, jfname, size=6)
        
        plt.savefig(osp.join(str(pathlib.Path(joints_path).parent), f'face.png'))
        plt.close()

if __name__ == '__main__':
    arg_formatter = argparse.ArgumentDefaultsHelpFormatter
    description = 'VMD Output'

    parser = argparse.ArgumentParser(formatter_class=arg_formatter,
                                     description=description)

    parser.add_argument('--folder-path', type=str, dest='folder_path', help='The folder with joint json')
    parser.add_argument('--bone-csv-path', type=str, dest='bone_csv_path', help='The csv file pmx born')
    parser.add_argument('--verbose', type=int, dest='verbose', default=20, help='The csv file pmx born')

    cmd_args = parser.parse_args()

    MLogger.initialize(level=cmd_args.verbose, is_file=True)

    execute(cmd_args)
