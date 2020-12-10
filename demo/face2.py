# -*- coding: utf-8 -*-
import os
import argparse
import glob
import re
import json
import csv
import datetime
import numpy as np
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

from mmd.utils.MLogger import MLogger
from mmd.utils.MServiceUtils import sort_by_numeric
from lighttrack.visualizer.detection_visualizer import draw_bbox

from mmd.mmd.VmdWriter import VmdWriter
from mmd.module.MMath import MQuaternion, MVector3D, MVector2D
from mmd.mmd.VmdData import VmdBoneFrame, VmdMorphFrame, VmdMotion, VmdShowIkFrame, VmdInfoIk, OneEuroFilter
from mmd.mmd.PmxData import PmxModel, Bone, Vertex, Bdef1
from mmd.utils.MServiceUtils import get_file_encoding, calc_global_pos, separate_local_qq

SCALE_MIKU = 0.06

logger = MLogger(__name__)

def execute(args):
    try:
        logger.info('モーション生成処理開始: %s', args.img_dir, decoration=MLogger.DECORATION_BOX)

        if not os.path.exists(args.img_dir):
            logger.error("指定された処理用ディレクトリが存在しません。: %s", args.img_dir, decoration=MLogger.DECORATION_BOX)
            return False

        if not os.path.exists(os.path.join(args.img_dir, "ordered")):
            logger.error("指定された順番ディレクトリが存在しません。\n順番指定が完了していない可能性があります。: %s", \
                         os.path.join(args.img_dir, "ordered"), decoration=MLogger.DECORATION_BOX)
            return False

        motion_dir_path = os.path.join(args.img_dir, "motion")
        os.makedirs(motion_dir_path, exist_ok=True)
        
        # 全人物分の順番別フォルダ
        ordered_person_dir_pathes = sorted(glob.glob(os.path.join(args.img_dir, "ordered", "*")), key=sort_by_numeric)

        smooth_pattern = re.compile(r'^smooth_(\d+)\.')

        for oidx, ordered_person_dir_path in enumerate(ordered_person_dir_pathes):    
            logger.info("【No.%s】モーション生成開始", f"{oidx:03}", decoration=MLogger.DECORATION_BOX)

            smooth_json_pathes = sorted(glob.glob(os.path.join(ordered_person_dir_path, "smooth_*.json")), key=sort_by_numeric)

            model = PmxModel()
            model.name = "Output Demo"
            motion = VmdMotion()
        
            for smooth_json_path in tqdm(smooth_json_pathes, desc=f"No.{oidx:03} ... "):
                m = smooth_pattern.match(os.path.basename(smooth_json_path))
                if m:
                    # キーフレの場所を確定（間が空く場合もある）
                    fno = int(m.groups()[0])

                    frame_joints = {}
                    with open(smooth_json_path, 'r', encoding='utf-8') as f:
                        frame_joints = json.load(f)
                    
                    # for k, v in frame_joints["left_hand_pose"].items():
                    #     qq = MQuaternion.fromAxes(MVector3D(float(v["xAxis"]["x"]), float(v["xAxis"]["y"]), float(v["xAxis"]["z"])), \
                    #                               MVector3D(float(v["yAxis"]["x"]), float(v["yAxis"]["y"]), float(v["yAxis"]["z"])), \
                    #                               MVector3D(float(v["zAxis"]["x"]), float(v["zAxis"]["y"]), float(v["zAxis"]["z"])))
                    #     bone_name_dict = {
                    #         "0": "左手首", "1": "左人指１", "2": "左人指２", "3": "左人指３", \
                    #         "4": "左中指１", "5": "左中指２", "6": "左中指３", \
                    #         "7": "左薬指１", "8": "左薬指２", "9": "左薬指３", \
                    #         "10": "左小指１", "11": "左小指２", "12": "左小指３", \
                    #         "13": "左親指１", "14": "左親指２", 
                    #     }
                        
                    #     bone_name = bone_name_dict[k]
                    #     bf = VmdBoneFrame(fno)
                    #     bf.rotation = qq
                    #     bf.set_name(bone_name)
                    #     motion.regist_bf(bf, bone_name, fno)                     

                    # 描画設定
                    fig = plt.figure(figsize=(15,15),dpi=100)
                    # 3DAxesを追加
                    ax = fig.add_subplot(111, projection='3d')

                    # ジョイント出力                    
                    ax.set_xlim3d(-1, 1)
                    ax.set_ylim3d(-1, 1)
                    ax.set_zlim3d(-1, 1)
                    ax.set(xlabel='x', ylabel='y', zlabel='z')

                    for i, (k, v) in enumerate(frame_joints["joints"].items()):
                        ax.plot3D(float(v["x"]), float(v["z"]), float(v["y"]), marker="o", ms=2, c="#0000FF")

                    plt.show()
                    plt.savefig(os.path.join(ordered_person_dir_path, f'joints_{fno:012}.png'))
                    plt.close()

            motion_path = os.path.join(motion_dir_path, "output_no{0:03}_{1}.vmd".format(oidx, datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
            writer = VmdWriter(model, motion, motion_path)
            writer.write()
                    
        logger.info('モーション生成処理全件終了', decoration=MLogger.DECORATION_BOX)

        return True
    except Exception as e:
        logger.critical("モーション生成で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        return False
