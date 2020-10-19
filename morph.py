# -*- coding: utf-8 -*-
from os import name
import numpy as np
import argparse
import csv

from numpy.core.defchararray import center
from miu.mmd.VmdWriter import VmdWriter
import os
import os.path as osp
import json
import glob
import datetime
import math

from miu.module.MMath import MMatrix4x4, MQuaternion, MVector2D
from miu.mmd.VmdData import VmdBoneFrame, VmdMotion, VmdShowIkFrame, VmdInfoIk, VmdMorphFrame
from miu.mmd.PmxData import PmxModel, Bone, Vertex
from miu.utils.MServiceUtils import get_file_encoding, calc_global_pos, separate_local_qq
from miu.utils.MLogger import MLogger

logger = MLogger(__name__, level=MLogger.DEBUG)

def execute(cmd_args):
    folder_path = cmd_args.folder_path

    motion = VmdMotion()
    
    # 動画上の関節位置
    for fno, joints_path in enumerate(glob.glob(osp.join(folder_path, 'faces_*.json'))):
        logger.info(f"■ fno: {fno} -----")

        frame_joints = {}
        with open(joints_path, 'r') as f:
            frame_joints = json.load(f)

        # まばたき
        calc_left_blink(fno, motion, frame_joints)
        calc_right_blink(fno, motion, frame_joints)
        blend_eye(fno, motion)

        # 口
        calc_lip(fno, motion, frame_joints)

        # 眉
        calc_eyebrow(fno, motion, frame_joints)

    model = PmxModel()
    model.name = "Morph Model"
    writer = VmdWriter(motion, model, osp.join(folder_path, "morph_{0}.vmd".format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))))
    writer.write()

# 眉モーフ
def calc_eyebrow(fno: int, motion: VmdMotion, frame_joints: dict):
    left_eye_brow1 = get_vec2(frame_joints["0"]["faces"], "left_eye_brow1")
    left_eye_brow2 = get_vec2(frame_joints["0"]["faces"], "left_eye_brow2")
    left_eye_brow3 = get_vec2(frame_joints["0"]["faces"], "left_eye_brow3")
    left_eye_brow4 = get_vec2(frame_joints["0"]["faces"], "left_eye_brow4")
    left_eye_brow5 = get_vec2(frame_joints["0"]["faces"], "left_eye_brow5")

    left_eye1 = get_vec2(frame_joints["0"]["faces"], "left_eye1")
    left_eye2 = get_vec2(frame_joints["0"]["faces"], "left_eye2")
    left_eye3 = get_vec2(frame_joints["0"]["faces"], "left_eye3")
    left_eye4 = get_vec2(frame_joints["0"]["faces"], "left_eye4")
    left_eye5 = get_vec2(frame_joints["0"]["faces"], "left_eye5")
    left_eye6 = get_vec2(frame_joints["0"]["faces"], "left_eye6")

    right_eye_brow1 = get_vec2(frame_joints["0"]["faces"], "right_eye_brow1")
    right_eye_brow2 = get_vec2(frame_joints["0"]["faces"], "right_eye_brow2")
    right_eye_brow3 = get_vec2(frame_joints["0"]["faces"], "right_eye_brow3")
    right_eye_brow4 = get_vec2(frame_joints["0"]["faces"], "right_eye_brow4")
    right_eye_brow5 = get_vec2(frame_joints["0"]["faces"], "right_eye_brow5")

    right_eye1 = get_vec2(frame_joints["0"]["faces"], "right_eye1")
    right_eye2 = get_vec2(frame_joints["0"]["faces"], "right_eye2")
    right_eye3 = get_vec2(frame_joints["0"]["faces"], "right_eye3")
    right_eye4 = get_vec2(frame_joints["0"]["faces"], "right_eye4")
    right_eye5 = get_vec2(frame_joints["0"]["faces"], "right_eye5")
    right_eye6 = get_vec2(frame_joints["0"]["faces"], "right_eye6")

    left_nose_1 = get_vec2(frame_joints["0"]["faces"], 'left_nose_1')
    right_nose_2 = get_vec2(frame_joints["0"]["faces"], 'right_nose_2')

    # 鼻の幅
    nose_width = abs(left_nose_1.x() - right_nose_2.x())
    
    # 眉のしかめ具合
    frown_ratio = abs(left_eye_brow1.x() - right_eye_brow1.x()) / nose_width

    # 眉の幅
    eye_brow_length = (euclidean_distance(left_eye_brow1, left_eye_brow5) + euclidean_distance(right_eye_brow1, right_eye_brow5)) / 2
    # 目の幅
    eye_length = (euclidean_distance(left_eye1, left_eye4) + euclidean_distance(right_eye1, right_eye4)) / 2

    # 目と眉の縦幅
    left_vertical_length = (euclidean_distance(left_eye1, left_eye_brow1) + euclidean_distance(right_eye1, right_eye_brow1)) / 2
    center_vertical_length = (euclidean_distance(left_eye2, left_eye_brow3) + euclidean_distance(right_eye2, right_eye_brow3)) / 2
    right_vertical_length = (euclidean_distance(left_eye4, left_eye_brow5) + euclidean_distance(right_eye4, right_eye_brow5)) / 2

    left_ratio = left_vertical_length / eye_brow_length
    center_ratio = center_vertical_length / eye_brow_length
    right_ratio = right_vertical_length / eye_brow_length

    updown_ratio = center_ratio - 0.5

    if updown_ratio >= 0.2:
        # 上
        mf = VmdMorphFrame(fno)
        mf.set_name("上")
        mf.ratio = max(0, min(1, abs(updown_ratio) + 0.3))
        motion.regist_mf(mf, mf.name, mf.fno)

        mf = VmdMorphFrame(fno)
        mf.set_name("下")
        mf.ratio = 0
        motion.regist_mf(mf, mf.name, mf.fno)
    else:
        # 下
        mf = VmdMorphFrame(fno)
        mf.set_name("下")
        mf.ratio = max(0, min(1, abs(updown_ratio) + 0.3))
        motion.regist_mf(mf, mf.name, mf.fno)

        mf = VmdMorphFrame(fno)
        mf.set_name("上")
        mf.ratio = 0
        motion.regist_mf(mf, mf.name, mf.fno)
    
    mf = VmdMorphFrame(fno)
    mf.set_name("困る")
    mf.ratio = max(0, min(1, (0.8 - frown_ratio)))
    motion.regist_mf(mf, mf.name, mf.fno)

    if left_ratio >= right_ratio:
        # 怒る系
        mf = VmdMorphFrame(fno)
        mf.set_name("怒り")
        mf.ratio = max(0, min(1, abs(left_ratio - right_ratio)))
        motion.regist_mf(mf, mf.name, mf.fno)

        mf = VmdMorphFrame(fno)
        mf.set_name("にこり")
        mf.ratio = 0
        motion.regist_mf(mf, mf.name, mf.fno)
    else:
        # 笑う系
        mf = VmdMorphFrame(fno)
        mf.set_name("にこり")
        mf.ratio = max(0, min(1, abs(right_ratio - left_ratio)))
        motion.regist_mf(mf, mf.name, mf.fno)

        mf = VmdMorphFrame(fno)
        mf.set_name("怒り")
        mf.ratio = 0
        motion.regist_mf(mf, mf.name, mf.fno)

# リップモーフ
def calc_lip(fno: int, motion: VmdMotion, frame_joints: dict):
    left_nose_1 = get_vec2(frame_joints["0"]["faces"], 'left_nose_1')
    right_nose_2 = get_vec2(frame_joints["0"]["faces"], 'right_nose_2')

    left_mouth_1 = get_vec2(frame_joints["0"]["faces"], 'left_mouth_1')
    left_mouth_2 = get_vec2(frame_joints["0"]["faces"], 'left_mouth_2')
    left_mouth_3 = get_vec2(frame_joints["0"]["faces"], 'left_mouth_3')
    mouth_top = get_vec2(frame_joints["0"]["faces"], 'mouth_top')
    right_mouth_3 = get_vec2(frame_joints["0"]["faces"], 'right_mouth_3')
    right_mouth_2 = get_vec2(frame_joints["0"]["faces"], 'right_mouth_2')
    right_mouth_1 = get_vec2(frame_joints["0"]["faces"], 'right_mouth_1')
    right_mouth_5 = get_vec2(frame_joints["0"]["faces"], 'right_mouth_5')
    right_mouth_4 = get_vec2(frame_joints["0"]["faces"], 'right_mouth_4')
    mouth_bottom = get_vec2(frame_joints["0"]["faces"], 'mouth_bottom')
    left_mouth_4 = get_vec2(frame_joints["0"]["faces"], 'left_mouth_4')
    left_mouth_5 = get_vec2(frame_joints["0"]["faces"], 'left_mouth_5')
    left_lip_1 = get_vec2(frame_joints["0"]["faces"], 'left_lip_1')
    left_lip_2 = get_vec2(frame_joints["0"]["faces"], 'left_lip_2')
    lip_top = get_vec2(frame_joints["0"]["faces"], 'lip_top')
    right_lip_2 = get_vec2(frame_joints["0"]["faces"], 'right_lip_2')
    right_lip_1 = get_vec2(frame_joints["0"]["faces"], 'right_lip_1')
    right_lip_3 = get_vec2(frame_joints["0"]["faces"], 'right_lip_3')
    lip_bottom = get_vec2(frame_joints["0"]["faces"], 'lip_bottom')
    left_lip_3 = get_vec2(frame_joints["0"]["faces"], 'left_lip_3')

    # 鼻の幅
    nose_width = abs(left_nose_1.x() - right_nose_2.x())

    # 口角の平均値
    corner_center = (left_mouth_1 + right_mouth_1) / 2
    # 口角の幅
    mouse_width = abs(left_mouth_1.x() - right_mouth_1.x())

    # 鼻基準の口の横幅比率
    mouse_width_ratio = mouse_width / nose_width

    # 上唇の平均値
    top_mouth_center = (right_mouth_3 + mouth_top + left_mouth_3) / 3
    top_lip_center = (right_lip_2 + lip_top + left_lip_2) / 3

    # 下唇の平均値
    bottom_mouth_center = (right_mouth_4 + mouth_bottom + left_mouth_4) / 3
    bottom_lip_center = (right_lip_3 + lip_bottom + left_lip_3) / 3

    # 唇の外側の開き具合に対する内側の開き具合
    open_ratio = (bottom_lip_center.y() - top_lip_center.y()) / (bottom_mouth_center.y() - top_mouth_center.y())

    # 笑いの比率
    smile_ratio = (bottom_mouth_center.y() - corner_center.y()) / (bottom_mouth_center.y() - top_mouth_center.y())

    if smile_ratio >= 0:
        mf = VmdMorphFrame(fno)
        mf.set_name("∧")
        mf.ratio = 0
        motion.regist_mf(mf, mf.name, mf.fno)

        mf = VmdMorphFrame(fno)
        mf.set_name("にやり")
        mf.ratio = max(0, min(1, abs(smile_ratio)))
        motion.regist_mf(mf, mf.name, mf.fno)
    else:
        mf = VmdMorphFrame(fno)
        mf.set_name("∧")
        mf.ratio = max(0, min(1, abs(smile_ratio)))
        motion.regist_mf(mf, mf.name, mf.fno)

        mf = VmdMorphFrame(fno)
        mf.set_name("にやり")
        mf.ratio = 0
        motion.regist_mf(mf, mf.name, mf.fno)

    if mouse_width_ratio > 1.3:
        # 横幅広がってる場合は「い」
        mf = VmdMorphFrame(fno)
        mf.set_name("い")
        mf.ratio = max(0, min(1, 1.5 / mouse_width_ratio) * open_ratio)
        motion.regist_mf(mf, mf.name, mf.fno)

        mf = VmdMorphFrame(fno)
        mf.set_name("う")
        mf.ratio = 0
        motion.regist_mf(mf, mf.name, mf.fno)

        mf = VmdMorphFrame(fno)
        mf.set_name("あ")
        mf.ratio = max(0, min(1 - min(0.7, smile_ratio), open_ratio))
        motion.regist_mf(mf, mf.name, mf.fno)

        mf = VmdMorphFrame(fno)
        mf.set_name("お")
        mf.ratio = 0
        motion.regist_mf(mf, mf.name, mf.fno)
    else:
        # 狭まっている場合は「う」
        mf = VmdMorphFrame(fno)
        mf.set_name("い")
        mf.ratio = 0
        motion.regist_mf(mf, mf.name, mf.fno)

        mf = VmdMorphFrame(fno)
        mf.set_name("う")
        mf.ratio = max(0, min(1, (1.2 / mouse_width_ratio) * open_ratio))
        motion.regist_mf(mf, mf.name, mf.fno)

        mf = VmdMorphFrame(fno)
        mf.set_name("あ")
        mf.ratio = 0
        motion.regist_mf(mf, mf.name, mf.fno)        

        mf = VmdMorphFrame(fno)
        mf.set_name("お")
        mf.ratio = max(0, min(1 - min(0.7, smile_ratio), open_ratio))
        motion.regist_mf(mf, mf.name, mf.fno)

def blend_eye(fno: int, motion: VmdMotion):
    min_blink = min(motion.morphs["ウィンク右"][fno].ratio, motion.morphs["ウィンク"][fno].ratio)
    min_smile = min(motion.morphs["ｳｨﾝｸ２右"][fno].ratio, motion.morphs["ウィンク２"][fno].ratio)

    # 両方の同じ値はさっぴく
    motion.morphs["ウィンク右"][fno].ratio -= min_smile
    motion.morphs["ウィンク"][fno].ratio -= min_smile

    motion.morphs["ｳｨﾝｸ２右"][fno].ratio -= min_blink
    motion.morphs["ウィンク２"][fno].ratio -= min_blink

    mf = VmdMorphFrame(fno)
    mf.set_name("笑い")
    mf.ratio = max(0, min(1, min_smile))
    motion.regist_mf(mf, mf.name, mf.fno)

    mf = VmdMorphFrame(fno)
    mf.set_name("まばたき")
    mf.ratio = max(0, min(1, min_blink))
    motion.regist_mf(mf, mf.name, mf.fno)


def calc_left_blink(fno: int, motion: VmdMotion, frame_joints: dict):
    left_eye1 = get_vec2(frame_joints["0"]["faces"], "left_eye1")
    left_eye2 = get_vec2(frame_joints["0"]["faces"], "left_eye2")
    left_eye3 = get_vec2(frame_joints["0"]["faces"], "left_eye3")
    left_eye4 = get_vec2(frame_joints["0"]["faces"], "left_eye4")
    left_eye5 = get_vec2(frame_joints["0"]["faces"], "left_eye5")
    left_eye6 = get_vec2(frame_joints["0"]["faces"], "left_eye6")

    # 左目のEAR(eyes aspect ratio)
    left_blink, left_smile = get_blink_ratio(left_eye1, left_eye2, left_eye3, left_eye4, left_eye5, left_eye6)

    mf = VmdMorphFrame(fno)
    mf.set_name("ウィンク右")
    mf.ratio = max(0, min(1, left_smile))

    motion.regist_mf(mf, mf.name, mf.fno)

    mf = VmdMorphFrame(fno)
    mf.set_name("ｳｨﾝｸ２右")
    mf.ratio = max(0, min(1, left_blink))

    motion.regist_mf(mf, mf.name, mf.fno)

def calc_right_blink(fno: int, motion: VmdMotion, frame_joints: dict):
    right_eye1 = get_vec2(frame_joints["0"]["faces"], "right_eye1")
    right_eye2 = get_vec2(frame_joints["0"]["faces"], "right_eye2")
    right_eye3 = get_vec2(frame_joints["0"]["faces"], "right_eye3")
    right_eye4 = get_vec2(frame_joints["0"]["faces"], "right_eye4")
    right_eye5 = get_vec2(frame_joints["0"]["faces"], "right_eye5")
    right_eye6 = get_vec2(frame_joints["0"]["faces"], "right_eye6")

    # 右目のEAR(eyes aspect ratio)
    right_blink, right_smile = get_blink_ratio(right_eye1, right_eye2, right_eye3, right_eye4, right_eye5, right_eye6)

    mf = VmdMorphFrame(fno)
    mf.set_name("ウィンク")
    mf.ratio = max(0, min(1, right_smile))
    motion.regist_mf(mf, mf.name, mf.fno)

    mf = VmdMorphFrame(fno)
    mf.set_name("ウィンク２")
    mf.ratio = max(0, min(1, right_blink))
    motion.regist_mf(mf, mf.name, mf.fno)

def euclidean_distance(point1: MVector2D, point2: MVector2D):
    return math.sqrt((point1.x() - point2.x())**2 + (point1.y() - point2.y())**2)

def get_blink_ratio(eye1: MVector2D, eye2: MVector2D, eye3: MVector2D, eye4: MVector2D, eye5: MVector2D, eye6: MVector2D):
    #loading all the required points
    corner_left  = eye1
    corner_right = eye4
    corner_center = (eye1 + eye4) / 2
    
    center_top = (eye2 + eye3) / 2
    center_bottom = (eye5 + eye6) / 2

    #calculating distance
    horizontal_length = euclidean_distance(corner_left, corner_right)
    vertical_length = euclidean_distance(center_top, center_bottom)

    ratio = horizontal_length / vertical_length
    new_ratio = min(1, calc_ratio(ratio, 0, 12, 0, 1))

    # 笑いの比率
    smile_ratio = (center_bottom.y() - corner_center.y()) / (center_bottom.y() - center_top.y())

    if smile_ratio > 1:
        # １より大きい場合、目頭よりも上瞼が下にあるという事なので、通常瞬きと見なす
        return 1, 0

    return new_ratio * (1 - smile_ratio), new_ratio * smile_ratio

def calc_ratio(ratio: float, oldmin: float, oldmax: float, newmin: float, newmax: float):
    # https://qastack.jp/programming/929103/convert-a-number-range-to-another-range-maintaining-ratio
    # NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
    return (((ratio - oldmin) * (newmax - newmin)) / (oldmax - oldmin)) + newmin

def get_vec2(joints: dict, jname: str):
    if jname in MORPH_CONNECTIONS:
        joint = joints[MORPH_CONNECTIONS[jname]]
        return MVector2D(joint["x"], joint["y"])
    
    return MVector2D()



MORPH_CONNECTIONS = {
    # Left Eye brow
    'left_eye_brow1': "23",
    'left_eye_brow2': "24",
    'left_eye_brow3': "25",
    'left_eye_brow4': "26",
    'left_eye_brow5': "27",

    # Right Eye brow
    'right_eye_brow1': "22",
    'right_eye_brow2': "21",
    'right_eye_brow3': "20",
    'right_eye_brow4': "19",
    'right_eye_brow5': "18",

    # Left Eye
    'left_eye1': "46",
    'left_eye2': "45",
    'left_eye3': "44",
    'left_eye4': "43",
    'left_eye5': "47",
    'left_eye6': "48",

    # Right Eye
    'right_eye1': "40",
    'right_eye2': "39",
    'right_eye3': "38",
    'right_eye4': "37",
    'right_eye5': "41",
    'right_eye6': "42",

    # Nose Vertical
    'nose1': "28",
    'nose2': "29",
    'nose3': "30",
    'nose4': "31",

    # Nose Horizontal
    'left_nose_1': "36",
    'left_nose_2': "35",
    'nose_middle': "34",
    'right_nose_1': "33",
    'right_nose_2': "32",

    # Mouth
    'left_mouth_1': "55",
    'left_mouth_2': "54",
    'left_mouth_3': "53",
    'mouth_top': "52",
    'right_mouth_3': "51",
    'right_mouth_2': "50",
    'right_mouth_1': "49",
    'right_mouth_5': "60",
    'right_mouth_4': "59",
    'mouth_bottom': "58",
    'left_mouth_4': "57",
    'left_mouth_5': "56",

    # Lips
    'left_lip_1': "65",
    'left_lip_2': "64",
    'lip_top': "63",
    'right_lip_2': "62",
    'right_lip_1': "61",
    'right_lip_3': "68",
    'lip_bottom': "67",
    'left_lip_3': "66",

}




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

    # # 終了音を鳴らす
    # if os.name == "nt":
    #     # Windows
    #     try:
    #         import winsound
    #         winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS)
    #     except Exception:
    #         pass


