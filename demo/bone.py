# -*- coding: utf-8 -*-
import numpy as np
import argparse
import csv
from miu.mmd.VmdWriter import VmdWriter
import os.path as osp
import json
import glob

from expose.data.targets.keypoints import ALL_CONNECTIONS, KEYPOINT_NAMES, VMD_CONNECTIONS

from miu.module.MMath import MMatrix4x4, MQuaternion, MVector3D
from miu.mmd.VmdData import VmdBoneFrame, VmdMotion, VmdShowIkFrame, VmdInfoIk
from miu.mmd.PmxData import PmxModel, Bone
from miu.utils.MServiceUtils import get_file_encoding, separate_local_qq
from miu.utils.MLogger import MLogger

logger = MLogger(__name__, level=MLogger.DEBUG)
SCALE_MIKU = 0.0625

def execute(cmd_args):
    folder_path = cmd_args.folder_path
    bone_csv_path = cmd_args.bone_csv_path

    model = read_bone_csv(bone_csv_path)
    logger.info(model)

    joint_dict = {}
    joint_dict["image"] = {"width": 1920, "height": 1080}
    joint_dict["depth"] = {"depth": 9.99464225769043}
    joint_dict["center"] = {"x": 960, "y": 540}
    joint_dict["joints"] = {}

    v = (model.bones["左足"].position + model.bones["左足"].position) / 2
    joint_dict["joints"]["pelvis"] = {"x": v.x(), "y": v.y(), "z": v.z()}
    
    v = model.bones["上半身"]
    joint_dict["joints"]["spine1"] = {"x": v.position.x(), "y": v.position.y(), "z": v.position.z()}

    v = model.bones["上半身2"]
    joint_dict["joints"]["spine3"] = {"x": v.position.x(), "y": v.position.y(), "z": v.position.z()}

    v = model.bones["首"]
    joint_dict["joints"]["neck"] = {"x": v.position.x(), "y": v.position.y(), "z": v.position.z()}

    v = model.bones["頭"]
    joint_dict["joints"]["head"] = {"x": v.position.x(), "y": v.position.y(), "z": v.position.z()}

    v = model.bones["左目"]
    joint_dict["joints"]["left_eye1"] = {"x": v.position.x(), "y": v.position.y(), "z": v.position.z()}

    v = model.bones["右目"]
    joint_dict["joints"]["right_eye1"] = {"x": v.position.x(), "y": v.position.y(), "z": v.position.z()}

    v = model.bones["右肩"]
    joint_dict["joints"]["right_shoulder"] = {"x": v.position.x(), "y": v.position.y(), "z": v.position.z()}

    v = model.bones["左肩"]
    joint_dict["joints"]["left_shoulder"] = {"x": v.position.x(), "y": v.position.y(), "z": v.position.z()}

    v = model.bones["右腕"]
    joint_dict["joints"]["right_arm"] = {"x": v.position.x(), "y": v.position.y(), "z": v.position.z()}

    v = model.bones["左腕"]
    joint_dict["joints"]["left_arm"] = {"x": v.position.x(), "y": v.position.y(), "z": v.position.z()}

    v = model.bones["右ひじ"]
    joint_dict["joints"]["right_elbow"] = {"x": v.position.x(), "y": v.position.y(), "z": v.position.z()}

    v = model.bones["左ひじ"]
    joint_dict["joints"]["left_elbow"] = {"x": v.position.x(), "y": v.position.y(), "z": v.position.z()}

    v = model.bones["右手首"]
    joint_dict["joints"]["right_wrist"] = {"x": v.position.x(), "y": v.position.y(), "z": v.position.z()}

    v = model.bones["左手首"]
    joint_dict["joints"]["left_wrist"] = {"x": v.position.x(), "y": v.position.y(), "z": v.position.z()}

    v = model.bones["右足"]
    joint_dict["joints"]["right_hip"] = {"x": v.position.x(), "y": v.position.y(), "z": v.position.z()}

    v = model.bones["左足"]
    joint_dict["joints"]["left_hip"] = {"x": v.position.x(), "y": v.position.y(), "z": v.position.z()}

    v = model.bones["右ひざ"]
    joint_dict["joints"]["right_knee"] = {"x": v.position.x(), "y": v.position.y(), "z": v.position.z()}

    v = model.bones["左ひざ"]
    joint_dict["joints"]["left_knee"] = {"x": v.position.x(), "y": v.position.y(), "z": v.position.z()}

    v = model.bones["右足首"]
    joint_dict["joints"]["right_ankle"] = {"x": v.position.x(), "y": v.position.y(), "z": v.position.z()}

    v = model.bones["左足首"]
    joint_dict["joints"]["left_ankle"] = {"x": v.position.x(), "y": v.position.y(), "z": v.position.z()}

    v = model.bones["右人指１"]
    joint_dict["joints"]["right_index1"] = {"x": v.position.x(), "y": v.position.y(), "z": v.position.z()}

    v = model.bones["左人指１"]
    joint_dict["joints"]["left_index1"] = {"x": v.position.x(), "y": v.position.y(), "z": v.position.z()}

    v = model.bones["右中指１"]
    joint_dict["joints"]["right_middle1"] = {"x": v.position.x(), "y": v.position.y(), "z": v.position.z()}

    v = model.bones["左中指１"]
    joint_dict["joints"]["left_middle1"] = {"x": v.position.x(), "y": v.position.y(), "z": v.position.z()}

    v = model.bones["右小指１"]
    joint_dict["joints"]["right_pinky1"] = {"x": v.position.x(), "y": v.position.y(), "z": v.position.z()}

    v = model.bones["左小指１"]
    joint_dict["joints"]["left_pinky1"] = {"x": v.position.x(), "y": v.position.y(), "z": v.position.z()}

    v = model.bones["右つま先"]
    joint_dict["joints"]["right_big_toe"] = {"x": v.position.x() + 1, "y": v.position.y(), "z": v.position.z()}

    v = model.bones["左つま先"]
    joint_dict["joints"]["left_big_toe"] = {"x": v.position.x() - 1, "y": v.position.y(), "z": v.position.z()}

    v = model.bones["右つま先"]
    joint_dict["joints"]["right_small_toe"] = {"x": v.position.x() - 1, "y": v.position.y(), "z": v.position.z()}

    v = model.bones["左つま先"]
    joint_dict["joints"]["left_small_toe"] = {"x": v.position.x() + 1, "y": v.position.y(), "z": v.position.z()}

    with open("C:\\MMD\\expose_mmd\\bone\\あにまさ式ミク準標準ボーン.json", 'w') as f:
        json.dump(joint_dict, f, indent=4)

def read_bone_csv(bone_csv_path: str):
    model = PmxModel()
    model.name = osp.splitext(osp.basename(bone_csv_path))[0]

    with open(bone_csv_path, "r", encoding=get_file_encoding(bone_csv_path)) as bf:
        reader = csv.reader(bf)

        # 列名行の代わりにルートボーン登録
        # サイジング用ルートボーン
        sizing_root_bone = Bone("SIZING_ROOT_BONE", "SIZING_ROOT_BONE", MVector3D(), -1, 0, 0)
        sizing_root_bone.index = -999

        model.bones[sizing_root_bone.name] = sizing_root_bone
        # インデックス逆引きも登録
        model.bone_indexes[sizing_root_bone.index] = sizing_root_bone.name

        for ridx, row in enumerate(reader):
            if row[0] == "Bone":
                bone = Bone(row[1], row[2], MVector3D(float(row[5]), float(row[6]), float(row[7])), row[13], int(row[3]), \
                            int(row[8]) | int(row[9]) | int(row[10]) | int(row[11]) | int(row[12]))
                bone.index = ridx - 1
                model.bones[row[1]] = bone
                model.bone_indexes[bone.index] = row[1]
    
    for bidx, bone in model.bones.items():
        # 親ボーンINDEXの設定
        if bone.parent_index and bone.parent_index in model.bones:
            bone.parent_index = model.bones[bone.parent_index].index
        else:
            bone.parent_index = -1

    return model


VMD_CONNECTIONS = {

    # 'left_wrist': ("左手首", calc_wrist),
    # ('left_eye', 'nose'): "",
    # ('right_eye', 'nose'): "",
    # ('right_eye', 'right_ear'): "",
    # ('left_eye', 'left_ear'): "",
    # ('right_shoulder', 'right_elbow'): "右腕",
    # ('right_elbow', 'right_wrist'): "右ひじ",
    # # Right Thumb
    # ('right_wrist', 'right_thumb1'): "",
    # ('right_thumb1', 'right_thumb2'): "",
    # ('right_thumb2', 'right_thumb3'): "",
    # ('right_thumb3', 'right_thumb'): "",
    # # Right Index
    # ('right_wrist', 'right_index1'): "",
    # ('right_index1', 'right_index2'): "",
    # ('right_index2', 'right_index3'): "",
    # ('right_index3', 'right_index'): "",
    # # Right Middle
    # ('right_wrist', 'right_middle1'): "",
    # ('right_middle1', 'right_middle2'): "",
    # ('right_middle2', 'right_middle3'): "",
    # ('right_middle3', 'right_middle'): "",
    # # Right Ring
    # ('right_wrist', 'right_ring1'): "",
    # ('right_ring1', 'right_ring2'): "",
    # ('right_ring2', 'right_ring3'): "",
    # ('right_ring3', 'right_ring'): "",
    # # Right Pinky
    # ('right_wrist', 'right_pinky1'): "",
    # ('right_pinky1', 'right_pinky2'): "",
    # ('right_pinky2', 'right_pinky3'): "",
    # ('right_pinky3', 'right_pinky'): "",
    # # Left Hand
    # ('left_shoulder', 'left_elbow'): "左腕",
    # ('left_elbow', 'left_wrist'): "左ひじ",
    # # Left Thumb
    # ('left_wrist', 'left_thumb1'): "",
    # ('left_thumb1', 'left_thumb2'): "",
    # ('left_thumb2', 'left_thumb3'): "",
    # ('left_thumb3', 'left_thumb'): "",
    # # Left Index
    # ('left_wrist', 'left_index1'): "",
    # ('left_index1', 'left_index2'): "",
    # ('left_index2', 'left_index3'): "",
    # ('left_index3', 'left_index'): "",
    # # Left Middle
    # ('left_wrist', 'left_middle1'): "",
    # ('left_middle1', 'left_middle2'): "",
    # ('left_middle2', 'left_middle3'): "",
    # ('left_middle3', 'left_middle'): "",
    # # Left Ring
    # ('left_wrist', 'left_ring1'): "",
    # ('left_ring1', 'left_ring2'): "",
    # ('left_ring2', 'left_ring3'): "",
    # ('left_ring3', 'left_ring'): "",
    # # Left Pinky
    # ('left_wrist', 'left_pinky1'): "",
    # ('left_pinky1', 'left_pinky2'): "",
    # ('left_pinky2', 'left_pinky3'): "",
    # ('left_pinky3', 'left_pinky'): "",

    # # Right Foot
    # ('right_hip', 'right_knee'): "右足",
    # ('right_knee', 'right_ankle'): "右ひざ",
    # ('right_ankle', 'right_heel'): "",
    # ('right_ankle', 'right_big_toe'): "",
    # ('right_ankle', 'right_small_toe'): "",

    # ('left_hip', 'left_knee'): "左足",
    # ('left_knee', 'left_ankle'): "左ひざ",
    # ('left_ankle', 'left_heel'): "",
    # ('left_ankle', 'left_big_toe'): "",
    # ('left_ankle', 'left_small_toe'): "",

    # ('neck', 'right_shoulder'): "右肩",
    # ('neck', 'left_shoulder'): "左肩",
    # ('neck', 'nose'): "頭",
    # ('spine1', 'spine3'): "上半身2",   # 上半身2対応のため、spine2はなし
    # ('spine3', 'neck'): "首",
    # ('pelvis', 'left_hip'): "",
    # ('pelvis', 'right_hip'): "",

    # # Left Eye brow
    # ('left_eye_brow1', 'left_eye_brow2'): "",
    # ('left_eye_brow2', 'left_eye_brow3'): "",
    # ('left_eye_brow3', 'left_eye_brow4'): "",
    # ('left_eye_brow4', 'left_eye_brow5'): "",

    # # Right Eye brow
    # ('right_eye_brow1', 'right_eye_brow2'): "",
    # ('right_eye_brow2', 'right_eye_brow3'): "",
    # ('right_eye_brow3', 'right_eye_brow4'): "",
    # ('right_eye_brow4', 'right_eye_brow5'): "",

    # # Left Eye
    # ('left_eye1', 'left_eye2'): "",
    # ('left_eye2', 'left_eye3'): "",
    # ('left_eye3', 'left_eye4'): "",
    # ('left_eye4', 'left_eye5'): "",
    # ('left_eye5', 'left_eye6'): "",
    # ('left_eye6', 'left_eye1'): "",

    # # Right Eye
    # ('right_eye1', 'right_eye2'): "",
    # ('right_eye2', 'right_eye3'): "",
    # ('right_eye3', 'right_eye4'): "",
    # ('right_eye4', 'right_eye5'): "",
    # ('right_eye5', 'right_eye6'): "",
    # ('right_eye6', 'right_eye1'): "",

    # # Nose Vertical
    # ('nose1', 'nose2'): "",
    # ('nose2', 'nose3'): "",
    # ('nose3', 'nose4'): "",

    # # Nose Horizontal
    # ('nose4', 'nose_middle'): "",
    # ('left_nose_1', 'left_nose_2'): "",
    # ('left_nose_2', 'nose_middle'): "",
    # ('nose_middle', 'right_nose_1'): "",
    # ('right_nose_1', 'right_nose_2'): "",

    # # Mouth
    # ('left_mouth_1', 'left_mouth_2'): "",
    # ('left_mouth_2', 'left_mouth_3'): "",
    # ('left_mouth_3', 'mouth_top'): "",
    # ('mouth_top', 'right_mouth_3'): "",
    # ('right_mouth_3', 'right_mouth_2'): "",
    # ('right_mouth_2', 'right_mouth_1'): "",
    # ('right_mouth_1', 'right_mouth_5'): "",
    # ('right_mouth_5', 'right_mouth_4'): "",
    # ('right_mouth_4', 'mouth_bottom'): "",
    # ('mouth_bottom', 'left_mouth_4'): "",
    # ('left_mouth_4', 'left_mouth_5'): "",
    # ('left_mouth_5', 'left_mouth_1'): "",

    # # Lips
    # ('left_lip_1', 'left_lip_2'): "",
    # ('left_lip_2', 'lip_top'): "",
    # ('lip_top', 'right_lip_2'): "",
    # ('right_lip_2', 'right_lip_1'): "",
    # ('right_lip_1', 'right_lip_3'): "",
    # ('right_lip_3', 'lip_bottom'): "",
    # ('lip_bottom', 'left_lip_3'): "",
    # ('left_lip_3', 'left_lip_1'): "",

    # # Contour
    # ('left_contour_1', 'left_contour_2'): "",
    # ('left_contour_2', 'left_contour_3'): "",
    # ('left_contour_3', 'left_contour_4'): "",
    # ('left_contour_4', 'left_contour_5'): "",
    # ('left_contour_5', 'left_contour_6'): "",
    # ('left_contour_6', 'left_contour_7'): "",
    # ('left_contour_7', 'left_contour_8'): "",
    # ('left_contour_8', 'contour_middle'): "",

    # ('contour_middle', 'right_contour_8'): "",
    # ('right_contour_8', 'right_contour_7'): "",
    # ('right_contour_7', 'right_contour_6'): "",
    # ('right_contour_6', 'right_contour_5'): "",
    # ('right_contour_5', 'right_contour_4'): "",
    # ('right_contour_4', 'right_contour_3'): "",
    # ('right_contour_3', 'right_contour_2'): "",
    # ('right_contour_2', 'right_contour_1'): "",
}

if __name__ == '__main__':
    arg_formatter = argparse.ArgumentDefaultsHelpFormatter
    description = 'VMD Output'

    parser = argparse.ArgumentParser(formatter_class=arg_formatter,
                                     description=description)

    parser.add_argument('--folder-path', type=str, dest='folder_path', help='The folder with joint json')
    parser.add_argument('--bone-csv-path', type=str, dest='bone_csv_path', help='The csv file pmx born')
    parser.add_argument('--bone-json-path', type=str, dest='bone_json_path', help='The json file pmx born')
    parser.add_argument('--verbose', type=int, dest='verbose', default=20, help='The csv file pmx born')

    cmd_args = parser.parse_args()

    MLogger.initialize(level=cmd_args.verbose, is_file=True)

    execute(cmd_args)
