# -*- coding: utf-8 -*-
import numpy as np
import argparse
import csv
from miu.mmd.VmdWriter import VmdWriter
import os.path as osp
import json
import glob
import datetime

from miu.module.MMath import MMatrix4x4, MQuaternion, MVector3D
from miu.mmd.VmdData import VmdBoneFrame, VmdMotion, VmdShowIkFrame, VmdInfoIk
from miu.mmd.PmxData import PmxModel, Bone
from miu.utils.MServiceUtils import get_file_encoding, calc_global_pos, calc_relative_rotation
from miu.utils.MLogger import MLogger

logger = MLogger(__name__, level=MLogger.DEBUG)
SCALE_MIKU = 0.0625

def execute(cmd_args):
    folder_path = cmd_args.folder_path
    bone_csv_path = cmd_args.bone_csv_path

    model = read_bone_csv(bone_csv_path)
    logger.info(model)

    # デフォルトの関節位置
    default_joints = {}
    with open(cmd_args.bone_json_path, 'r') as f:
        default_joints = json.load(f)
    
    motion = VmdMotion()    
    
    # 動画上の関節位置
    for fno, joints_path in enumerate(glob.glob(osp.join(folder_path, '**/*_joints.json'))):

        frame_joints = {}
        with open(joints_path, 'r') as f:
            frame_joints = json.load(f)

        bf = VmdBoneFrame(fno)
        bf.set_name("センター")
        bf.position = calc_center(frame_joints)
        motion.regist_bf(bf, bf.name, bf.fno)

        for jname, (bone_name, calc_bone, name_list, parent_list) in VMD_CONNECTIONS.items():
            bf = VmdBoneFrame(fno)
            bf.set_name(bone_name)
            
            if calc_bone is None:
                rotation = calc_direction_qq(bf.fno, motion, frame_joints, *name_list)
                initial = calc_direction_qq(bf.fno, motion, default_joints, *name_list)

                qq = MQuaternion()
                for parent_name in reversed(parent_list):
                    qq *= motion.calc_bf(parent_name, bf.fno).rotation.inverted()
                qq = qq * rotation * initial.inverted()
                bf.rotation = qq

                motion.regist_bf(bf, bf.name, bf.fno)
            else:
                # 独自計算が必要な場合、設定
                calc_bone(bf, motion, model, jname, default_joints, frame_joints)
    
    # 動画内の半分は地面に足が着いていると見なす
    center_values = np.zeros((1, 3))
    for bf in motion.bones["センター"].values():
        center_values = np.insert(center_values, 0, np.array([bf.position.x(), bf.position.y(), bf.position.z()]), axis=0)
    
    center_median = np.median(center_values, axis=0)

    for bf in motion.bones["センター"].values():
        bf.position.setY(bf.position.y() - center_median[1])
    
    # IK変換
    convert_leg_fk2ik("左", motion, model)
    convert_leg_fk2ik("右", motion, model)

    writer = VmdWriter(motion, model, osp.join(folder_path, "output_{0}.vmd".format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))))
    writer.write()

# 足ＩＫ変換処理実行
def convert_leg_fk2ik(direction: str, motion: VmdMotion, model: PmxModel):
    logger.info("足ＩＫ変換　【%s足ＩＫ】", direction, decoration=MLogger.DECORATION_LINE)

    leg_ik_bone_name = "{0}足ＩＫ".format(direction)
    toe_ik_bone_name = "{0}つま先ＩＫ".format(direction)
    leg_bone_name = "{0}足".format(direction)
    knee_bone_name = "{0}ひざ".format(direction)
    ankle_bone_name = "{0}足首".format(direction)

    # 足FK末端までのリンク
    fk_links = model.create_link_2_top_one(ankle_bone_name, is_defined=False)
    # 足IK末端までのリンク
    ik_links = model.create_link_2_top_one(leg_ik_bone_name, is_defined=False)
    # つま先IK末端までのリンク
    toe_ik_links = model.create_link_2_top_one(toe_ik_bone_name, is_defined=False)
    # つま先（足首の子ボーン）の名前
    ankle_child_bone_name = "{0}つま先".format(direction)
    # つま先末端までのリンク
    toe_fk_links = model.create_link_2_top_one(ankle_child_bone_name, is_defined=False)

    fnos = motion.get_bone_fnos(leg_bone_name, knee_bone_name, ankle_bone_name)

    # まずキー登録
    prev_sep_fno = 0
    for fno in fnos:
        bf = motion.calc_bf(leg_ik_bone_name, fno)
        motion.regist_bf(bf, leg_ik_bone_name, fno)

        if fno // 2000 > prev_sep_fno and fnos[-1] > 0:
            logger.info("-- %sフレーム目:終了(%s％)【準備 - %s】", fno, round((fno / fnos[-1]) * 100, 3), leg_ik_bone_name)
            prev_sep_fno = fno // 2000
    
    if len(fnos) > 0 and fnos[-1] > 0:
        logger.info("-- %sフレーム目:終了(%s％)【準備 - %s】", fnos[-1], round((fnos[-1] / fnos[-1]) * 100, 3), leg_ik_bone_name)

    logger.info("準備完了　【%s足ＩＫ】", direction, decoration=MLogger.DECORATION_LINE)

    ik_parent_name = ik_links.get(leg_ik_bone_name, offset=-1).name

    # 足IKの移植
    prev_sep_fno = 0

    # 移植
    for fno in fnos:
        leg_fk_3ds_dic = calc_global_pos(model, fk_links, motion, fno)
        _, leg_ik_matrixs = calc_global_pos(model, ik_links, motion, fno, return_matrix=True)

        # IKの親から見た相対位置
        leg_ik_parent_matrix = leg_ik_matrixs[ik_parent_name]

        bf = motion.calc_bf(leg_ik_bone_name, fno)
        # 足ＩＫの位置は、足ＩＫの親から見た足首のローカル位置（足首位置マイナス）
        bf.position = leg_ik_parent_matrix.inverted() * (leg_fk_3ds_dic[ankle_bone_name] - (model.bones[ankle_bone_name].position - model.bones[ik_parent_name].position))

        # 足首の角度がある状態での、つま先までのグローバル位置
        leg_toe_fk_3ds_dic = calc_global_pos(model, toe_fk_links, motion, fno)

        # 一旦足ＩＫの位置が決まった時点で登録
        motion.regist_bf(bf, leg_ik_bone_name, fno)
        # 足ＩＫ回転なし状態でのつま先までのグローバル位置
        leg_ik_3ds_dic, leg_ik_matrisxs = calc_global_pos(model, toe_ik_links, motion, fno, return_matrix=True)

        # つま先のローカル位置
        ankle_child_initial_local_pos = leg_ik_matrisxs[leg_ik_bone_name].inverted() * leg_ik_3ds_dic[toe_ik_bone_name]
        ankle_child_local_pos = leg_ik_matrisxs[leg_ik_bone_name].inverted() * leg_toe_fk_3ds_dic[ankle_child_bone_name]

        # 足ＩＫの回転は、足首から見たつま先の方向
        bf.rotation = MQuaternion.rotationTo(ankle_child_initial_local_pos, ankle_child_local_pos)

        motion.regist_bf(bf, leg_ik_bone_name, fno)

        if fno // 2000 > prev_sep_fno and fnos[-1] > 0:
            logger.info("-- %sフレーム目:終了(%s％)【足ＩＫ変換 - %s】", fno, round((fno / fnos[-1]) * 100, 3), leg_ik_bone_name)
            prev_sep_fno = fno // 2000

    if len(fnos) > 0 and fnos[-1] > 0:
        logger.info("-- %sフレーム目:終了(%s％)【足ＩＫ変換 - %s】", fnos[-1], round((fnos[-1] / fnos[-1]) * 100, 3), leg_ik_bone_name)

    logger.info("変換完了　【%s足ＩＫ】", direction, decoration=MLogger.DECORATION_LINE)


def calc_direction_qq(bf: VmdBoneFrame, motion: VmdMotion, joints: dict, direction_from_name: str, direction_to_name: str, up_from_name: str, up_to_name: str):
    direction_from_vec = get_vec3(joints["joints"][direction_from_name])
    direction_to_vec = get_vec3(joints["joints"][direction_to_name])
    up_from_vec = get_vec3(joints["joints"][up_from_name])
    up_to_vec = get_vec3(joints["joints"][up_to_name])

    direction = (direction_to_vec - direction_from_vec).normalized()
    up = (up_to_vec - up_from_vec).normalized()
    cross = MVector3D.crossProduct(direction, up)
    qq = MQuaternion.fromDirection(direction, cross)

    return qq

def get_vec3(joint):
    return MVector3D(joint["x"], joint["z"], joint["y"]) * SCALE_MIKU

def calc_center(frame_joints: dict):
    # プロジェクション座標系の位置
    px = (frame_joints["proj_joints"]["pelvis"]["x"] - (frame_joints["image"]["width"] / 2)) * SCALE_MIKU
    py = (frame_joints["proj_joints"]["pelvis"]["y"] - (frame_joints["image"]["height"] / 2)) * SCALE_MIKU
    cz = frame_joints["depth"]["depth"] * SCALE_MIKU
    return MVector3D(px, py, cz)

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
    'spine1': ("上半身", None, ['spine1', 'spine3', 'left_shoulder', 'right_shoulder'], []),
    'spine2': ("上半身2", None, ['spine3', 'neck', 'left_shoulder', 'right_shoulder'], ["上半身"]),
    'neck': ("首", None, ['spine3', 'neck', 'left_shoulder', 'right_shoulder'], ["上半身", "上半身2"]),
    'head': ("頭", None, ['neck', 'head', 'left_eye1', 'right_eye1'], ["上半身", "上半身2", "首"]),
    'right_shoulder': ("右肩", None, ['spine3', 'right_shoulder', 'right_shoulder', 'left_shoulder'], ["上半身", "上半身2"]),
    'left_shoulder': ("左肩", None, ['spine3', 'left_shoulder', 'left_shoulder', 'right_shoulder'], ["上半身", "上半身2"]),
    'right_arm': ("右腕", None, ['right_shoulder', 'right_elbow', 'right_elbow', 'right_wrist'], ["上半身", "上半身2", "右肩"]),
    'left_arm': ("左腕", None, ['left_shoulder', 'left_elbow', 'left_elbow', 'left_wrist'], ["上半身", "上半身2", "左肩"]),
    'right_elbow': ("右ひじ", None, ['right_elbow', 'right_wrist', 'right_shoulder', 'right_elbow'], ["上半身", "上半身2", "右肩", "右腕"]),
    'left_elbow': ("左ひじ", None, ['left_elbow', 'left_wrist', 'left_shoulder', 'left_elbow'], ["上半身", "上半身2", "左肩", "左腕"]),
    'right_wrist': ("右手首", None, ['right_wrist', 'right_middle1', 'right_index1', 'right_pinky1'], ["上半身", "上半身2", "右肩", "右腕", "右ひじ"]),
    # 'left_wrist': ("左手首", None, ['left_wrist', 'left_middle1', 'left_elbow', 'left_wrist'], ["上半身", "上半身2", "左肩", "左腕", "左ひじ"]),
    'pelvis': ("下半身", None, ['spine1', 'pelvis', 'left_hip', 'right_hip'], []),
    'right_hip': ("右足", None, ['right_hip', 'right_knee', 'right_knee', 'right_ankle'], ["下半身"]),
    'left_hip': ("左足", None, ['left_hip', 'left_knee', 'left_knee', 'left_ankle'], ["下半身"]),
    'right_knee': ("右ひざ", None, ['right_knee', 'right_ankle', 'right_ankle', 'right_big_toe'], ["下半身", "右足"]),
    'left_knee': ("左ひざ", None, ['left_knee', 'left_ankle', 'left_ankle', 'left_big_toe'], ["下半身", "左足"]),
    'right_ankle': ("右足首", None, ['right_ankle', 'right_big_toe', 'right_big_toe', 'right_small_toe'], ["下半身", "右足", "右ひざ"]),
    'left_ankle': ("左足首", None, ['left_ankle', 'left_big_toe', 'left_big_toe', 'left_small_toe'], ["下半身", "左足", "左ひざ"]),
    # 'right_index1': ("右人指１", calc_finger),
    # 'left_index1': ("左人指１", calc_finger),
    # 'right_index2': ("右人指２", calc_finger),
    # 'left_index2': ("左人指２", calc_finger),
    # 'right_index3': ("右人指３", calc_finger),
    # 'left_index3': ("左人指３", calc_finger),
    # 'right_middle1': ("右中指１", calc_finger),
    # 'left_middle1': ("左中指１", calc_finger),
    # 'right_middle2': ("右中指２", calc_finger),
    # 'left_middle2': ("左中指２", calc_finger),
    # 'right_middle3': ("右中指３", calc_finger),
    # 'left_middle3': ("左中指３", calc_finger),
    # 'right_ring1': ("右薬指１", calc_finger),
    # 'left_ring1': ("左薬指１", calc_finger),
    # 'right_ring2': ("右薬指２", calc_finger),
    # 'left_ring2': ("左薬指２", calc_finger),
    # 'right_ring3': ("右薬指３", calc_finger),
    # 'left_ring3': ("左薬指３", calc_finger),
    # 'right_pinky1': ("右小指１", calc_finger),
    # 'left_pinky1': ("左小指１", calc_finger),
    # 'right_pinky2': ("右小指２", calc_finger),
    # 'left_pinky2': ("左小指２", calc_finger),
    # 'right_pinky3': ("右小指３", calc_finger),
    # 'left_pinky3': ("左小指３", calc_finger),
    # 'right_thumb1': ("右親指０", calc_finger),
    # 'left_thumb1': ("左親指０", calc_finger),
    # 'right_thumb2': ("右親指１", calc_finger),
    # 'left_thumb2': ("左親指１", calc_finger),
    # 'right_thumb3': ("右親指２", calc_finger),
    # 'left_thumb3': ("左親指２", calc_finger),

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
