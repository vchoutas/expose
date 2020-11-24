# -*- coding: utf-8 -*-
from os import name
import numpy as np
import argparse
import csv
from miu.mmd.VmdWriter import VmdWriter
import os
import os.path as osp
import json
import glob
import datetime

from miu.module.MMath import MMatrix4x4, MQuaternion, MVector3D
from miu.mmd.VmdData import VmdBoneFrame, VmdMotion, VmdShowIkFrame, VmdInfoIk, VmdMorphFrame
from miu.mmd.PmxData import PmxModel, Bone, Vertex
from miu.utils.MServiceUtils import get_file_encoding, calc_global_pos, separate_local_qq
from miu.utils.MLogger import MLogger

logger = MLogger(__name__, level=MLogger.DEBUG)
SCALE_MIKU = 0.0625

def execute(cmd_args):
    folder_path = cmd_args.folder_path
    bone_csv_path = cmd_args.bone_csv_path

    model = read_bone_csv(bone_csv_path)
    logger.info(model)

    motion = VmdMotion()
    
    # 動画上の関節位置
    for fno, joints_path in enumerate(glob.glob(osp.join(folder_path, '**/*_joints.json'))):
        logger.info(f"■ fno: {fno} -----")

        frame_joints = {}
        with open(joints_path, 'r') as f:
            frame_joints = json.load(f)

        bf = VmdBoneFrame(fno)
        bf.set_name("センター")
        bf.position = calc_center(frame_joints)
        motion.regist_bf(bf, bf.name, bf.fno)

        for jname, (bone_name, calc_bone, name_list, parent_list, ranges) in VMD_CONNECTIONS.items():
            if name_list is None:
                continue

            bf = VmdBoneFrame(fno)
            bf.set_name(bone_name)
            
            if calc_bone is None:
                if len(name_list) == 4:
                    rotation = calc_direction_qq(bf.fno, motion, frame_joints, *name_list)
                    initial = calc_bone_direction_qq(bf, motion, model, jname, *name_list)
                else:
                    rotation = calc_direction_qq2(bf.fno, motion, frame_joints, *name_list)
                    initial = calc_bone_direction_qq2(bf, motion, model, jname, *name_list)

                qq = MQuaternion()
                for parent_name in reversed(parent_list):
                    qq *= motion.calc_bf(parent_name, bf.fno).rotation.inverted()
                qq = qq * rotation * initial.inverted()

                if ranges:
                    # 可動域指定がある場合
                    x_qq, y_qq, z_qq, _ = separate_local_qq(bf.fno, bf.name, qq, model.get_local_x_axis(bf.name))
                    local_x_axis = model.get_local_x_axis(bf.name)
                    local_z_axis = MVector3D(0, 0, -1 * (-1 if "right" in jname else 1))
                    local_y_axis = MVector3D.crossProduct(local_x_axis, local_z_axis)
                    x_limited_qq = MQuaternion.fromAxisAndAngle(local_x_axis, max(ranges["x"]["min"], min(ranges["x"]["max"], x_qq.toDegree() * MVector3D.dotProduct(local_x_axis, x_qq.vector()))))
                    y_limited_qq = MQuaternion.fromAxisAndAngle(local_y_axis, max(ranges["y"]["min"], min(ranges["y"]["max"], y_qq.toDegree() * MVector3D.dotProduct(local_y_axis, y_qq.vector()))))
                    z_limited_qq = MQuaternion.fromAxisAndAngle(local_z_axis, max(ranges["z"]["min"], min(ranges["z"]["max"], z_qq.toDegree() * MVector3D.dotProduct(local_z_axis, z_qq.vector()))))
                    bf.rotation = y_limited_qq * x_limited_qq * z_limited_qq
                else:
                    bf.rotation = qq

                motion.regist_bf(bf, bf.name, bf.fno)

    # 動画内の半分は地面に足が着いていると見なす
    center_values = np.zeros((1, 3))
    for bf in motion.bones["センター"].values():
        center_values = np.insert(center_values, 0, np.array([bf.position.x(), bf.position.y(), bf.position.z()]), axis=0)
    
    center_median = np.median(center_values, axis=0)

    for bf in motion.bones["センター"].values():
        bf.position.setY(bf.position.y() - center_median[1])
    
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
    direction_from_vec = get_vec3(joints["joints"], direction_from_name)
    direction_to_vec = get_vec3(joints["joints"], direction_to_name)
    up_from_vec = get_vec3(joints["joints"], up_from_name)
    up_to_vec = get_vec3(joints["joints"], up_to_name)

    direction = (direction_to_vec - direction_from_vec).normalized()
    up = (up_to_vec - up_from_vec).normalized()
    cross = MVector3D.crossProduct(direction, up)
    qq = MQuaternion.fromDirection(direction, cross)

    return qq

def calc_direction_qq2(bf: VmdBoneFrame, motion: VmdMotion, joints: dict, direction_from_name: str, direction_to_name: str, up_from_name: str, up_to_name: str, cross_from_name: str, cross_to_name: str):
    direction_from_vec = get_vec3(joints["joints"], direction_from_name)
    direction_to_vec = get_vec3(joints["joints"], direction_to_name)
    up_from_vec = get_vec3(joints["joints"], up_from_name)
    up_to_vec = get_vec3(joints["joints"], up_to_name)
    cross_from_vec = get_vec3(joints["joints"], cross_from_name)
    cross_to_vec = get_vec3(joints["joints"], cross_to_name)

    direction = (direction_to_vec - direction_from_vec).normalized()
    up = (up_to_vec - up_from_vec).normalized()
    cross = (cross_to_vec - cross_from_vec).normalized()
    qq = MQuaternion.fromDirection(direction, MVector3D.crossProduct(up, cross))

    return qq

def calc_finger(bf: VmdBoneFrame, motion: VmdMotion, model: PmxModel, jname: str, default_joints: dict, frame_joints: dict, name_list: list, parent_list: list):
    rotation = calc_direction_qq(bf.fno, motion, frame_joints, *name_list)
    bone_initial = calc_bone_direction_qq(bf, motion, model, jname, *name_list)

    qq = MQuaternion()
    for parent_name in reversed(parent_list):
        qq *= motion.calc_bf(parent_name, bf.fno).rotation.inverted()
    qq = qq * rotation * bone_initial.inverted()

    _, _, z_qq, _ = separate_local_qq(bf.fno, bf.name, qq, model.get_local_x_axis(bf.name))
    z_limited_qq = MQuaternion.fromAxisAndAngle(MVector3D(0, 0, -1 * (-1 if "right" in jname else 1)), min(90, z_qq.toDegree()))
    bf.rotation = z_limited_qq

    motion.regist_bf(bf, bf.name, bf.fno)

def calc_bone_direction_qq(bf: VmdBoneFrame, motion: VmdMotion, model: PmxModel, jname: str, direction_from_name: str, direction_to_name: str, up_from_name: str, up_to_name: str):
    direction_from_vec = get_bone_vec3(model, direction_from_name)
    direction_to_vec = get_bone_vec3(model, direction_to_name)
    up_from_vec = get_bone_vec3(model, up_from_name)
    up_to_vec = get_bone_vec3(model, up_to_name)

    direction = (direction_to_vec - direction_from_vec).normalized()
    up = (up_to_vec - up_from_vec).normalized()
    qq = MQuaternion.fromDirection(direction, up)

    return qq

def calc_bone_direction_qq2(bf: VmdBoneFrame, motion: VmdMotion, model: PmxModel, jname: str, direction_from_name: str, direction_to_name: str, up_from_name: str, up_to_name: str, cross_from_name: str, cross_to_name: str):
    direction_from_vec = get_bone_vec3(model, direction_from_name)
    direction_to_vec = get_bone_vec3(model, direction_to_name)
    up_from_vec = get_bone_vec3(model, up_from_name)
    up_to_vec = get_bone_vec3(model, up_to_name)
    cross_from_vec = get_bone_vec3(model, cross_from_name)
    cross_to_vec = get_bone_vec3(model, cross_to_name)

    direction = (direction_to_vec - direction_from_vec).normalized()
    up = (up_to_vec - up_from_vec).normalized()
    cross = (cross_to_vec - cross_from_vec).normalized()
    qq = MQuaternion.fromDirection(direction, MVector3D.crossProduct(up, cross))

    return qq

def get_bone_vec3(model: PmxModel, joint_name: str):
    bone_name, _, _, _, _ = VMD_CONNECTIONS[joint_name]
    if bone_name in model.bones:
        return model.bones[bone_name].position
    
    return MVector3D()

def get_vec3(joints: dict, jname: str):
    if jname in joints:
        joint = joints[jname]
        return MVector3D(joint["x"], joint["y"], joint["z"]) / SCALE_MIKU
    else:
        if jname == "pelvis2":
            # 尾てい骨くらい
            right_hip_vec = get_vec3(joints, "right_hip")
            left_hip_vec = get_vec3(joints, "left_hip")
            return (right_hip_vec + left_hip_vec) / 2
    
    return MVector3D()

def calc_center(frame_joints: dict):
    # プロジェクション座標系の位置
    px = (frame_joints["proj_joints"]["pelvis"]["x"] - (frame_joints["image"]["width"] / 2)) * SCALE_MIKU
    py = ((frame_joints["image"]["height"] / 2) - frame_joints["proj_joints"]["pelvis"]["y"]) * SCALE_MIKU
    cz = frame_joints["depth"]["depth"] * SCALE_MIKU
    return MVector3D(px, py, cz)

def read_bone_csv(bone_csv_path: str):
    model = PmxModel()
    model.name = osp.splitext(osp.basename(bone_csv_path))[0]

    with open(bone_csv_path, "r", encoding=get_file_encoding(bone_csv_path)) as f:
        reader = csv.reader(f)

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

    # 首根元ボーン
    if "左肩" in model.bones and "右肩" in model.bones:
        neck_base_vertex = Vertex(-1, (model.bones["左肩"].position + model.bones["右肩"].position) / 2 + MVector3D(0, -0.1, 0), MVector3D(), [], [], Vertex.Bdef1(-1), -1)
        neck_base_vertex.position.setX(0)
        neck_base_bone = Bone("首根元", "base of neck", neck_base_vertex.position.copy(), -1, 0, 0)

        if "上半身2" in model.bones:
            # 上半身2がある場合、表示先は、上半身2
            neck_base_bone.parent_index = model.bones["上半身2"].index
            neck_base_bone.tail_index = model.bones["上半身2"].index
        elif "上半身" in model.bones:
            neck_base_bone.parent_index = model.bones["上半身"].index
            neck_base_bone.tail_index = model.bones["上半身"].index

        neck_base_bone.index = len(model.bones.keys())
        model.bones[neck_base_bone.name] = neck_base_bone
        model.bone_indexes[neck_base_bone.index] = neck_base_bone.name

    # 鼻ボーン
    if "頭" in model.bones and "首" in model.bones:
        nose_bone = Bone("鼻", "nose", MVector3D(0, model.bones["頭"].position.y(), model.bones["頭"].position.z() - 0.5), -1, 0, 0)
        nose_bone.parent_index = model.bones["首"].index
        nose_bone.tail_index = model.bones["頭"].index
        nose_bone.index = len(model.bones.keys())
        model.bones[nose_bone.name] = nose_bone
        model.bone_indexes[nose_bone.index] = nose_bone.name

    # 頭頂ボーン
    if "頭" in model.bones:
        head_top_bone = Bone("頭頂", "top of head", MVector3D(0, model.bones["頭"].position.y() + 1, 0), -1, 0, 0)
        head_top_bone.parent_index = model.bones["鼻"].index
        head_top_bone.index = len(model.bones.keys())
        model.bones[head_top_bone.name] = head_top_bone
        model.bone_indexes[head_top_bone.index] = head_top_bone.name

    if "右つま先" in model.bones:
        right_big_toe_bone = Bone("右足親指", "", model.bones["右つま先"].position + MVector3D(0.5, 0, 0), -1, 0, 0)
        right_big_toe_bone.parent_index = model.bones["右足首"].index
        right_big_toe_bone.index = len(model.bones.keys())
        model.bones[right_big_toe_bone.name] = right_big_toe_bone
        model.bone_indexes[right_big_toe_bone.index] = right_big_toe_bone.name

        right_small_toe_bone = Bone("右足小指", "", model.bones["右つま先"].position + MVector3D(-0.5, 0, 0), -1, 0, 0)
        right_small_toe_bone.parent_index = model.bones["右足首"].index
        right_small_toe_bone.index = len(model.bones.keys())
        model.bones[right_small_toe_bone.name] = right_small_toe_bone
        model.bone_indexes[right_small_toe_bone.index] = right_small_toe_bone.name

    if "左つま先" in model.bones:
        left_big_toe_bone = Bone("左足親指", "", model.bones["左つま先"].position + MVector3D(-0.5, 0, 0), -1, 0, 0)
        left_big_toe_bone.parent_index = model.bones["左足首"].index
        left_big_toe_bone.index = len(model.bones.keys())
        model.bones[left_big_toe_bone.name] = left_big_toe_bone
        model.bone_indexes[left_big_toe_bone.index] = left_big_toe_bone.name

        left_small_toe_bone = Bone("左足小指", "", model.bones["左つま先"].position + MVector3D(0.5, 0, 0), -1, 0, 0)
        left_small_toe_bone.parent_index = model.bones["左足首"].index
        left_small_toe_bone.index = len(model.bones.keys())
        model.bones[left_small_toe_bone.name] = left_small_toe_bone
        model.bone_indexes[left_small_toe_bone.index] = left_small_toe_bone.name

    return model


VMD_CONNECTIONS = {
    'spine1': ("上半身", None, ['pelvis', 'spine1', 'left_shoulder', 'right_shoulder', 'spine1', 'spine2'], [], None),
    'spine2': ("上半身2", None, ['spine1', 'spine2', 'left_shoulder', 'right_shoulder', 'spine2', 'spine3'], ["上半身"], None),
    'spine3': ("首根元", None, None, None, None),
    'neck': ("首", None, ['spine3', 'neck', 'left_eye', 'right_eye', 'neck', 'nose'], ["上半身", "上半身2"], \
        {"x": {"min": -30, "max": 40}, "y": {"min": -20, "max": 20}, "z": {"min": -20, "max": 20}}),
    'head': ("頭", None, ['neck', 'head', 'left_eye', 'right_eye', 'neck', 'head'], ["上半身", "上半身2", "首"], \
        {"x": {"min": -10, "max": 20}, "y": {"min": -30, "max": 30}, "z": {"min": -20, "max": 20}}),
    'pelvis': ("下半身", None, None, None, None),
    'nose': ("鼻", None, None, None, None),
    'right_eye': ("右目", None, None, None, None),
    'left_eye': ("左目", None, None, None, None),
    'right_shoulder': ("右肩", None, ['spine3', 'right_shoulder', 'spine1', 'spine3', 'right_shoulder', 'left_shoulder'], ["上半身", "上半身2"], None),
    'left_shoulder': ("左肩", None, ['spine3', 'left_shoulder', 'spine1', 'spine3', 'left_shoulder', 'right_shoulder'], ["上半身", "上半身2"], None),
    'right_arm': ("右腕", None, ['right_shoulder', 'right_elbow', 'spine3', 'right_shoulder', 'right_shoulder', 'right_elbow'], ["上半身", "上半身2", "右肩"], None),
    'left_arm': ("左腕", None, ['left_shoulder', 'left_elbow', 'spine3', 'left_shoulder', 'left_shoulder', 'left_elbow'], ["上半身", "上半身2", "左肩"], None),
    'right_elbow': ("右ひじ", None, ['right_elbow', 'right_wrist', 'spine3', 'right_shoulder', 'right_elbow', 'right_wrist'], ["上半身", "上半身2", "右肩", "右腕"], None),
    'left_elbow': ("左ひじ", None, ['left_elbow', 'left_wrist', 'spine3', 'left_shoulder', 'left_elbow', 'left_wrist'], ["上半身", "上半身2", "左肩", "左腕"], None),
    'right_wrist': ("右手首", None, ['right_wrist', 'right_middle1', 'right_index1', 'right_pinky1', 'right_wrist', 'right_middle1'], ["上半身", "上半身2", "右肩", "右腕", "右ひじ"], None),
    'left_wrist': ("左手首", None, ['left_wrist', 'left_middle1', 'left_index1', 'left_pinky1', 'left_wrist', 'left_middle1'], ["上半身", "上半身2", "左肩", "左腕", "左ひじ"], None),
    'right_thumb1': ("右親指１", None, None, None, None),
    'left_thumb1': ("左親指１", None, None, None, None),
    'right_middle1': ("右中指１", None, None, None, None),
    'left_middle1': ("左中指１", None, None, None, None),
    'right_index1': ("右人指１", None, None, None, None),
    'left_index1': ("左人指１", None, None, None, None),
    'right_pinky1': ("右小指１", None, None, None, None),
    'left_pinky1': ("左小指１", None, None, None, None),
    'pelvis': ("下半身", None, ['spine1', 'pelvis', 'left_hip', 'right_hip', 'pelvis', 'pelvis2'], [], None),
    'pelvis2': ("尾てい骨", None, None, None, None),
    'right_hip': ("右足", None, ['right_hip', 'right_knee', 'pelvis2', 'right_hip', 'right_hip', 'right_knee'], ["下半身"], None),
    'left_hip': ("左足", None, ['left_hip', 'left_knee', 'pelvis2', 'left_hip', 'left_hip', 'left_knee'], ["下半身"], None),
    'right_knee': ("右ひざ", None, ['right_knee', 'right_ankle', 'pelvis2', 'right_hip', 'right_knee', 'right_ankle'], ["下半身", "右足"], None),
    'left_knee': ("左ひざ", None, ['left_knee', 'left_ankle', 'pelvis2', 'left_hip', 'left_knee', 'left_ankle'], ["下半身", "左足"], None),
    'right_ankle': ("右足首", None, ['right_ankle', 'right_big_toe', 'right_big_toe', 'right_small_toe', 'right_ankle', 'right_big_toe'], ["下半身", "右足", "右ひざ"], None),
    'left_ankle': ("左足首", None, ['left_ankle', 'left_big_toe', 'left_big_toe', 'left_small_toe', 'left_ankle', 'left_big_toe'], ["下半身", "左足", "左ひざ"], None),
    'right_big_toe': ("右足親指", None, None, None, None),
    'right_small_toe': ("右足小指", None, None, None, None),
    'left_big_toe': ("左足親指", None, None, None, None),
    'left_small_toe': ("左足小指", None, None, None, None),
    'right_index1': ("右人指１", None, ['right_index1', 'right_index2', 'right_index1', 'right_pinky1', 'right_wrist', 'right_index1'], ["上半身", "上半身2", "右肩", "右腕", "右ひじ", "右手首"], None),
    'left_index1': ("左人指１", None, ['left_index1', 'left_index2', 'left_index1', 'left_pinky1', 'left_wrist', 'left_index1'], ["上半身", "上半身2", "左肩", "左腕", "左ひじ", "左手首"], None),
    'right_index2': ("右人指２", None, ['right_index2', 'right_index3', 'right_index1', 'right_pinky1', 'right_index1', 'right_index2'], ["上半身", "上半身3", "右肩", "右腕", "右ひじ", "右手首", "右人指１"], None),
    'left_index2': ("左人指２", None, ['left_index2', 'left_index3', 'left_index1', 'left_pinky1', 'left_index1', 'left_index2'], ["上半身", "上半身3", "左肩", "左腕", "左ひじ", "左手首", "左人指１"], None),
    'right_index3': ("右人指３", None, ['right_index3', 'right_index', 'right_index1', 'right_pinky1', 'right_index2', 'right_index3'], ["上半身", "上半身3", "右肩", "右腕", "右ひじ", "右手首", "右人指１", "右人指２"], None),
    'left_index3': ("左人指３", None, ['left_index3', 'left_index', 'left_index1', 'left_pinky1', 'left_index2', 'left_index3'], ["上半身", "上半身3", "左肩", "左腕", "左ひじ", "左手首", "左人指１", "左人指２"], None),
    'right_index': ("右人差指先", None, None, None, None),
    'left_index': ("左人差指先", None, None, None, None),
    'right_middle1': ("右中指１", None, ['right_middle1', 'right_middle2', 'right_index1', 'right_pinky1', 'right_wrist', 'right_middle1'], ["上半身", "上半身2", "右肩", "右腕", "右ひじ", "右手首"], None),
    'left_middle1': ("左中指１", None, ['left_middle1', 'left_middle2', 'left_index1', 'left_pinky1', 'left_wrist', 'left_middle1'], ["上半身", "上半身2", "左肩", "左腕", "左ひじ", "左手首"], None),
    'right_middle2': ("右中指２", None, ['right_middle2', 'right_middle3', 'right_index1', 'right_pinky1', 'right_middle1', 'right_middle2'], ["上半身", "上半身3", "右肩", "右腕", "右ひじ", "右手首", "右中指１"], None),
    'left_middle2': ("左中指２", None, ['left_middle2', 'left_middle3', 'left_index1', 'left_pinky1', 'left_middle1', 'left_middle2'], ["上半身", "上半身3", "左肩", "左腕", "左ひじ", "左手首", "左中指１"], None),
    'right_middle3': ("右中指３", None, ['right_middle3', 'right_middle', 'right_index1', 'right_pinky1', 'right_middle2', 'right_middle3'], ["上半身", "上半身3", "右肩", "右腕", "右ひじ", "右手首", "右中指１", "右中指２"], None),
    'left_middle3': ("左中指３", None, ['left_middle3', 'left_middle', 'left_index1', 'left_pinky1', 'left_middle2', 'left_middle3'], ["上半身", "上半身3", "左肩", "左腕", "左ひじ", "左手首", "左中指１", "左中指２"], None),
    'right_middle': ("右中指先", None, None, None, None),
    'left_middle': ("左中指先", None, None, None, None),
    'right_ring1': ("右薬指１", None, ['right_ring1', 'right_ring2', 'right_index1', 'right_pinky1', 'right_wrist', 'right_ring1'], ["上半身", "上半身2", "右肩", "右腕", "右ひじ", "右手首"], None),
    'left_ring1': ("左薬指１", None, ['left_ring1', 'left_ring2', 'left_index1', 'left_pinky1', 'left_wrist', 'left_ring1'], ["上半身", "上半身2", "左肩", "左腕", "左ひじ", "左手首"], None),
    'right_ring2': ("右薬指２", None, ['right_ring2', 'right_ring3', 'right_index1', 'right_pinky1', 'right_ring1', 'right_ring2'], ["上半身", "上半身3", "右肩", "右腕", "右ひじ", "右手首", "右薬指１"], None),
    'left_ring2': ("左薬指２", None, ['left_ring2', 'left_ring3', 'left_index1', 'left_pinky1', 'left_ring1', 'left_ring2'], ["上半身", "上半身3", "左肩", "左腕", "左ひじ", "左手首", "左薬指１"], None),
    'right_ring3': ("右薬指３", None, ['right_ring3', 'right_ring', 'right_index1', 'right_pinky1', 'right_ring2', 'right_ring3'], ["上半身", "上半身3", "右肩", "右腕", "右ひじ", "右手首", "右薬指１", "右薬指２"], None),
    'left_ring3': ("左薬指３", None, ['left_ring3', 'left_ring', 'left_index1', 'left_pinky1', 'left_ring2', 'left_ring3'], ["上半身", "上半身3", "左肩", "左腕", "左ひじ", "左手首", "左薬指１", "左薬指２"], None),
    'right_ring': ("右薬指先", None, None, None, None),
    'left_ring': ("左薬指先", None, None, None, None),
    'right_pinky1': ("右小指１", None, ['right_pinky1', 'right_pinky2', 'right_index1', 'right_pinky1', 'right_wrist', 'right_pinky1'], ["上半身", "上半身2", "右肩", "右腕", "右ひじ", "右手首"], None),
    'left_pinky1': ("左小指１", None, ['left_pinky1', 'left_pinky2', 'left_index1', 'left_pinky1', 'left_wrist', 'left_pinky1'], ["上半身", "上半身2", "左肩", "左腕", "左ひじ", "左手首"], None),
    'right_pinky2': ("右小指２", None, ['right_pinky2', 'right_pinky3', 'right_index1', 'right_pinky1', 'right_pinky1', 'right_pinky2'], ["上半身", "上半身3", "右肩", "右腕", "右ひじ", "右手首", "右小指１"], None),
    'left_pinky2': ("左小指２", None, ['left_pinky2', 'left_pinky3', 'left_index1', 'left_pinky1', 'left_pinky1', 'left_pinky2'], ["上半身", "上半身3", "左肩", "左腕", "左ひじ", "左手首", "左小指１"], None),
    'right_pinky3': ("右小指３", None, ['right_pinky3', 'right_pinky', 'right_index1', 'right_pinky1', 'right_pinky2', 'right_pinky3'], ["上半身", "上半身3", "右肩", "右腕", "右ひじ", "右手首", "右小指１", "右小指２"], None),
    'left_pinky3': ("左小指３", None, ['left_pinky3', 'left_pinky', 'left_index1', 'left_pinky1', 'left_pinky2', 'left_pinky3'], ["上半身", "上半身3", "左肩", "左腕", "左ひじ", "左手首", "左小指１", "左小指２"], None),
    'right_pinky': ("右小指先", None, None, None, None),
    'left_pinky': ("左小指先", None, None, None, None),

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


