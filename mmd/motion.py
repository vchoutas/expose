# -*- coding: utf-8 -*-
import os
import argparse
import glob
import re
import json
import csv
import datetime

# import vision essentials
import numpy as np
from tqdm import tqdm

from mmd.utils.MLogger import MLogger
from mmd.utils.MServiceUtils import sort_by_numeric
from lighttrack.visualizer.detection_visualizer import draw_bbox

from mmd.mmd.VmdWriter import VmdWriter
from mmd.module.MMath import MMatrix4x4, MQuaternion, MVector3D
from mmd.mmd.VmdData import VmdBoneFrame, VmdMotion, VmdShowIkFrame, VmdInfoIk, OneEuroFilter
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

        motion_dir_path = os.path.join(args.img_dir, "motion")
        os.makedirs(motion_dir_path, exist_ok=True)

        # 全人物分の順番別フォルダ
        ordered_person_dir_pathes = sorted(glob.glob(os.path.join(args.img_dir, "ordered", "*")), key=sort_by_numeric)

        frame_pattern = re.compile(r'^frame_(\d+)\.')

        # モデルをCSVから読み込む
        model = read_bone_csv(args.bone_config)

        for oidx, ordered_person_dir_path in enumerate(ordered_person_dir_pathes):    
            logger.info("【No.%s】モーション生成準備開始", f"{oidx:03}", decoration=MLogger.DECORATION_BOX)

            frame_json_pathes = sorted(glob.glob(os.path.join(ordered_person_dir_path, "frame_*.json")), key=sort_by_numeric)

            all_joints = {}
        
            for frame_json_path in tqdm(frame_json_pathes, desc=f"Read No.{oidx:03} ... "):
                m = frame_pattern.match(os.path.basename(frame_json_path))
                if m:
                    # キーフレの場所を確定（間が空く場合もある）
                    fno = int(m.groups()[0])

                    frame_joints = {}
                    with open(frame_json_path, 'r') as f:
                        frame_joints = json.load(f)
                    
                    # ジョイントグローバル座標を保持
                    for jname, joint in frame_joints["joints"].items():
                        if (jname, 'x') not in all_joints:
                            all_joints[(jname, 'x')] = {}

                        if (jname, 'y') not in all_joints:
                            all_joints[(jname, 'y')] = {}

                        if (jname, 'z') not in all_joints:
                            all_joints[(jname, 'z')] = {}
                        
                        all_joints[(jname, 'x')][fno] = joint["x"]
                        all_joints[(jname, 'y')][fno] = joint["y"]
                        all_joints[(jname, 'z')][fno] = joint["z"]

            # スムージング
            for (jname, axis), joints in tqdm(all_joints.items(), desc=f"Filter No.{oidx:03} ... "):
                filter = OneEuroFilter(freq=30, mincutoff=1, beta=0.01, dcutoff=1)
                for fno, joint in joints.items():
                    all_joints[(jname, axis)][fno] = filter(joint, fno)
            
            # 出力
            for frame_json_path in tqdm(frame_json_pathes, desc=f"Save No.{oidx:03} ... "):
                m = frame_pattern.match(os.path.basename(frame_json_path))
                if m:
                    # キーフレの場所を確定（間が空く場合もある）
                    fno = int(m.groups()[0])

                    frame_joints = {}
                    with open(frame_json_path, 'r', encoding='utf-8') as f:
                        frame_joints = json.load(f)
                    
                    # ジョイントグローバル座標を保存
                    for jname, joint in frame_joints["joints"].items():
                        frame_joints["joints"][jname]["x"] = all_joints[(jname, 'x')][fno]
                        frame_joints["joints"][jname]["y"] = all_joints[(jname, 'y')][fno]
                        frame_joints["joints"][jname]["z"] = all_joints[(jname, 'z')][fno]

                    smooth_json_path = os.path.join(ordered_person_dir_path, f"smooth_{fno:012}.json")
                    
                    with open(smooth_json_path, 'w', encoding='utf-8') as f:
                        json.dump(frame_joints, f, indent=4)

        smooth_pattern = re.compile(r'^smooth_(\d+)\.')

        for oidx, ordered_person_dir_path in enumerate(ordered_person_dir_pathes):    
            logger.info("【No.%s】モーション生成開始", f"{oidx:03}", decoration=MLogger.DECORATION_BOX)

            smooth_json_pathes = sorted(glob.glob(os.path.join(ordered_person_dir_path, "smooth_*.json")), key=sort_by_numeric)

            motion = VmdMotion()
        
            for smooth_json_path in tqdm(smooth_json_pathes, desc=f"No.{oidx:03} ... "):
                m = smooth_pattern.match(os.path.basename(smooth_json_path))
                if m:
                    # キーフレの場所を確定（間が空く場合もある）
                    fno = int(m.groups()[0])

                    frame_joints = {}
                    with open(smooth_json_path, 'r', encoding='utf-8') as f:
                        frame_joints = json.load(f)

                    bf = VmdBoneFrame(fno)
                    bf.set_name("センター")
                    bf.position = calc_center(frame_joints)
                    motion.regist_bf(bf, bf.name, bf.fno)

                    for jname, (bone_name, calc_bone, name_list, parent_list, ranges, is_hand) in VMD_CONNECTIONS.items():
                        if name_list is None:
                            continue

                        if not args.hand_motion and is_hand:
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
            
            # とりあえずIKoff
            left_leg_ik = VmdShowIkFrame()
            left_leg_ik.show = 1
            left_leg_ik.ik.append(VmdInfoIk(name="左足ＩＫ", onoff=0))
            motion.showiks.append(left_leg_ik)

            right_leg_ik = VmdShowIkFrame()
            right_leg_ik.show = 1
            right_leg_ik.ik.append(VmdInfoIk(name="右足ＩＫ", onoff=0))
            motion.showiks.append(right_leg_ik)

            left_toe_ik = VmdShowIkFrame()
            left_toe_ik.show = 1
            left_toe_ik.ik.append(VmdInfoIk(name="左つま先ＩＫ", onoff=0))
            motion.showiks.append(left_toe_ik)

            right_toe_ik = VmdShowIkFrame()
            right_toe_ik.show = 1
            right_toe_ik.ik.append(VmdInfoIk(name="右つま先ＩＫ", onoff=0))
            motion.showiks.append(right_toe_ik)

            motion_path = os.path.join(motion_dir_path, "output_no{0:03}_{1}.vmd".format(oidx, datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
            writer = VmdWriter(model, motion, motion_path)
            writer.write()

            logger.info("【No.%s】モーション生成開始: %s", f"{oidx:03}", motion_path, decoration=MLogger.DECORATION_BOX)

        logger.info('モーション生成処理全件終了', decoration=MLogger.DECORATION_BOX)

        return True
    except Exception as e:
        logger.critical("モーション生成で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        return False


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
    bone_name, _, _, _, _, _ = VMD_CONNECTIONS[joint_name]
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

    py = 0
    if float(frame_joints["proj_joints"]["left_knee"]["y"]) < float(frame_joints["image"]["height"]):
        # ひざまで映っている場合のみセンター設定
        py = ((frame_joints["image"]["height"] / 2) - frame_joints["proj_joints"]["pelvis"]["y"]) * SCALE_MIKU

    cz = frame_joints["depth"]["depth"] * SCALE_MIKU
    return MVector3D(px, py, cz)


def read_bone_csv(bone_csv_path: str):
    model = PmxModel()
    model.name = os.path.splitext(os.path.basename(bone_csv_path))[0]

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
        neck_base_vertex = Vertex(-1, (model.bones["左肩"].position + model.bones["右肩"].position) / 2 + MVector3D(0, -0.1, 0), MVector3D(), [], [], Bdef1(-1), -1)
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
    'spine1': ("上半身", None, ['pelvis', 'spine1', 'left_shoulder', 'right_shoulder', 'spine1', 'spine2'], [], None, False),
    'spine2': ("上半身2", None, ['spine1', 'spine2', 'left_shoulder', 'right_shoulder', 'spine2', 'spine3'], ["上半身"], None, False),
    'spine3': ("首根元", None, None, None, None, False),
    'neck': ("首", None, ['spine3', 'neck', 'left_eye', 'right_eye', 'neck', 'nose'], ["上半身", "上半身2"], \
        {"x": {"min": -30, "max": 40}, "y": {"min": -20, "max": 20}, "z": {"min": -20, "max": 20}}, False),
    'head': ("頭", None, ['neck', 'head', 'left_eye', 'right_eye', 'neck', 'head'], ["上半身", "上半身2", "首"], \
        {"x": {"min": -10, "max": 20}, "y": {"min": -30, "max": 30}, "z": {"min": -20, "max": 20}}, False),
    'pelvis': ("下半身", None, None, None, None, False),
    'nose': ("鼻", None, None, None, None, False),
    'right_eye': ("右目", None, None, None, None, False),
    'left_eye': ("左目", None, None, None, None, False),
    'right_shoulder': ("右肩", None, ['spine3', 'right_shoulder', 'spine1', 'spine3', 'right_shoulder', 'left_shoulder'], ["上半身", "上半身2"], None, False),
    'left_shoulder': ("左肩", None, ['spine3', 'left_shoulder', 'spine1', 'spine3', 'left_shoulder', 'right_shoulder'], ["上半身", "上半身2"], None, False),
    'right_arm': ("右腕", None, ['right_shoulder', 'right_elbow', 'spine3', 'right_shoulder', 'right_shoulder', 'right_elbow'], ["上半身", "上半身2", "右肩"], None, False),
    'left_arm': ("左腕", None, ['left_shoulder', 'left_elbow', 'spine3', 'left_shoulder', 'left_shoulder', 'left_elbow'], ["上半身", "上半身2", "左肩"], None, False),
    'right_elbow': ("右ひじ", None, ['right_elbow', 'right_wrist', 'spine3', 'right_shoulder', 'right_elbow', 'right_wrist'], ["上半身", "上半身2", "右肩", "右腕"], \
        {"x": {"min": -10, "max": 10}, "y": {"min": -180, "max": 180}, "z": {"min": -180, "max": 180}}, False),
    'left_elbow': ("左ひじ", None, ['left_elbow', 'left_wrist', 'spine3', 'left_shoulder', 'left_elbow', 'left_wrist'], ["上半身", "上半身2", "左肩", "左腕"], \
        {"x": {"min": -10, "max": 10}, "y": {"min": -180, "max": 180}, "z": {"min": -180, "max": 180}}, False),
    'right_wrist': ("右手首", None, ['right_wrist', 'right_middle1', 'right_index1', 'right_pinky1', 'right_wrist', 'right_middle1'], ["上半身", "上半身2", "右肩", "右腕", "右ひじ"], \
        {"x": {"min": -90, "max": 90}, "y": {"min": -30, "max": 30}, "z": {"min": -90, "max": 90}}, True),
    'left_wrist': ("左手首", None, ['left_wrist', 'left_middle1', 'left_index1', 'left_pinky1', 'left_wrist', 'left_middle1'], ["上半身", "上半身2", "左肩", "左腕", "左ひじ"], \
        {"x": {"min": -90, "max": 90}, "y": {"min": -30, "max": 30}, "z": {"min": -90, "max": 90}}, True),
    'pelvis': ("下半身", None, ['spine1', 'pelvis', 'left_hip', 'right_hip', 'pelvis', 'pelvis2'], [], None, False),
    'pelvis2': ("尾てい骨", None, None, None, None, False),
    'right_hip': ("右足", None, ['right_hip', 'right_knee', 'pelvis2', 'right_hip', 'right_hip', 'right_knee'], ["下半身"], None, False),
    'left_hip': ("左足", None, ['left_hip', 'left_knee', 'pelvis2', 'left_hip', 'left_hip', 'left_knee'], ["下半身"], None, False),
    'right_knee': ("右ひざ", None, ['right_knee', 'right_ankle', 'pelvis2', 'right_hip', 'right_knee', 'right_ankle'], ["下半身", "右足"], None, False),
    'left_knee': ("左ひざ", None, ['left_knee', 'left_ankle', 'pelvis2', 'left_hip', 'left_knee', 'left_ankle'], ["下半身", "左足"], None, False),
    'right_ankle': ("右足首", None, ['right_ankle', 'right_big_toe', 'right_big_toe', 'right_small_toe', 'right_ankle', 'right_big_toe'], ["下半身", "右足", "右ひざ"], None, False),
    'left_ankle': ("左足首", None, ['left_ankle', 'left_big_toe', 'left_big_toe', 'left_small_toe', 'left_ankle', 'left_big_toe'], ["下半身", "左足", "左ひざ"], None, False),
    'right_big_toe': ("右足親指", None, None, None, None, False),
    'right_small_toe': ("右足小指", None, None, None, None, False),
    'left_big_toe': ("左足親指", None, None, None, None, False),
    'left_small_toe': ("左足小指", None, None, None, None, False),
    'right_index1': ("右人指１", None, ['right_index1', 'right_index2', 'right_index1', 'right_pinky1', 'right_wrist', 'right_index1'], ["上半身", "上半身2", "右肩", "右腕", "右ひじ", "右手首"], \
        {"x": {"min": -3, "max": 3}, "y": {"min": -30, "max": 30}, "z": {"min": -10, "max": 130}}, True),
    'left_index1': ("左人指１", None, ['left_index1', 'left_index2', 'left_index1', 'left_pinky1', 'left_wrist', 'left_index1'], ["上半身", "上半身2", "左肩", "左腕", "左ひじ", "左手首"], \
        {"x": {"min": -3, "max": 3}, "y": {"min": -30, "max": 30}, "z": {"min": -10, "max": 130}}, True),
    'right_index2': ("右人指２", None, ['right_index2', 'right_index3', 'right_index1', 'right_pinky1', 'right_index1', 'right_index2'], ["上半身", "上半身3", "右肩", "右腕", "右ひじ", "右手首", "右人指１"], \
        {"x": {"min": -3, "max": 3}, "y": {"min": -3, "max": 3}, "z": {"min": -10, "max": 130}}, True),
    'left_index2': ("左人指２", None, ['left_index2', 'left_index3', 'left_index1', 'left_pinky1', 'left_index1', 'left_index2'], ["上半身", "上半身3", "左肩", "左腕", "左ひじ", "左手首", "左人指１"], \
        {"x": {"min": -3, "max": 3}, "y": {"min": -3, "max": 3}, "z": {"min": -10, "max": 130}}, True),
    'right_index3': ("右人指３", None, ['right_index3', 'right_index', 'right_index1', 'right_pinky1', 'right_index2', 'right_index3'], ["上半身", "上半身3", "右肩", "右腕", "右ひじ", "右手首", "右人指１", "右人指２"], \
        {"x": {"min": -3, "max": 3}, "y": {"min": -3, "max": 3}, "z": {"min": -10, "max": 130}}, True),
    'left_index3': ("左人指３", None, ['left_index3', 'left_index', 'left_index1', 'left_pinky1', 'left_index2', 'left_index3'], ["上半身", "上半身3", "左肩", "左腕", "左ひじ", "左手首", "左人指１", "左人指２"], \
        {"x": {"min": -3, "max": 3}, "y": {"min": -3, "max": 3}, "z": {"min": -10, "max": 130}}, True),
    'right_index': ("右人差指先", None, None, None, None, True),
    'left_index': ("左人差指先", None, None, None, None, True),
    'right_middle1': ("右中指１", None, ['right_middle1', 'right_middle2', 'right_index1', 'right_pinky1', 'right_wrist', 'right_middle1'], ["上半身", "上半身2", "右肩", "右腕", "右ひじ", "右手首"], \
        {"x": {"min": -3, "max": 3}, "y": {"min": -30, "max": 30}, "z": {"min": -10, "max": 130}}, True),
    'left_middle1': ("左中指１", None, ['left_middle1', 'left_middle2', 'left_index1', 'left_pinky1', 'left_wrist', 'left_middle1'], ["上半身", "上半身2", "左肩", "左腕", "左ひじ", "左手首"], \
        {"x": {"min": -3, "max": 3}, "y": {"min": -30, "max": 30}, "z": {"min": -10, "max": 130}}, True),
    'right_middle2': ("右中指２", None, ['right_middle2', 'right_middle3', 'right_index1', 'right_pinky1', 'right_middle1', 'right_middle2'], ["上半身", "上半身3", "右肩", "右腕", "右ひじ", "右手首", "右中指１"], \
        {"x": {"min": -3, "max": 3}, "y": {"min": -3, "max": 3}, "z": {"min": -10, "max": 130}}, True),
    'left_middle2': ("左中指２", None, ['left_middle2', 'left_middle3', 'left_index1', 'left_pinky1', 'left_middle1', 'left_middle2'], ["上半身", "上半身3", "左肩", "左腕", "左ひじ", "左手首", "左中指１"], \
        {"x": {"min": -3, "max": 3}, "y": {"min": -3, "max": 3}, "z": {"min": -10, "max": 130}}, True),
    'right_middle3': ("右中指３", None, ['right_middle3', 'right_middle', 'right_index1', 'right_pinky1', 'right_middle2', 'right_middle3'], ["上半身", "上半身3", "右肩", "右腕", "右ひじ", "右手首", "右中指１", "右中指２"], \
        {"x": {"min": -3, "max": 3}, "y": {"min": -3, "max": 3}, "z": {"min": -10, "max": 130}}, True),
    'left_middle3': ("左中指３", None, ['left_middle3', 'left_middle', 'left_index1', 'left_pinky1', 'left_middle2', 'left_middle3'], ["上半身", "上半身3", "左肩", "左腕", "左ひじ", "左手首", "左中指１", "左中指２"], \
        {"x": {"min": -3, "max": 3}, "y": {"min": -3, "max": 3}, "z": {"min": -10, "max": 130}}, True),
    'right_middle': ("右中指先", None, None, None, None, True),
    'left_middle': ("左中指先", None, None, None, None, True),
    'right_ring1': ("右薬指１", None, ['right_ring1', 'right_ring2', 'right_index1', 'right_pinky1', 'right_wrist', 'right_ring1'], ["上半身", "上半身2", "右肩", "右腕", "右ひじ", "右手首"], \
        {"x": {"min": -3, "max": 3}, "y": {"min": -30, "max": 30}, "z": {"min": -10, "max": 130}}, True),
    'left_ring1': ("左薬指１", None, ['left_ring1', 'left_ring2', 'left_index1', 'left_pinky1', 'left_wrist', 'left_ring1'], ["上半身", "上半身2", "左肩", "左腕", "左ひじ", "左手首"], \
        {"x": {"min": -3, "max": 3}, "y": {"min": -30, "max": 30}, "z": {"min": -10, "max": 130}}, True),
    'right_ring2': ("右薬指２", None, ['right_ring2', 'right_ring3', 'right_index1', 'right_pinky1', 'right_ring1', 'right_ring2'], ["上半身", "上半身3", "右肩", "右腕", "右ひじ", "右手首", "右薬指１"], \
        {"x": {"min": -3, "max": 3}, "y": {"min": -3, "max": 3}, "z": {"min": -10, "max": 130}}, True),
    'left_ring2': ("左薬指２", None, ['left_ring2', 'left_ring3', 'left_index1', 'left_pinky1', 'left_ring1', 'left_ring2'], ["上半身", "上半身3", "左肩", "左腕", "左ひじ", "左手首", "左薬指１"], \
        {"x": {"min": -3, "max": 3}, "y": {"min": -3, "max": 3}, "z": {"min": -10, "max": 130}}, True),
    'right_ring3': ("右薬指３", None, ['right_ring3', 'right_ring', 'right_index1', 'right_pinky1', 'right_ring2', 'right_ring3'], ["上半身", "上半身3", "右肩", "右腕", "右ひじ", "右手首", "右薬指１", "右薬指２"], \
        {"x": {"min": -3, "max": 3}, "y": {"min": -3, "max": 3}, "z": {"min": -10, "max": 130}}, True),
    'left_ring3': ("左薬指３", None, ['left_ring3', 'left_ring', 'left_index1', 'left_pinky1', 'left_ring2', 'left_ring3'], ["上半身", "上半身3", "左肩", "左腕", "左ひじ", "左手首", "左薬指１", "左薬指２"], \
        {"x": {"min": -3, "max": 3}, "y": {"min": -3, "max": 3}, "z": {"min": -10, "max": 130}}, True),
    'right_ring': ("右薬指先", None, None, None, None, True),
    'left_ring': ("左薬指先", None, None, None, None, True),
    'right_pinky1': ("右小指１", None, ['right_pinky1', 'right_pinky2', 'right_index1', 'right_pinky1', 'right_wrist', 'right_pinky1'], ["上半身", "上半身2", "右肩", "右腕", "右ひじ", "右手首"], \
        {"x": {"min": -3, "max": 3}, "y": {"min": -30, "max": 30}, "z": {"min": -10, "max": 130}}, True),
    'left_pinky1': ("左小指１", None, ['left_pinky1', 'left_pinky2', 'left_index1', 'left_pinky1', 'left_wrist', 'left_pinky1'], ["上半身", "上半身2", "左肩", "左腕", "左ひじ", "左手首"], \
        {"x": {"min": -3, "max": 3}, "y": {"min": -30, "max": 30}, "z": {"min": -10, "max": 130}}, True),
    'right_pinky2': ("右小指２", None, ['right_pinky2', 'right_pinky3', 'right_index1', 'right_pinky1', 'right_pinky1', 'right_pinky2'], ["上半身", "上半身3", "右肩", "右腕", "右ひじ", "右手首", "右小指１"], \
        {"x": {"min": -3, "max": 3}, "y": {"min": -3, "max": 3}, "z": {"min": -10, "max": 130}}, True),
    'left_pinky2': ("左小指２", None, ['left_pinky2', 'left_pinky3', 'left_index1', 'left_pinky1', 'left_pinky1', 'left_pinky2'], ["上半身", "上半身3", "左肩", "左腕", "左ひじ", "左手首", "左小指１"], \
        {"x": {"min": -3, "max": 3}, "y": {"min": -3, "max": 3}, "z": {"min": -10, "max": 130}}, True),
    'right_pinky3': ("右小指３", None, ['right_pinky3', 'right_pinky', 'right_index1', 'right_pinky1', 'right_pinky2', 'right_pinky3'], ["上半身", "上半身3", "右肩", "右腕", "右ひじ", "右手首", "右小指１", "右小指２"], \
        {"x": {"min": -3, "max": 3}, "y": {"min": -3, "max": 3}, "z": {"min": -10, "max": 130}}, True),
    'left_pinky3': ("左小指３", None, ['left_pinky3', 'left_pinky', 'left_index1', 'left_pinky1', 'left_pinky2', 'left_pinky3'], ["上半身", "上半身3", "左肩", "左腕", "左ひじ", "左手首", "左小指１", "左小指２"], \
        {"x": {"min": -3, "max": 3}, "y": {"min": -3, "max": 3}, "z": {"min": -10, "max": 130}}, True),
    'right_pinky': ("右小指先", None, None, None, None, True),
    'left_pinky': ("左小指先", None, None, None, None, True),
}
