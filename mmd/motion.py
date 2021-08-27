# -*- coding: utf-8 -*-
from numpy.core.fromnumeric import trace
from numpy.lib.function_base import average, flip
from mmd.module.MParams import BoneLinks
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
import shutil

from mmd.utils import MServiceUtils

from mmd.utils.MLogger import MLogger
from mmd.utils.MServiceUtils import sort_by_numeric
from lighttrack.visualizer.detection_visualizer import draw_bbox

from mmd.utils.MBezierUtils import join_value_2_bezier, R_x1_idxs, R_y1_idxs, R_x2_idxs, R_y2_idxs, MX_x1_idxs, MX_y1_idxs, MX_x2_idxs, MX_y2_idxs
from mmd.utils.MBezierUtils import MY_x1_idxs, MY_y1_idxs, MY_x2_idxs, MY_y2_idxs, MZ_x1_idxs, MZ_y1_idxs, MZ_x2_idxs, MZ_y2_idxs
from mmd.mmd.VmdWriter import VmdWriter
from mmd.module.MMath import MQuaternion, MVector3D, MVector2D, MMatrix4x4, MRect, MVector4D, fromEulerAngles
from mmd.mmd.VmdData import VmdBoneFrame, VmdMorphFrame, VmdMotion, VmdShowIkFrame, VmdInfoIk, OneEuroFilter
from mmd.mmd.PmxData import Material, PmxModel, Bone, Vertex, Bdef1, Ik, IkLink, DisplaySlot
from mmd.mmd.PmxWriter import PmxWriter
from mmd.mmd.PmxReader import PmxReader
from mmd.utils.MServiceUtils import get_file_encoding, calc_global_pos, separate_local_qq

logger = MLogger(__name__, level=1)

MIKU_METER = 12.5

def execute(args):
    try:
        logger.info('モーション生成処理開始: {0}', args.img_dir, decoration=MLogger.DECORATION_BOX)

        if not os.path.exists(args.img_dir):
            logger.error("指定された処理用ディレクトリが存在しません。: {0}", args.img_dir, decoration=MLogger.DECORATION_BOX)
            return False

        if not os.path.exists(os.path.join(args.img_dir, "smooth")):
            logger.error("指定された順番ディレクトリが存在しません。\n順番指定が完了していない可能性があります。: {0}", \
                         os.path.join(args.img_dir, "smooth"), decoration=MLogger.DECORATION_BOX)
            return False

        motion_dir_path = os.path.join(args.img_dir, "motion")
        os.makedirs(motion_dir_path, exist_ok=True)

        # モデルをCSVから読み込む
        miku_model = read_bone_csv(args.bone_config)
        process_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        # 全人物分の順番別フォルダ
        ordered_person_dir_pathes = sorted(glob.glob(os.path.join(args.img_dir, "smooth", "*")), key=sort_by_numeric)

        smooth_pattern = re.compile(r'^smooth_(\d+)\.')
        start_lower_depth = None

        pmx_writer = PmxWriter()
        vmd_writer = VmdWriter()

        for oidx, ordered_person_dir_path in enumerate(ordered_person_dir_pathes):    
            smooth_json_pathes = sorted(glob.glob(os.path.join(ordered_person_dir_path, "smooth_*.json")), key=sort_by_numeric)

            trace_mov_motion = VmdMotion()
            trace_rot_motion = VmdMotion()
            trace_miku_motion = VmdMotion()

            mp_trace_mov_motion = VmdMotion()
            mp_trace_rot_motion = VmdMotion()
            mp_trace_miku_motion = VmdMotion()

            # トレース用モデルを読み込む
            trace_model = PmxReader(args.trace_mov_model_config).read_data()
            mp_trace_model = PmxReader(args.trace_mov_model_config).read_data()

            # KEY: 処理対象ボーン名, VALUE: vecリスト
            target_bone_vecs = {}
            fnos = []
            face_fnos = []
            mp_fnos = []
        
            for mbname in ["全ての親", "センター", "グルーブ", "pelvis2", "head_tail", "right_wrist_tail", "left_wrist_tail", "params", \
                           "Groove", "mp_pelvis", "mp_spine1", "mp_neck", "mp_head", "mp_head_tail", "mp_left_collar", "mp_right_collar", "mp_pelvis2"]:
                target_bone_vecs[mbname] = {}
            
            logger.info("【No.{0}】モーション結果位置計算開始", f"{oidx:02d}", decoration=MLogger.DECORATION_LINE)

            with tqdm(total=(len(smooth_json_pathes) * (len(PMX_CONNECTIONS.keys()) * 2)), desc=f'No.{oidx:02d}') as pchar:
                for sidx, smooth_json_path in enumerate(smooth_json_pathes):
                    m = smooth_pattern.match(os.path.basename(smooth_json_path))
                    if m:
                        # キーフレの場所を確定（間が空く場合もある）
                        fno = int(m.groups()[0])
                        fnos.append(fno)

                        frame_joints = {}
                        with open(smooth_json_path, 'r', encoding='utf-8') as f:
                            frame_joints = json.load(f)

                        if "joints" in frame_joints:
                            for jname, joint in frame_joints["joints"].items():
                                if jname not in target_bone_vecs:
                                    target_bone_vecs[jname] = {}
                                target_bone_vecs[jname][fno] = np.array([joint["x"], joint["y"], joint["z"]]) * MIKU_METER
                                pchar.update(1)

                            for mbname in ["全ての親", "センター", "グルーブ"]:
                                target_bone_vecs[mbname][fno] = np.array([0, 0, 0])

                            # 下半身先
                            target_bone_vecs["pelvis2"][fno] = np.mean([target_bone_vecs["left_hip"][fno], target_bone_vecs["right_hip"][fno]], axis=0)
                            pchar.update(1)

                            # 頭先
                            target_bone_vecs["head_tail"][fno] = np.mean([target_bone_vecs["left_ear"][fno], target_bone_vecs["right_ear"][fno]], axis=0) + ((target_bone_vecs["head"][fno] - target_bone_vecs["neck"][fno]) / 2)
                            pchar.update(1)

                            # 手首先
                            for direction in ["left", "right"]:
                                target_bone_vecs[f"{direction}_wrist_tail"][fno] = np.mean([target_bone_vecs[f"{direction}_index1"][fno], target_bone_vecs[f"{direction}_middle1"][fno], target_bone_vecs[f"{direction}_ring1"][fno], target_bone_vecs[f"{direction}_pinky1"][fno]], axis=0)
                                pchar.update(1)
                        
                        for joint_group_name in ["mp_left_hand_joints", "mp_right_hand_joints"]:
                            if joint_group_name in frame_joints:
                                for jname, joint in frame_joints[joint_group_name].items():
                                    if jname not in target_bone_vecs:
                                        target_bone_vecs[jname] = {}
                                    target_bone_vecs[jname][fno] = np.array([joint["x"], joint["y"], joint["z"]]) * MIKU_METER
                                    pchar.update(1)
                        
                        if "mp_body_joints" in frame_joints:
                            mp_fnos.append(fno)

                            for jname, joint in frame_joints["mp_body_joints"].items():
                                if jname not in target_bone_vecs:
                                    target_bone_vecs[jname] = {}
                                target_bone_vecs[jname][fno] = np.array([joint["wx"], joint["wy"], joint["wz"]]) * MIKU_METER
                                pchar.update(1)

                            # グルーブ
                            target_bone_vecs["Groove"][fno] = np.array([0, 0, 0])

                            # 下半身
                            target_bone_vecs['mp_pelvis'][fno] = np.mean([target_bone_vecs['mp_left_hip'][fno], target_bone_vecs['mp_right_hip'][fno]], axis=0)

                            # 上半身
                            target_bone_vecs['mp_spine1'][fno] = np.mean([target_bone_vecs['mp_left_hip'][fno], target_bone_vecs['mp_right_hip'][fno]], axis=0)
                            
                            # 首
                            target_bone_vecs['mp_neck'][fno] = np.mean([target_bone_vecs['mp_left_shoulder'][fno], target_bone_vecs['mp_right_shoulder'][fno]], axis=0)

                            # 頭
                            target_bone_vecs['mp_head'][fno] = np.mean([target_bone_vecs['mp_left_ear'][fno], target_bone_vecs['mp_right_ear'][fno]], axis=0)

                            # 頭先
                            target_bone_vecs['mp_head_tail'][fno] = target_bone_vecs['mp_head'][fno] + ((target_bone_vecs['mp_head'][fno] - target_bone_vecs['mp_neck'][fno]) / 3)
                        
                            # 左肩
                            target_bone_vecs['mp_left_collar'][fno] = np.mean([target_bone_vecs['mp_left_shoulder'][fno], target_bone_vecs['mp_right_shoulder'][fno]], axis=0)
                        
                            # 右肩
                            target_bone_vecs['mp_right_collar'][fno] = np.mean([target_bone_vecs['mp_left_shoulder'][fno], target_bone_vecs['mp_right_shoulder'][fno]], axis=0)

                            # 下半身先
                            target_bone_vecs['mp_pelvis2'][fno] = target_bone_vecs['mp_pelvis'][fno] + ((target_bone_vecs['mp_neck'][fno] - target_bone_vecs['mp_spine1'][fno]) / 3)

                        if "mp_face_joints" in frame_joints:
                            face_fnos.append(fno)

                            for jidx, joint in frame_joints["mp_face_joints"].items():
                                for jname, pconn in PMX_CONNECTIONS.items():
                                    # 表情はINDEXから逆引きして設定する
                                    if "mp_index" in pconn and pconn["mp_index"] == jidx:
                                        if jname not in target_bone_vecs:
                                            target_bone_vecs[jname] = {}
                                        target_bone_vecs[jname][fno] = MVector3D([joint["x"], joint["y"], joint["z"]])
                                        pchar.update(1)

                        # パラも別保持
                        for group_name in ["image", "others", "camera", "depth", "bbox"]:
                            if fno not in target_bone_vecs["params"]:
                                target_bone_vecs["params"][fno] = {}
                            target_bone_vecs["params"][fno][group_name] = frame_joints[group_name]
                        
                        # 下半身・足の平面XY位置をパラに保持
                        target_bone_vecs["params"][fno]["proj_pelvis"] = frame_joints["proj_joints"]["pelvis"]
                        target_bone_vecs["params"][fno]["proj_left_foot"] = frame_joints["proj_joints"]["left_foot"]
                        target_bone_vecs["params"][fno]["proj_right_foot"] = frame_joints["proj_joints"]["right_foot"]
                        target_bone_vecs["params"][fno]["proj_left_heel"] = frame_joints["proj_joints"]["left_heel"]
                        target_bone_vecs["params"][fno]["proj_right_heel"] = frame_joints["proj_joints"]["right_heel"]

                        # 四肢の信頼度を保持
                        target_bone_vecs["params"][fno]["visibility"] = {}
                        for jname, mname in [("mp_nose", "頭先"), ("mp_body_right_index", "右人指３"), ("mp_body_left_index", "左人指３"), ("mp_right_hip", "右足"), ("mp_left_hip", "左足"), 
                                             ("mp_right_heel", "右かかと"), ("mp_left_heel", "左かかと"), ("mp_right_foot_index", "右つま先"), ("mp_left_foot_index", "左つま先"), 
                                             ("mp_body_right_wrist", "右手首"), ("mp_body_left_wrist", "左手首")]:
                            if "mp_body_joints" in frame_joints and jname in frame_joints["mp_body_joints"]:
                                target_bone_vecs["params"][fno]["visibility"][mname] = frame_joints["mp_body_joints"][jname]["visibility"]
                            else:
                                target_bone_vecs["params"][fno]["visibility"][mname] = 0
                        
                        for jname, mname in [("mp_spine1", "上半身"), ("mp_pelvis", "下半身")]:
                            if "mp_body_joints" in frame_joints:                        
                                # 体幹は両足の信頼度
                                target_bone_vecs["params"][fno]["visibility"][mname] = np.mean([frame_joints["mp_body_joints"]["mp_left_hip"]["visibility"], frame_joints["mp_body_joints"]["mp_right_hip"]["visibility"]])
                            else:
                                target_bone_vecs["params"][fno]["visibility"][mname] = 0
            
            # # 初期化
            # trace_model = init_trace_model()
            # mp_trace_model = init_trace_model()
            
            logger.info("【No.{0}】モーション(移動)計算開始", f"{oidx:02d}", decoration=MLogger.DECORATION_LINE)

            with tqdm(total=((len(PMX_CONNECTIONS.keys()) + len(MP_PMX_CONNECTIONS.keys())) * (len(fnos) + 1)), desc=f'No.{oidx:02d}') as pchar:
                for fno in fnos:
                    groove_bf = trace_mov_motion.calc_bf("グルーブ", fno)
                    groove_bf.key = True
                    trace_mov_motion.bones[groove_bf.name][fno] = groove_bf

                    center_bf = trace_mov_motion.calc_bf("センター", fno)
                    center_bf.key = True
                    trace_mov_motion.bones[center_bf.name][fno] = center_bf
                    pchar.update(1)

                for conns, target_trace_model, target_trace_mov_motion in [(PMX_CONNECTIONS, trace_model, trace_mov_motion), (MP_PMX_CONNECTIONS, mp_trace_model, mp_trace_mov_motion)]:
                    for jidx, (jname, pconn) in enumerate(conns.items()):
                        if "mp_index" in pconn:
                            # mediapipeのfaceは対象外
                            pchar.update(len(fnos))
                            continue
                        
                        if not args.hand_motion and "hand" in pconn:
                            # 手首から先はオプショナル
                            pchar.update(len(fnos))
                            continue

                        # # ボーン登録        
                        # create_bone(target_trace_model, jname, pconn, target_bone_vecs, miku_model, conns)

                        mname = pconn['mmd']
                        pmname = conns[pconn['parent']]['mmd'] if pconn['parent'] in conns else pconn['parent']
                            
                        if jname not in ["mp_left_wrist", "mp_right_wrist"]:
                            # モーションも登録
                            for fno in fnos:
                                if jname in target_bone_vecs and fno in target_bone_vecs[jname] and pconn['parent'] in target_bone_vecs and fno in target_bone_vecs[pconn['parent']]:
                                    joint = target_bone_vecs[jname][fno]
                                    parent_joint = target_bone_vecs[pconn['parent']][fno]
                                    
                                    trace_bf = target_trace_mov_motion.calc_bf(mname, fno)

                                    parent_bone = target_trace_model.bones[pmname] if pmname in target_trace_model.bones else None
                                    if not parent_bone:
                                        for b in trace_model.bones.values():
                                            if b.english_name == pmname:
                                                parent_bone = b
                                                break

                                    # # parentが全ての親の場合
                                    # trace_bf.position = MVector3D(joint) - target_trace_model.bones[mname].position
                                    # parentが親ボーンの場合
                                    trace_bf.position = MVector3D(joint) - MVector3D(parent_joint) \
                                                        - (target_trace_model.bones[mname].position - parent_bone.position)
                                    trace_bf.key = True
                                    target_trace_mov_motion.bones[mname][fno] = trace_bf
                                    pchar.update(1)

            for target_trace_model in [trace_model, mp_trace_model]:
                for bname, bone in target_trace_model.bones.items():
                    # 表示先の設定
                    if bone.tail_index and bone.tail_index in target_trace_model.bones:
                        bone.flag |= 0x0001
                        bone.tail_index = target_trace_model.bones[bone.tail_index].index
                    else:
                        bone.tail_index = -1

            logger.info("【No.{0}】トレース(センター)計算開始", f"{oidx:02d}", decoration=MLogger.DECORATION_LINE)
            
            with tqdm(total=(len(fnos) * 2), desc=f'No.{oidx:02d}') as pchar:
                target_links = {}
                for target_name in ["下半身", "上半身", "上半身2", "左足", "右足", "左かかと", "右かかと", "左つま先", "右つま先"]:
                    target_links[target_name] = trace_model.create_link_2_top_one(target_name, is_defined=False)
                
                depths = []
                min_leg_ys = []
                groove_ys = []
                for fno in fnos:
                    center_pos, min_leg_y, start_lower_depth = calc_center(trace_model, target_links, trace_mov_motion, target_bone_vecs["params"][fno], fno, start_lower_depth)
                    min_leg_ys.append(min_leg_y)
                    groove_ys.append(center_pos.y())
                    depths.append(center_pos.z())

                    groove_bf = trace_mov_motion.calc_bf("グルーブ", fno)
                    groove_bf.position.setY(center_pos.y())
                    groove_bf.key = True
                    trace_mov_motion.bones[groove_bf.name][fno] = groove_bf

                    center_bf = trace_mov_motion.calc_bf("センター", fno)
                    center_bf.position.setX(center_pos.x())
                    center_bf.position.setZ(center_pos.z())
                    center_bf.key = True
                    trace_mov_motion.bones[center_bf.name][fno] = center_bf
                    pchar.update(1)
                
                # 中央のY値が接地してると見なす
                all_min_leg_ys = np.array(min_leg_ys) + min(trace_model.bones["左つま先"].position.y(), 
                                    trace_model.bones["右つま先"].position.y(), trace_model.bones["左かかと"].position.y(), trace_model.bones["右かかと"].position.y())
                median_leg_y = np.median(all_min_leg_ys)

                for fno in fnos:
                    # グルーブを設定しなおす
                    groove_bf = trace_mov_motion.calc_bf("グルーブ", fno)
                    groove_bf.position.setY(groove_bf.position.y() - median_leg_y)
                    trace_mov_motion.bones[groove_bf.name][fno] = groove_bf
                    pchar.update(1)

            # expose
            trace_model_path = os.path.join(motion_dir_path, f"trace_{process_datetime}_mov_model_no{oidx:02d}.pmx")
            logger.info("【No.{0}】トレース(移動)モデル生成開始【{1}】", f"{oidx:02d}", os.path.basename(trace_model_path), decoration=MLogger.DECORATION_LINE)
            shutil.copy(args.trace_mov_model_config, trace_model_path)
            
            trace_mov_motion_path = os.path.join(motion_dir_path, f"trace_{process_datetime}_mov_no{oidx:02d}.vmd")
            logger.info("【No.{0}】モーション(移動)生成開始【{1}】", f"{oidx:02d}", os.path.basename(trace_mov_motion_path), decoration=MLogger.DECORATION_LINE)
            vmd_writer.write(trace_model, trace_mov_motion, trace_mov_motion_path)

            mp_trace_mov_motion_path = os.path.join(motion_dir_path, f"mp_trace_{process_datetime}_mov_no{oidx:02d}.vmd")
            logger.info("【No.{0}】mediapipeモーション(移動)生成開始【{1}】", f"{oidx:02d}", os.path.basename(mp_trace_mov_motion_path), decoration=MLogger.DECORATION_LINE)
            vmd_writer.write(mp_trace_model, mp_trace_mov_motion, mp_trace_mov_motion_path)

            # 回転トレース用モデルを読み込
            trace_model = PmxReader(args.trace_rot_model_config).read_data()
            mp_trace_model = PmxReader(args.trace_rot_model_config).read_data()

            # 足ＩＫ ------------------------
            ik_show = VmdShowIkFrame()
            ik_show.fno = 0
            ik_show.show = 1

            for direction in ["左", "右"]:
                create_bone_leg_ik(trace_model, direction)
                create_bone_leg_ik(mp_trace_model, direction)
                
                ik_show.ik.append(VmdInfoIk(f'{direction}足ＩＫ', 0))
                ik_show.ik.append(VmdInfoIk(f'{direction}つま先ＩＫ', 0))

            trace_rot_motion.showiks.append(ik_show)
            trace_miku_motion.showiks.append(ik_show)
            mp_trace_rot_motion.showiks.append(ik_show)
            mp_trace_miku_motion.showiks.append(ik_show)

            trace_model_path = os.path.join(motion_dir_path, f"trace_{process_datetime}_rot_model_no{oidx:02d}.pmx")
            logger.info("【No.{0}】トレース(回転)モデル生成開始【{1}】", f"{oidx:02d}", os.path.basename(trace_model_path), decoration=MLogger.DECORATION_LINE)
            shutil.copy(args.trace_rot_model_config, trace_model_path)
 
            logger.info("【No.{0}】モーション(回転)計算開始", f"{oidx:02d}", decoration=MLogger.DECORATION_LINE)
            
            with tqdm(total=((len(VMD_CONNECTIONS.keys()) + len(MP_VMD_CONNECTIONS.keys()) + 2) * (len(fnos))), desc=f'No.{oidx:02d}') as pchar:
                for mname in ["センター", "グルーブ"]:
                    for fno in fnos:
                        mov_bf = trace_mov_motion.calc_bf(mname, fno)
                        now_bf = trace_rot_motion.calc_bf(mname, fno)
                        now_bf.position = mov_bf.position
                        now_bf.key = True
                        trace_rot_motion.bones[mname][fno] = now_bf

                        pchar.update(1)

                for jconns, pconns, target_trace_model, target_trace_mov_motion, target_trace_rot_motion in [(VMD_CONNECTIONS, PMX_CONNECTIONS, trace_model, trace_mov_motion, trace_rot_motion), 
                                                                                                             (MP_VMD_CONNECTIONS, MP_PMX_CONNECTIONS, mp_trace_model, mp_trace_mov_motion, mp_trace_rot_motion)]:
                    for jname, jconn in jconns.items():
                        if not args.hand_motion and ("hand" in pconns[jname] or "_wrist" in jname):
                            # 手首から先はオプショナル
                            pchar.update(len(fnos))
                            continue

                        mname = pconns[jname]['mmd']
                        direction_from_mname = pconns[jconn["direction"][0]]['mmd']
                        direction_to_mname = pconns[jconn["direction"][1]]['mmd']
                        up_from_mname = pconns[jconn["up"][0]]['mmd']
                        up_to_mname = pconns[jconn["up"][1]]['mmd']
                        cross_from_mname = pconns[jconn["cross"][0]]['mmd'] if 'cross' in jconn else direction_from_mname
                        cross_to_mname = pconns[jconn["cross"][1]]['mmd'] if 'cross' in jconn else direction_to_mname

                        # トレースモデルの初期姿勢
                        target_trace_direction_from_vec = target_trace_model.bones[direction_from_mname].position
                        target_trace_direction_to_vec = target_trace_model.bones[direction_to_mname].position
                        target_trace_direction = (target_trace_direction_to_vec - target_trace_direction_from_vec).normalized()

                        target_trace_up_from_vec = target_trace_model.bones[up_from_mname].position
                        target_trace_up_to_vec = target_trace_model.bones[up_to_mname].position
                        target_trace_up = (target_trace_up_to_vec - target_trace_up_from_vec).normalized()

                        target_trace_cross_from_vec = target_trace_model.bones[cross_from_mname].position
                        target_trace_cross_to_vec = target_trace_model.bones[cross_to_mname].position
                        target_trace_cross = (target_trace_cross_to_vec - target_trace_cross_from_vec).normalized()
                        
                        target_trace_up_cross = MVector3D.crossProduct(target_trace_up, target_trace_cross).normalized()
                        target_trace_stance_qq = MQuaternion.fromDirection(target_trace_direction, target_trace_up_cross)

                        direction_from_links = target_trace_model.create_link_2_top_one(direction_from_mname, is_defined=False)
                        direction_to_links = target_trace_model.create_link_2_top_one(direction_to_mname, is_defined=False)
                        up_from_links = target_trace_model.create_link_2_top_one(up_from_mname, is_defined=False)
                        up_to_links = target_trace_model.create_link_2_top_one(up_to_mname, is_defined=False)
                        cross_from_links = target_trace_model.create_link_2_top_one(cross_from_mname, is_defined=False)
                        cross_to_links = target_trace_model.create_link_2_top_one(cross_to_mname, is_defined=False)

                        for fno in fnos:
                            now_direction_from_vec = calc_global_pos_from_mov(target_trace_model, direction_from_links, target_trace_mov_motion, fno)
                            now_direction_to_vec = calc_global_pos_from_mov(target_trace_model, direction_to_links, target_trace_mov_motion, fno)
                            now_up_from_vec = calc_global_pos_from_mov(target_trace_model, up_from_links, target_trace_mov_motion, fno)
                            now_up_to_vec = calc_global_pos_from_mov(target_trace_model, up_to_links, target_trace_mov_motion, fno)
                            now_cross_from_vec = calc_global_pos_from_mov(target_trace_model, cross_from_links, target_trace_mov_motion, fno)
                            now_cross_to_vec = calc_global_pos_from_mov(target_trace_model, cross_to_links, target_trace_mov_motion, fno)

                            # トレースモデルの回転量 ------------
                            now_direction = (now_direction_to_vec - now_direction_from_vec).normalized()
                            now_up = (now_up_to_vec - now_up_from_vec).normalized()
                            now_cross = (now_cross_to_vec - now_cross_from_vec).normalized()

                            now_up_cross = MVector3D.crossProduct(now_up, now_cross).normalized()
                            now_stance_qq = MQuaternion.fromDirection(now_direction, now_up_cross)

                            cancel_qq = MQuaternion()
                            for cancel_jname in jconn["cancel"]:
                                cancel_qq *= target_trace_rot_motion.calc_bf(pconns[cancel_jname]['mmd'], fno).rotation

                            now_qq = cancel_qq.inverted() * now_stance_qq * target_trace_stance_qq.inverted()

                            now_bf = target_trace_rot_motion.calc_bf(mname, fno)
                            now_bf.rotation = now_qq
                            now_bf.key = True
                            target_trace_rot_motion.bones[mname][fno] = now_bf

                            pchar.update(1)

            trace_rot_motion_path = os.path.join(motion_dir_path, f"trace_{process_datetime}_rot_no{oidx:02d}.vmd")
            logger.info("【No.{0}】トレースモーション(回転)生成開始【{1}】", f"{oidx:02d}", os.path.basename(trace_rot_motion_path), decoration=MLogger.DECORATION_LINE)
            vmd_writer.write(trace_model, trace_rot_motion, trace_rot_motion_path)

            mp_trace_rot_motion_path = os.path.join(motion_dir_path, f"mp_trace_{process_datetime}_rot_no{oidx:02d}.vmd")
            logger.info("【No.{0}】Mediapipe トレースモーション(回転)生成開始【{1}】", f"{oidx:02d}", os.path.basename(mp_trace_rot_motion_path), decoration=MLogger.DECORATION_LINE)
            vmd_writer.write(mp_trace_model, mp_trace_rot_motion, mp_trace_rot_motion_path)

            logger.info("【No.{0}】モーション(あにまさ式ミク)計算開始", f"{oidx:02d}", decoration=MLogger.DECORATION_LINE)

            # 移動ボーンのサイジング
            xz_ratio, y_ratio = calc_leg_ik_ratio(trace_model, miku_model)

            with tqdm(total=((len(trace_rot_motion.bones.keys()) + len(mp_trace_rot_motion.bones.keys()) + 34) * len(fnos)), desc=f'No.{oidx:02d}') as pchar:
                for mname in ["センター", "グルーブ"]:
                    for fno in fnos:
                        rot_bf = trace_rot_motion.calc_bf(mname, fno)
                        miku_bf = trace_miku_motion.calc_bf(mname, fno)
                        miku_bf.position = MVector3D(rot_bf.position.x() * xz_ratio, rot_bf.position.y() * y_ratio, rot_bf.position.z() * xz_ratio)
                        miku_bf.key = True

                        trace_miku_motion.bones[mname][fno] = miku_bf
                        pchar.update(1)
                        continue
                    
                for jconns, pconns, target_trace_model, target_trace_rot_motion, target_trace_miku_motion in [(VMD_CONNECTIONS, PMX_CONNECTIONS, trace_model, trace_rot_motion, trace_miku_motion), 
                                                                                                             (MP_VMD_CONNECTIONS, MP_PMX_CONNECTIONS, mp_trace_model, mp_trace_rot_motion, mp_trace_miku_motion)]:
                    for jname, jconn in jconns.items():
                        if not args.hand_motion and ("hand" in pconns[jname] or "_wrist" in jname):
                            # 手首から先はオプショナル
                            pchar.update(len(fnos))
                            continue

                        mname = pconns[jname]['mmd']
                        parent_jnmae = pconns[jname]['parent']

                        parent_name = "センター"
                        if mname not in ["全ての親", "センター", "グルーブ"] and parent_jnmae in pconns:
                            parent_name = pconns[parent_jnmae]['mmd']
                        
                        base_axis = pconns[jname]["axis"] if jname in pconns else None
                        parent_axis = pconns[jname]["parent_axis"] if jname in pconns and "parent_axis" in pconns[jname] else pconns[parent_jnmae]["axis"] if parent_jnmae in pconns else None
                        target_trace_parent_local_x_qq = target_trace_model.get_local_x_qq(parent_name, parent_axis)
                        target_trace_target_local_x_qq = target_trace_model.get_local_x_qq(mname, base_axis)
                        miku_parent_local_x_qq = miku_model.get_local_x_qq(parent_name, parent_axis)
                        miku_target_local_x_qq = miku_model.get_local_x_qq(mname, base_axis)

                        parent_local_x_qq = miku_parent_local_x_qq.inverted() * target_trace_parent_local_x_qq
                        target_local_x_qq = miku_target_local_x_qq.inverted() * target_trace_target_local_x_qq

                        miku_local_x_axis = miku_model.get_local_x_axis(mname)
                        miku_local_y_axis = MVector3D.crossProduct(miku_local_x_axis, MVector3D(0, 0, 1))

                        for fno in fnos:
                            rot_bf = target_trace_rot_motion.calc_bf(mname, fno)
                            miku_bf = target_trace_miku_motion.calc_bf(mname, fno)

                            miku_bf.position = rot_bf.position.copy()
                            new_miku_qq = rot_bf.rotation.copy()
                            
                            if mname[1:] in ["肩"]:
                                new_miku_qq = new_miku_qq * target_local_x_qq
                            else:
                                new_miku_qq = parent_local_x_qq.inverted() * new_miku_qq * target_local_x_qq

                            miku_bf.rotation = new_miku_qq
                            miku_bf.key = True

                            target_trace_miku_motion.bones[mname][fno] = miku_bf
                            pchar.update(1)
                    
                    for mname in ["左ひざ", "右ひざ"]:
                        miku_local_x_axis = miku_model.get_local_x_axis(mname)
                        miku_local_y_axis = MVector3D.crossProduct(miku_local_x_axis, MVector3D(0, 0, 1))

                        for fno in fnos:
                            miku_bf = target_trace_miku_motion.calc_bf(mname, fno)

                            # X捩り除去
                            _, _, _, now_yz_qq = MServiceUtils.separate_local_qq(fno, mname, miku_bf.rotation, miku_local_x_axis)
                            miku_bf.rotation = MQuaternion.fromAxisAndAngle(miku_local_y_axis, now_yz_qq.toDegree())
                            pchar.update(1)
                    
                    if args.hand_motion:
                        for mname in ['左親指０' , '左親指１' , '左親指２' , '左人指１' , '左人指２' , '左人指３' , '左中指１' , '左中指２' , '左中指３' , '左薬指１' , '左薬指２' , '左薬指３' , '左小指１' , '左小指２' , '左小指３' , \
                                    '右親指０' , '右親指１' , '右親指２' , '右人指１' , '右人指２' , '右人指３' , '右中指１' , '右中指２' , '右中指３' , '右薬指１' , '右薬指２' , '右薬指３' , '右小指１' , '右小指２' , '右小指３']:
                            miku_local_x_axis = miku_model.get_local_x_axis(mname)

                            for fno in fnos:
                                miku_bf = target_trace_miku_motion.calc_bf(mname, fno)

                                # 指は正方向にしか曲がらない
                                _, _, _, now_yz_qq = MServiceUtils.separate_local_qq(fno, mname, miku_bf.rotation, miku_local_x_axis)
                                miku_bf.rotation = MQuaternion.fromAxisAndAngle((MVector3D(0, 0, -1) if mname[0] == "左" else MVector3D(0, 0, 1)), now_yz_qq.toDegree())
                                pchar.update(1)
                
            if args.face_motion:
                # モーフ生成ONの場合
                logger.info("【No.{0}】モーフ計算開始", f"{oidx:02d}", decoration=MLogger.DECORATION_LINE)

                with tqdm(total=(len(face_fnos)), desc=f'No.{oidx:02d}') as pchar:
                    for fno in face_fnos:
                        # 口
                        calc_lip(fno, trace_miku_motion, target_bone_vecs)
                        # 眉
                        calc_eyebrow(fno, trace_miku_motion, target_bone_vecs)
                        # 目
                        calc_eye(fno, trace_miku_motion, target_bone_vecs)
                        pchar.update(1)

                logger.info("【No.{0}】モーフ間引き開始", f"{oidx:02d}", decoration=MLogger.DECORATION_LINE)

                with tqdm(total=(len(trace_miku_motion.morphs)), desc=f'No.{oidx:02d}') as pchar:
                    for morph_name in trace_miku_motion.morphs.keys():
                        trace_miku_motion.remove_unnecessary_mf(0, morph_name)
                        pchar.update(1)

            trace_miku_motion_path = os.path.join(motion_dir_path, f"trace_{process_datetime}_miku_no{oidx:02d}.vmd")
            logger.info("【No.{0}】トレースモーション(あにまさ式ミク)生成開始【{1}】", f"{oidx:02d}", os.path.basename(trace_miku_motion_path), decoration=MLogger.DECORATION_LINE)
            vmd_writer.write(miku_model, trace_miku_motion, trace_miku_motion_path)

            mp_trace_miku_motion_path = os.path.join(motion_dir_path, f"mp_trace_{process_datetime}_miku_no{oidx:02d}.vmd")
            logger.info("【No.{0}】Mediapipe トレースモーション(あにまさ式ミク)生成開始【{1}】", f"{oidx:02d}", os.path.basename(mp_trace_miku_motion_path), decoration=MLogger.DECORATION_LINE)
            vmd_writer.write(miku_model, mp_trace_miku_motion, mp_trace_miku_motion_path)

            logger.info("【No.{0}】モーション(あにまさ式ミク)外れ値削除開始", f"{oidx:02d}", decoration=MLogger.DECORATION_LINE)

            loop_cnt = 2
            with tqdm(total=((len(trace_miku_motion.bones.keys()) * len(fnos) * loop_cnt)), desc=f'No.{oidx:02d}') as pchar:
                # for mname in ["上半身", "下半身"]:
                #     local_x_axis = trace_model.get_local_x_axis(mname)

                #     for fno in mp_trace_miku_motion.bones[mname].keys():
                #         mp_bf = mp_trace_miku_motion.calc_bf(mname, fno)
                #         ex_bf = trace_miku_motion.calc_bf(mname, fno)

                #         if fno in target_bone_vecs["params"] and mname in target_bone_vecs["params"][fno]["visibility"] and target_bone_vecs["params"][fno]["visibility"][mname] > 0.7:
                #             # mediapipeの値が信頼性ある場合に判定対象とする
                #             dot = MQuaternion.dotProduct(mp_bf.rotation, ex_bf.rotation)
                #             logger.info("{0}: {1}, dot: {2}, vis: {3}", fno, mname, dot, target_bone_vecs["params"][fno]["visibility"][mname])
                #             if abs(dot) < 0.8:
                #                 # 離れすぎてるのは削除
                #                 if fno in trace_miku_motion.bones[mname]:
                #                     del trace_miku_motion.bones[mname][fno]
                #         pchar.update(1)

                for n in range(loop_cnt):
                    bone_names = list(trace_miku_motion.bones.keys())
                    for mname in bone_names:
                        if mname in ["センター", "グルーブ"]:
                            for prev_fno, next_fno in zip(fnos[n:], fnos[3+n:]):
                                prev_bf = trace_miku_motion.calc_bf(mname, prev_fno)
                                next_bf = trace_miku_motion.calc_bf(mname, next_fno)

                                prev_next_x_distance = abs(prev_bf.position.x() - next_bf.position.x())
                                prev_next_y_distance = abs(prev_bf.position.y() - next_bf.position.y())
                                prev_next_z_distance = abs(prev_bf.position.z() - next_bf.position.z())

                                for fno in range(prev_fno + 1, next_fno):
                                    now_bf = trace_miku_motion.calc_bf(mname, fno)

                                    prev_now_x_distance = abs(prev_bf.position.x() - now_bf.position.x())
                                    prev_now_y_distance = abs(prev_bf.position.y() - now_bf.position.y())
                                    prev_now_z_distance = abs(prev_bf.position.z() - now_bf.position.z())

                                    fill_pos = now_bf.position.copy()

                                    if not (prev_next_x_distance - 0.2 < prev_now_x_distance < prev_next_x_distance + 0.2):
                                        # 離れすぎてるのは削除(線形補間で強制上書き)
                                        fill_pos.setX(prev_bf.position.x() + ((next_bf.position.x() - prev_bf.position.x()) * ((fno - prev_fno) / (next_fno - prev_fno))))

                                    if not (prev_next_y_distance - 0.2 < prev_now_y_distance < prev_next_y_distance + 0.2):
                                        # 離れすぎてるのは削除(線形補間で強制上書き)
                                        fill_pos.setY(prev_bf.position.y() + ((next_bf.position.y() - prev_bf.position.y()) * ((fno - prev_fno) / (next_fno - prev_fno))))

                                    if not (prev_next_z_distance - 0.2 < prev_now_z_distance < prev_next_z_distance + 0.2):
                                        # 離れすぎてるのは削除(線形補間で強制上書き)
                                        fill_pos.setZ(prev_bf.position.z() + ((next_bf.position.z() - prev_bf.position.z()) * ((fno - prev_fno) / (next_fno - prev_fno))))

                                    now_bf.position = fill_pos

                                pchar.update(1)
                            continue
                        
                        for prev_fno, next_fno in zip(fnos[n:], fnos[3+n:]):
                            prev_bf = trace_miku_motion.calc_bf(mname, prev_fno)
                            next_bf = trace_miku_motion.calc_bf(mname, next_fno)

                            prev_next_dot = MQuaternion.dotProduct(prev_bf.rotation, next_bf.rotation)

                            for fno in range(prev_fno + 1, next_fno):
                                now_bf = trace_miku_motion.calc_bf(mname, fno)

                                # まず前後の中間をそのまま求める
                                filterd_qq = MQuaternion.slerp(prev_bf.rotation, next_bf.rotation, (fno - prev_fno) / (next_fno - prev_fno))
                                # 前後の中間と実際の値の内積を求める
                                dot = MQuaternion.dotProduct(filterd_qq, now_bf.rotation)
                                if prev_next_dot - 0.3 > dot:
                                    # 離れすぎてるのは削除
                                    if fno in trace_miku_motion.bones[mname]:
                                        del trace_miku_motion.bones[mname][fno]
                            pchar.update(1)

            logger.info("【No.{0}】モーション(あにまさ式ミク)スムージング開始", f"{oidx:02d}", decoration=MLogger.DECORATION_LINE)

            loop_cnt = 3
            with tqdm(total=(len(trace_miku_motion.bones.keys()) * len(fnos) * loop_cnt), desc=f'No.{oidx:02d}') as pchar:
                for n in range(loop_cnt):
                    bone_names = list(trace_miku_motion.bones.keys())
                    for mname in bone_names:
                        if mname in ["センター", "グルーブ"]:
                            mxfilter = OneEuroFilter(freq=30, mincutoff=1, beta=0.1, dcutoff=1)
                            myfilter = OneEuroFilter(freq=30, mincutoff=1, beta=0.1, dcutoff=1)
                            mzfilter = OneEuroFilter(freq=30, mincutoff=1, beta=0.1, dcutoff=1)
                            for fidx, fno in enumerate(fnos):
                                bf = trace_miku_motion.calc_bf(mname, fno)
                                smooth_pos = MVector3D()
                                smooth_pos.setX(mxfilter(bf.position.x(), fno))
                                smooth_pos.setY(myfilter(bf.position.y(), fno))
                                smooth_pos.setZ(mzfilter(bf.position.z(), fno))

                                bf.position = smooth_pos
                                bf.key = True
                                trace_miku_motion.bones[mname][fno] = bf
                                pchar.update(1)
                            continue

                        for prev_fno, next_fno in zip(fnos[n:], fnos[3+n:]):
                            prev_bf = trace_miku_motion.calc_bf(mname, prev_fno)
                            next_bf = trace_miku_motion.calc_bf(mname, next_fno)
                            for fno in range(prev_fno + 1, next_fno):
                                bf = trace_miku_motion.calc_bf(mname, fno)
                                if bf.key:
                                    # まず前後の中間をそのまま求める
                                    filterd_qq = MQuaternion.slerp(prev_bf.rotation, next_bf.rotation, (fno - prev_fno) / (next_fno - prev_fno))
                                    # 現在の回転にも少し近づける
                                    bf.rotation = MQuaternion.slerp(filterd_qq, bf.rotation, 0.8)
                                    bf.key = True
                                    trace_miku_motion.bones[mname][fno] = bf

                            pchar.update(1)

            trace_miku_motion_path = os.path.join(motion_dir_path, f"trace_{process_datetime}_miku_smooth_no{oidx:02d}.vmd")
            logger.info("【No.{0}】スムージングトレースモーション(あにまさ式ミク)生成開始【{1}】", f"{oidx:02d}", os.path.basename(trace_miku_motion_path), decoration=MLogger.DECORATION_LINE)
            vmd_writer.write(miku_model, trace_miku_motion, trace_miku_motion_path)

        logger.info('モーション生成処理全件終了', decoration=MLogger.DECORATION_BOX)

        return True
    except Exception as e:
        logger.critical("モーション生成で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        return False

def normalize(v, axis=-1, order=2):
    l2 = np.linalg.norm(v, ord = order, axis=axis, keepdims=True)
    l2[l2==0] = 1
    return v/l2

def calc_leg_ik_ratio(trace_model: PmxModel, miku_model: PmxModel):
    target_bones = ["左足", "左ひざ", "左足首", "センター"]

    if set(target_bones).issubset(trace_model.bones) and set(target_bones).issubset(miku_model.bones):
        # XZ比率(足の長さ)
        org_leg_length = ((trace_model.bones["左足首"].position - trace_model.bones["左ひざ"].position) \
                          + (trace_model.bones["左ひざ"].position - trace_model.bones["左足"].position)).length()
        rep_leg_length = ((miku_model.bones["左足首"].position - miku_model.bones["左ひざ"].position) \
                          + (miku_model.bones["左ひざ"].position - miku_model.bones["左足"].position)).length()
        logger.test("xz_ratio rep_leg_length: %s, org_leg_length: %s", rep_leg_length, org_leg_length)
        xz_ratio = 1 if org_leg_length == 0 else (rep_leg_length / org_leg_length)

        # Y比率(股下のY差)
        rep_leg_length = (miku_model.bones["左足首"].position - miku_model.bones["左足"].position).y()
        org_leg_length = (trace_model.bones["左足首"].position - trace_model.bones["左足"].position).y()
        logger.test("y_ratio rep_leg_length: %s, org_leg_length: %s", rep_leg_length, org_leg_length)
        y_ratio = 1 if org_leg_length == 0 else (rep_leg_length / org_leg_length)

        return xz_ratio, y_ratio
    
    return 1, 1

def create_bone(trace_model: PmxModel, jname: str, jconn: dict, target_bone_vecs: dict, miku_model: PmxModel, conns: dict):
    parent_jname = jconn["parent"]
    # MMDボーン名
    mname = jconn["mmd"]
    # 親ボーン
    parent_bone = trace_model.bones[conns[parent_jname]['mmd']] if parent_jname in conns else trace_model.bones[parent_jname] if parent_jname in trace_model.bones else None
    if not parent_bone:
        for b in trace_model.bones.values():
            if b.english_name == parent_jname:
                parent_bone = b
                break

    if jname not in target_bone_vecs or parent_jname not in target_bone_vecs:
        bone_relative_pos = MVector3D()
    else:
        joints = list(target_bone_vecs[jname].values())
        parent_joints = list(target_bone_vecs[parent_jname].values())

        if jname in ["mp_left_wrist", "mp_right_wrist"]:
            # 手首2は手首と合わせる
            bone_relative_pos = MVector3D()
        elif "指" in jconn["display"]:
            if "手首2" in parent_bone.name:
                # 手首2を参照している場合、手首を参照に切替
                parent_bone = trace_model.bones[parent_bone.name[:3]]
            # 手首と指は完全にミクに合わせる
            bone_relative_pos = miku_model.bones[mname].position - miku_model.bones[parent_bone.name].position
        else:
            bone_length = np.median(np.linalg.norm(np.array(joints) - np.array(parent_joints), ord=2, axis=1))

            if mname[1:] in ["手首", "ひじ", "手先", "ひざ", "足首", "つま先"] and mname in miku_model.bones and parent_bone.name in miku_model.bones:
                # 腕・足は方向はミクに合わせる。長さはトレース元
                bone_relative_pos = miku_model.bones[mname].position - miku_model.bones[parent_bone.name].position
                bone_relative_pos *= bone_length / bone_relative_pos.length()
            else:
                # トレース元から採取
                bone_axis = MVector3D(np.median(np.array(joints), axis=0) - np.median(np.array(parent_joints), axis=0)).normalized()
                bone_relative_pos = MVector3D(bone_axis * bone_length)
    bone_pos = parent_bone.position + bone_relative_pos
    bone = Bone(mname, mname, bone_pos, parent_bone.index, 0, 0x0000 | 0x0002 | 0x0004 | 0x0008 | 0x0010)
    bone.index = len(list(trace_model.bones.keys()))
    if len(jconn['tail']) > 0:
        bone.tail_index = conns[jconn["tail"]]['mmd']

    # ボーンINDEX
    trace_model.bone_indexes[bone.index] = bone.name
    # ボーン
    trace_model.bones[bone.name] = bone
    # 表示枠
    trace_model.display_slots[jconn["display"]].references.append(trace_model.bones[bone.name].index)
    if jconn["parent"] not in ["全ての親", "センター", "グルーブ"]:
        # 材質
        trace_model.materials[bone.name] = Material(bone.name, bone.name, MVector3D(0, 0, 1), 1, 0, MVector3D(), MVector3D(0.5, 0.5, 0.5), 0x02 | 0x08, MVector4D(0, 0, 0, 1), 1, 0, 0, 0, 0, 0)
        trace_model.materials[bone.name].vertex_count = 12 * 3
        start_vidx = len(trace_model.vertices["vertices"])

        for vidx, (b, v) in enumerate([(bone, bone.position + MVector3D(-0.05, 0, -0.05)), (bone, bone.position + MVector3D(-0.05, 0, 0.05)), \
                                       (parent_bone, parent_bone.position + MVector3D(-0.05, 0, 0.05)), (parent_bone, parent_bone.position + MVector3D(-0.05, 0, -0.05)), \
                                       (parent_bone, parent_bone.position + MVector3D(-0.05, 0, -0.05)), (parent_bone, parent_bone.position + MVector3D(-0.05, 0, 0.05)), \
                                       (parent_bone, parent_bone.position + MVector3D(0.05, 0, 0.05)), (parent_bone, parent_bone.position + MVector3D(0.05, 0, -0.05)), \
                                       (parent_bone, parent_bone.position + MVector3D(0.05, 0, -0.05)), (parent_bone, parent_bone.position + MVector3D(0.05, 0, 0.05)), \
                                       (bone, bone.position + MVector3D(0.05, 0, 0.05)), (bone, bone.position + MVector3D(0.05, 0, -0.05)), \
                                       (bone, bone.position + MVector3D(-0.05, 0, 0.05)), (bone, bone.position + MVector3D(-0.05, 0, -0.05)), \
                                       (bone, bone.position + MVector3D(0.05, 0, -0.05)), (bone, bone.position + MVector3D(0.05, 0, 0.05)), \
                                       (bone, bone.position + MVector3D(-0.05, 0, 0.05)), (bone, bone.position + MVector3D(0.05, 0, 0.05)), \
                                       (parent_bone, parent_bone.position + MVector3D(0.05, 0, 0.05)), (parent_bone, parent_bone.position + MVector3D(-0.05, 0, 0.05)), \
                                       (bone, bone.position + MVector3D(-0.05, 0, -0.05)), (parent_bone, parent_bone.position + MVector3D(-0.05, 0, -0.05)), \
                                       (parent_bone, parent_bone.position + MVector3D(0.05, 0, -0.05)), (bone, bone.position + MVector3D(0.05, 0, -0.05)), \
                                       ]):
            v1 = Vertex(start_vidx + vidx, v, MVector3D(0, 0, -1), MVector3D(), [], Bdef1(b.index), 1)
            trace_model.vertices["vertices"].append(v1)
        # 面1（上下辺）
        trace_model.indices.append(start_vidx + 0)
        trace_model.indices.append(start_vidx + 1)
        trace_model.indices.append(start_vidx + 2)
        # 面2（上下辺）
        trace_model.indices.append(start_vidx + 2)
        trace_model.indices.append(start_vidx + 3)
        trace_model.indices.append(start_vidx + 0)
        # 面3(縦前)
        trace_model.indices.append(start_vidx + 4)
        trace_model.indices.append(start_vidx + 5)
        trace_model.indices.append(start_vidx + 6)
        # 面4(縦前)
        trace_model.indices.append(start_vidx + 6)
        trace_model.indices.append(start_vidx + 7)
        trace_model.indices.append(start_vidx + 4)
        # 面3(縦左)
        trace_model.indices.append(start_vidx + 8)
        trace_model.indices.append(start_vidx + 9)
        trace_model.indices.append(start_vidx + 10)
        # 面4(縦左)
        trace_model.indices.append(start_vidx + 10)
        trace_model.indices.append(start_vidx + 11)
        trace_model.indices.append(start_vidx + 8)
        # 面5(縦右)
        trace_model.indices.append(start_vidx + 12)
        trace_model.indices.append(start_vidx + 13)
        trace_model.indices.append(start_vidx + 14)
        # 面6(縦右)
        trace_model.indices.append(start_vidx + 14)
        trace_model.indices.append(start_vidx + 15)
        trace_model.indices.append(start_vidx + 12)
        # 面7(縦後)
        trace_model.indices.append(start_vidx + 16)
        trace_model.indices.append(start_vidx + 17)
        trace_model.indices.append(start_vidx + 18)
        # 面8(縦後)
        trace_model.indices.append(start_vidx + 18)
        trace_model.indices.append(start_vidx + 19)
        trace_model.indices.append(start_vidx + 16)
        # 面9(縦後)
        trace_model.indices.append(start_vidx + 20)
        trace_model.indices.append(start_vidx + 21)
        trace_model.indices.append(start_vidx + 22)
        # 面10(縦後)
        trace_model.indices.append(start_vidx + 22)
        trace_model.indices.append(start_vidx + 23)
        trace_model.indices.append(start_vidx + 20)

# 平滑化
def smooth_values(delimiter: int, values: list):
    smooth_vs = []
    data = np.array(values)

    # 前後のフレームで平均を取る
    if len(data) > delimiter:
        move_avg = np.convolve(data, np.ones(delimiter)/delimiter, 'valid')
        # 移動平均でデータ数が減るため、前と後ろに同じ値を繰り返しで補填する
        fore_n = int((delimiter - 1)/2)
        back_n = delimiter - 1 - fore_n
        smooth_vs = np.hstack((np.tile([move_avg[0]], fore_n), move_avg, np.tile([move_avg[-1]], back_n)))
    else:
        avg = np.mean(data)
        smooth_vs = np.tile([avg], len(data))

    smooth_vs *= delimiter / 9

    return smooth_vs

def calc_global_pos_from_mov(model: PmxModel, links: BoneLinks, motion: VmdMotion, fno: int):
    trans_vs = MServiceUtils.calc_relative_position(model, links, motion, fno)
    
    result_pos = MVector3D()
    for v in trans_vs:
        result_pos += v

    return result_pos

def calc_ratio(ratio: float, oldmin: float, oldmax: float, newmin: float, newmax: float):
    # https://qastack.jp/programming/929103/convert-a-number-range-to-another-range-maintaining-ratio
    # NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
    return (((ratio - oldmin) * (newmax - newmin)) / (oldmax - oldmin)) + newmin

def calc_center(trace_model: PmxModel, target_all_links: dict, trace_mov_motion: VmdMotion, params: dict, fno: int, start_lower_depth: None):
    # 画像サイズ
    image_size = np.array([params["image"]["width"], params["image"]["height"]])
    # 画像サイズの最大公約数
    image_gcd = np.gcd.reduce(image_size)
    # 画像サイズの比率
    image_ratio = (int(image_size[0] / image_gcd), int(image_size[1] / image_gcd))

    # カメラ中央
    camera_center_pos = MVector3D(0, 0, 0.001)
    # カメラ倍率
    camera_scale = params["camera"]["scale"]
    # センサー幅（画角？）
    sensor_width = params["others"]["sensor_width"]
    # フォーカスのmm単位
    focal_length_in_mm = params["others"]["focal_length_in_mm"]
    # カメラ深度
    depth = params["depth"]["depth"]

    zs = []
    leg_ys = []
    target_global_poses = {}
    for target_name, target_links in target_all_links.items():
        # 四肢のグローバル位置を求める
        target_global_poses[target_name] = calc_global_pos_from_mov(trace_model, target_links, trace_mov_motion, fno)
        if target_name in params["visibility"] and params["visibility"][target_name] > 0.7:
            # Zはある程度信頼度があるものだけ対象とする
            zs.append(target_global_poses[target_name].z())
        if target_name in ["左つま先", "右つま先", "左かかと", "右かかと"]:
            # 足系のY
            leg_ys.append(target_global_poses[target_name].y())
        
    center_pos = MVector3D()

    # Xは下半身の画面内位置
    center_pos.setX((params["proj_pelvis"]["x"] - (image_size[0] / 2)) * depth * 0.002)

    # 上半身の傾き
    upper_dot = MVector3D.dotProduct((trace_model.bones["上半身2"].position - trace_model.bones["上半身"].position).normalized(), \
                                     (target_global_poses["上半身2"] - target_global_poses["上半身"]).normalized())
    upper_direction = calc_ratio(upper_dot, -1, 1, 0.8, 1)

    # 下半身の向き
    lower_dot = MVector3D.dotProduct(MVector3D(1, 0, 0), (target_global_poses["左足"] - target_global_poses["右足"]).normalized())
    lower_direction = max(upper_direction, calc_ratio(abs(lower_dot), 0, 1, 0.8, 1))

    # 最も下にある足のY
    min_leg_y = min(leg_ys)

    # 下半身の奥行き
    # 上半身が傾いている場合は浅めに見積もる(ただし、下半身が横を向いている時は相殺する)
    lower_depth = depth * (upper_direction / lower_direction) * 3

    # Yはプロジェクション座標の相対位置から仮算出
    upper_project_square_pos = calc_project_square_vec(camera_center_pos, focal_length_in_mm, sensor_width, image_ratio, target_global_poses["上半身"])
    proj_pelvis_square_pos = np.array([params["proj_pelvis"]["x"], params["proj_pelvis"]["y"]]) / image_size

    center_pos.setY(((1 - proj_pelvis_square_pos[1]) - (1 - upper_project_square_pos.y())) * lower_depth)

    if not start_lower_depth:
        return center_pos, min_leg_y, lower_depth

    center_pos.setZ(lower_depth - start_lower_depth)

    return center_pos, min_leg_y, start_lower_depth

# プロジェクション座標正規位置算出
def calc_project_square_vec(camera_pos: MVector3D, camera_length: float, camera_angle: float, image_ratio: tuple, global_vec: MVector3D):
    # モデル座標系
    model_view = create_model_view(camera_pos, camera_length)

    # プロジェクション座標系
    projection_view = create_projection_view(camera_angle, image_ratio)

    # viewport
    viewport_rect = MRect(0, 0, image_ratio[0], image_ratio[1])

    # プロジェクション座標位置
    project_vec = global_vec.project(model_view, projection_view, viewport_rect)

    # プロジェクション座標正規位置
    project_square_vec = MVector3D()
    project_square_vec.setX(project_vec.x() / image_ratio[0])

    if camera_length <= 0:
        project_square_vec.setY((-project_vec.y() + image_ratio[1]) / image_ratio[1])
    else:
        project_square_vec.setY(project_vec.y() / image_ratio[1])
    
    project_square_vec.setZ(project_vec.z())

    return project_square_vec

# プロジェクション座標系作成
def create_projection_view(camera_angle: float, image_ratio: tuple):
    mat = MMatrix4x4()
    mat.setToIdentity()
    # MMDの縦の視野角。
    # https://ch.nicovideo.jp/t-ebiing/blomaga/ar510610
    mat.perspective(camera_angle * 0.98, image_ratio[0] / image_ratio[1], 0.001, 50000)

    return mat

# モデル座標系作成
def create_model_view(camera_pos: MVector3D, camera_length: float):
    # モデル座標系（原点を見るため、単位行列）
    model_view = MMatrix4x4()
    model_view.setToIdentity()

    # カメラ角度
    camera_qq = MQuaternion()

    # カメラの原点（グローバル座標）
    mat_origin = MMatrix4x4()
    mat_origin.setToIdentity()
    mat_origin.translate(camera_pos)
    mat_origin.rotate(camera_qq)
    mat_origin.translate(MVector3D(0, 0, camera_length))
    camera_origin = mat_origin * MVector3D()

    mat_up = MMatrix4x4()
    mat_up.setToIdentity()
    mat_up.rotate(camera_qq)
    camera_up = mat_up * MVector3D(0, 1, 0)

    # カメラ座標系の行列
    # eye: カメラの原点（グローバル座標）
    # center: カメラの注視点（グローバル座標）
    # up: カメラの上方向ベクトル
    model_view.lookAt(camera_origin, camera_pos, camera_up)

    return model_view

def init_trace_model():
    trace_model = PmxModel()
    trace_model.vertices["vertices"] = []
    # 空テクスチャを登録
    trace_model.textures.append("")

    # 全ての親 ------------------------
    trace_model.bones["全ての親"] = Bone("全ての親", "Root", MVector3D(), -1, 0, 0x0000 | 0x0002 | 0x0004 | 0x0008 | 0x0010)
    trace_model.bones["全ての親"].index = 0
    trace_model.bone_indexes[0] = "全ての親"
    trace_model.display_slots["Root"] = DisplaySlot("Root", "Root", 1, 0)
    trace_model.display_slots["Root"].references.append(trace_model.bones["全ての親"].index)

    # モーフの表示枠
    trace_model.display_slots["表情"] = DisplaySlot("表情", "Exp", 1, 1)

    # センター ------------------------
    trace_model.bones["センター"] = Bone("センター", "Center", MVector3D(0, 9, 0), trace_model.bones["全ての親"].index, 0, 0x0000 | 0x0002 | 0x0004 | 0x0008 | 0x0010)
    trace_model.bones["センター"].index = len(trace_model.bones) - 1
    trace_model.bone_indexes[trace_model.bones["センター"].index] = "センター"
    trace_model.display_slots["センター"] = DisplaySlot("センター", "Center", 0, 0)
    trace_model.display_slots["センター"].references.append(trace_model.bones["センター"].index)

    # グルーブ ------------------------
    trace_model.bones["グルーブ"] = Bone("グルーブ", "Groove", MVector3D(0, 9.5, 0), trace_model.bones["センター"].index, 0, 0x0000 | 0x0002 | 0x0004 | 0x0008 | 0x0010)
    trace_model.bones["グルーブ"].index = len(trace_model.bones) - 1
    trace_model.bone_indexes[trace_model.bones["グルーブ"].index] = "グルーブ"
    trace_model.display_slots["センター"].references.append(trace_model.bones["グルーブ"].index)

    # その他
    for display_name in ["体幹", "左足", "右足", "左手", "右手", "左指", "右指", "顔", "眉", "鼻", "目", "口", "輪郭"]:
        trace_model.display_slots[display_name] = DisplaySlot(display_name, display_name, 0, 0)

    return trace_model

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
                            int(row[14]) * 0x0001| int(row[8]) * 0x0002| int(row[9]) * 0x0004 | int(row[10]) * 0x0020 | int(row[11]) * 0x0008 | int(row[12]) * 0x0010)
                bone.index = ridx - 1

                if len(row[15]) > 0:
                    # 表示先が指定されている場合、設定
                    bone.tail_index = row[15]

                if len(row[37]) > 0:
                    # IKターゲットがある場合、IK登録
                    bone.ik = Ik(model.bones[row[37]].index, int(row[38]), math.radians(float(row[39])))

                model.bones[row[1]] = bone
                model.bone_indexes[bone.index] = row[1]
            elif row[0] == "IKLink":
                iklink = IkLink(model.bones[row[2]].index, int(row[3]), MVector3D(float(row[4]), float(row[6]), float(row[8])), MVector3D(float(row[5]), float(row[7]), float(row[9])))
                model.bones[row[1]].ik.link.append(iklink)
    
    for bidx, bone in model.bones.items():
        # 親ボーンINDEXの設定
        if bone.parent_index and bone.parent_index in model.bones:
            bone.parent_index = model.bones[bone.parent_index].index
        else:
            bone.parent_index = -1
        
        if bone.tail_index and bone.tail_index in model.bones:
            bone.tail_index = model.bones[bone.tail_index].index
        else:
            bone.tail_index = -1

    # 首根元ボーン
    if "左腕" in model.bones and "右腕" in model.bones:
        neck_base_vertex = Vertex(-1, (model.bones["左腕"].position + model.bones["右腕"].position) / 2 + MVector3D(0, -0.05, 0), MVector3D(), [], [], Bdef1(-1), -1)
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

    # 指根元ボーン
    if "左手首" in model.bones:
        left_finger_base_bone = Bone("左指根元", "", (model.bones["左親指０"].position + model.bones["左人指１"].position + model.bones["左中指１"].position + model.bones["左薬指１"].position + model.bones["左小指１"].position) / 5, -1, 0, 0)
        left_finger_base_bone.parent_index = model.bones["首"].index
        left_finger_base_bone.tail_index = model.bones["頭"].index
        left_finger_base_bone.index = len(model.bones.keys())
        model.bones[left_finger_base_bone.name] = left_finger_base_bone
        model.bone_indexes[left_finger_base_bone.index] = left_finger_base_bone.name

    if "右手首" in model.bones:
        right_finger_base_bone = Bone("右指根元", "", (model.bones["右親指０"].position + model.bones["右人指１"].position + model.bones["右中指１"].position + model.bones["右薬指１"].position + model.bones["右小指１"].position) / 5, -1, 0, 0)
        right_finger_base_bone.parent_index = model.bones["首"].index
        right_finger_base_bone.tail_index = model.bones["頭"].index
        right_finger_base_bone.index = len(model.bones.keys())
        model.bones[right_finger_base_bone.name] = right_finger_base_bone
        model.bone_indexes[right_finger_base_bone.index] = right_finger_base_bone.name

    # 頭頂ボーン
    if "頭" in model.bones:
        head_top_bone = Bone("頭頂", "top of head", MVector3D(0, model.bones["頭"].position.y() + 1, 0), -1, 0, 0)
        head_top_bone.parent_index = model.bones["鼻"].index
        head_top_bone.index = len(model.bones.keys())
        model.bones[head_top_bone.name] = head_top_bone
        model.bone_indexes[head_top_bone.index] = head_top_bone.name

    # 下半身先ボーン
    if "左足" in model.bones and "右足" in model.bones:
        lower_tail_vertex = Vertex(-1, (model.bones["左足"].position + model.bones["右足"].position) / 2, MVector3D(), [], [], Bdef1(-1), -1)
        lower_tail_vertex.position.setX(0)
        lower_tail_bone = Bone("下半身先", "lower tail", lower_tail_vertex.position.copy(), -1, 0, 0)

        lower_tail_bone.parent_index = model.bones["下半身"].index

        lower_tail_bone.index = len(model.bones.keys())
        model.bones[lower_tail_bone.name] = lower_tail_bone
        model.bone_indexes[lower_tail_bone.index] = lower_tail_bone.name

    if "右足首" in model.bones:
        right_heel_bone = Bone("右かかと", "", MVector3D(model.bones["右足首"].position.x(), 0, model.bones["右足首"].position.z()), -1, 0, 0)
        right_heel_bone.parent_index = model.bones["右つま先"].index
        right_heel_bone.index = len(model.bones.keys())
        model.bones[right_heel_bone.name] = right_heel_bone
        model.bone_indexes[right_heel_bone.index] = right_heel_bone.name

    if "左足首" in model.bones:
        left_heel_bone = Bone("左かかと", "", MVector3D(model.bones["左足首"].position.x(), 0, model.bones["左足首"].position.z()), -1, 0, 0)
        left_heel_bone.parent_index = model.bones["左つま先"].index
        left_heel_bone.index = len(model.bones.keys())
        model.bones[left_heel_bone.name] = left_heel_bone
        model.bone_indexes[left_heel_bone.index] = left_heel_bone.name

    if "右つま先" in model.bones:
        right_big_toe_bone = Bone("右足親指", "", model.bones["右つま先"].position + MVector3D(0.5, 0, 0), -1, 0, 0)
        right_big_toe_bone.parent_index = model.bones["右つま先"].index
        right_big_toe_bone.index = len(model.bones.keys())
        model.bones[right_big_toe_bone.name] = right_big_toe_bone
        model.bone_indexes[right_big_toe_bone.index] = right_big_toe_bone.name

        right_small_toe_bone = Bone("右足小指", "", model.bones["右つま先"].position + MVector3D(-0.5, 0, 0), -1, 0, 0)
        right_small_toe_bone.parent_index = model.bones["右つま先"].index
        right_small_toe_bone.index = len(model.bones.keys())
        model.bones[right_small_toe_bone.name] = right_small_toe_bone
        model.bone_indexes[right_small_toe_bone.index] = right_small_toe_bone.name

    if "左つま先" in model.bones:
        left_big_toe_bone = Bone("左足親指", "", model.bones["左つま先"].position + MVector3D(-0.5, 0, 0), -1, 0, 0)
        left_big_toe_bone.parent_index = model.bones["左つま先"].index
        left_big_toe_bone.index = len(model.bones.keys())
        model.bones[left_big_toe_bone.name] = left_big_toe_bone
        model.bone_indexes[left_big_toe_bone.index] = left_big_toe_bone.name

        left_small_toe_bone = Bone("左足小指", "", model.bones["左つま先"].position + MVector3D(0.5, 0, 0), -1, 0, 0)
        left_small_toe_bone.parent_index = model.bones["左つま先"].index
        left_small_toe_bone.index = len(model.bones.keys())
        model.bones[left_small_toe_bone.name] = left_small_toe_bone
        model.bone_indexes[left_small_toe_bone.index] = left_small_toe_bone.name

    return model

# リップモーフ
def calc_lip(fno: int, motion: VmdMotion, target_bone_vecs: dict):
    left_nose_2 = target_bone_vecs['left_nose_2'][fno]
    right_nose_2 = target_bone_vecs['right_nose_2'][fno]

    right_lip_top_1 = target_bone_vecs['right_lip_top_1'][fno]
    right_lip_top_5 = target_bone_vecs['right_lip_top_5'][fno]
    lip_top = target_bone_vecs['lip_top'][fno]
    left_lip_top_1 = target_bone_vecs['left_lip_top_1'][fno]
    left_lip_top_5 = target_bone_vecs['left_lip_top_5'][fno]

    right_lip_bottom_5 = target_bone_vecs['right_lip_bottom_5'][fno]
    lip_bottom = target_bone_vecs['lip_bottom'][fno]
    left_lip_bottom_5 = target_bone_vecs['left_lip_bottom_5'][fno]

    right_mouth_top_5 = target_bone_vecs['right_mouth_top_5'][fno]
    mouth_top = target_bone_vecs['mouth_top'][fno]
    left_mouth_top_5 = target_bone_vecs['left_mouth_top_5'][fno]

    right_mouth_bottom_5 = target_bone_vecs['right_mouth_bottom_5'][fno]
    mouth_bottom = target_bone_vecs['mouth_bottom'][fno]
    left_mouth_bottom_5 = target_bone_vecs['left_mouth_bottom_5'][fno]

    # 鼻の幅
    nose_width = abs(left_nose_2.x() - right_nose_2.x())

    # 口角の平均値
    corner_center = (left_lip_top_1 + right_lip_top_1) / 2
    # 口角の幅
    mouse_width = abs(left_lip_top_1.x() - right_lip_top_1.x())

    # 鼻基準の口の横幅比率
    mouse_width_ratio = mouse_width / nose_width

    # 上唇の平均値
    top_lip_center = (right_lip_top_5 + lip_top + left_lip_top_5) / 3
    top_mouth_center = (right_mouth_top_5 + mouth_top + left_mouth_top_5) / 3

    # 下唇の平均値
    bottom_lip_center = (right_lip_bottom_5 + lip_bottom + left_lip_bottom_5) / 3
    bottom_mouth_center = (right_mouth_bottom_5 + mouth_bottom + left_mouth_bottom_5) / 3

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

# 眉モーフ
def calc_eyebrow(fno: int, motion: VmdMotion, target_bone_vecs: dict):
    right_eye_brow1 = target_bone_vecs['right_eye_brow1'][fno]
    right_eye_brow4 = target_bone_vecs['right_eye_brow4'][fno]
    right_eye_brow6 = target_bone_vecs['right_eye_brow6'][fno]
    left_eye_brow1 = target_bone_vecs['left_eye_brow1'][fno]
    left_eye_brow4 = target_bone_vecs['left_eye_brow4'][fno]
    left_eye_brow6 = target_bone_vecs['left_eye_brow6'][fno]

    left_nose_2 = target_bone_vecs['left_nose_2'][fno]
    right_nose_2 = target_bone_vecs['right_nose_2'][fno]

    right_eye_top1 = target_bone_vecs['right_eye_top1'][fno]
    right_eye_top5 = target_bone_vecs['right_eye_top5'][fno]
    right_eye_top9 = target_bone_vecs['right_eye_top9'][fno]
    left_eye_top1 = target_bone_vecs['left_eye_top1'][fno]
    left_eye_top5 = target_bone_vecs['left_eye_top5'][fno]
    left_eye_top9 = target_bone_vecs['left_eye_top9'][fno]

    # 鼻の幅
    nose_width = abs(left_nose_2.x() - right_nose_2.x())
    
    # 眉のしかめ具合
    frown_ratio = abs(left_eye_brow1.x() - right_eye_brow1.x()) / nose_width

    # 眉の幅
    eye_brow_length = (euclidean_distance(left_eye_brow1, left_eye_brow6) + euclidean_distance(right_eye_brow1, right_eye_brow6)) / 2
    # 目の幅
    eye_length = (euclidean_distance(left_eye_top1, left_eye_top9) + euclidean_distance(right_eye_top1, right_eye_top9)) / 2

    # 目と眉の縦幅
    left_vertical_length = (euclidean_distance(left_eye_top1, left_eye_brow1) + euclidean_distance(right_eye_top1, right_eye_brow1)) / 2
    center_vertical_length = (euclidean_distance(left_eye_top5, left_eye_brow4) + euclidean_distance(right_eye_top5, right_eye_brow4)) / 2
    right_vertical_length = (euclidean_distance(left_eye_top9, left_eye_brow6) + euclidean_distance(right_eye_top9, right_eye_brow6)) / 2

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

# 目モーフ
def calc_eye(fno: int, motion: VmdMotion, target_bone_vecs: dict):
    calc_left_eye(fno, motion, target_bone_vecs)
    calc_right_eye(fno, motion, target_bone_vecs)
    blend_eye(fno, motion)

def calc_left_eye(fno: int, motion: VmdMotion, target_bone_vecs: dict):
    left_eye_top1 = target_bone_vecs['left_eye_top1'][fno]
    left_eye_top2 = target_bone_vecs['left_eye_top2'][fno]
    left_eye_top3 = target_bone_vecs['left_eye_top3'][fno]
    left_eye_top4 = target_bone_vecs['left_eye_top4'][fno]
    left_eye_top5 = target_bone_vecs['left_eye_top5'][fno]
    left_eye_top6 = target_bone_vecs['left_eye_top6'][fno]
    left_eye_top7 = target_bone_vecs['left_eye_top7'][fno]
    left_eye_top8 = target_bone_vecs['left_eye_top8'][fno]
    left_eye_top9 = target_bone_vecs['left_eye_top9'][fno]
    left_eye_bottom1 = target_bone_vecs['left_eye_bottom1'][fno]
    left_eye_bottom2 = target_bone_vecs['left_eye_bottom2'][fno]
    left_eye_bottom3 = target_bone_vecs['left_eye_bottom3'][fno]
    left_eye_bottom4 = target_bone_vecs['left_eye_bottom4'][fno]
    left_eye_bottom5 = target_bone_vecs['left_eye_bottom5'][fno]
    left_eye_bottom6 = target_bone_vecs['left_eye_bottom6'][fno]
    left_eye_bottom7 = target_bone_vecs['left_eye_bottom7'][fno]
    left_eye_bottom8 = target_bone_vecs['left_eye_bottom8'][fno]
    left_eye_bottom9 = target_bone_vecs['left_eye_bottom9'][fno]

    # 左目のEAR(eyes aspect ratio)
    left_blink, left_smile = get_blink_ratio(fno, left_eye_top1, left_eye_top3, left_eye_top6, left_eye_top9, left_eye_bottom3, left_eye_bottom6)

    mf = VmdMorphFrame(fno)
    mf.set_name("ウィンク右")
    mf.ratio = max(0, min(1, left_smile))

    motion.regist_mf(mf, mf.name, mf.fno)

    mf = VmdMorphFrame(fno)
    mf.set_name("ｳｨﾝｸ２右")
    mf.ratio = max(0, min(1, left_blink))

    motion.regist_mf(mf, mf.name, mf.fno)

def calc_right_eye(fno: int, motion: VmdMotion, target_bone_vecs: dict):
    right_eye_top1 = target_bone_vecs['right_eye_top1'][fno]
    right_eye_top2 = target_bone_vecs['right_eye_top2'][fno]
    right_eye_top3 = target_bone_vecs['right_eye_top3'][fno]
    right_eye_top4 = target_bone_vecs['right_eye_top4'][fno]
    right_eye_top5 = target_bone_vecs['right_eye_top5'][fno]
    right_eye_top6 = target_bone_vecs['right_eye_top6'][fno]
    right_eye_top7 = target_bone_vecs['right_eye_top7'][fno]
    right_eye_top8 = target_bone_vecs['right_eye_top8'][fno]
    right_eye_top9 = target_bone_vecs['right_eye_top9'][fno]
    right_eye_bottom1 = target_bone_vecs['right_eye_bottom1'][fno]
    right_eye_bottom2 = target_bone_vecs['right_eye_bottom2'][fno]
    right_eye_bottom3 = target_bone_vecs['right_eye_bottom3'][fno]
    right_eye_bottom4 = target_bone_vecs['right_eye_bottom4'][fno]
    right_eye_bottom5 = target_bone_vecs['right_eye_bottom5'][fno]
    right_eye_bottom6 = target_bone_vecs['right_eye_bottom6'][fno]
    right_eye_bottom7 = target_bone_vecs['right_eye_bottom7'][fno]
    right_eye_bottom8 = target_bone_vecs['right_eye_bottom8'][fno]
    right_eye_bottom9 = target_bone_vecs['right_eye_bottom9'][fno]

    # 右目のEAR(eyes aspect ratio)
    right_blink, right_smile = get_blink_ratio(fno, right_eye_top1, right_eye_top3, right_eye_top6, right_eye_top9, right_eye_bottom3, right_eye_bottom6)

    mf = VmdMorphFrame(fno)
    mf.set_name("ウィンク")
    mf.ratio = max(0, min(1, right_smile))
    motion.regist_mf(mf, mf.name, mf.fno)

    mf = VmdMorphFrame(fno)
    mf.set_name("ウィンク２")
    mf.ratio = max(0, min(1, right_blink))
    motion.regist_mf(mf, mf.name, mf.fno)

    motion.regist_mf(mf, mf.name, mf.fno)

def blend_eye(fno: int, motion: VmdMotion):
    min_blink = min(motion.morphs["ウィンク右"][fno].ratio, motion.morphs["ウィンク"][fno].ratio)
    min_smile = min(motion.morphs["ｳｨﾝｸ２右"][fno].ratio, motion.morphs["ウィンク２"][fno].ratio)

    # 両方の同じ値はさっぴく
    motion.morphs["ウィンク右"][fno].ratio -= min_smile
    motion.morphs["ウィンク"][fno].ratio -= min_smile

    motion.morphs["ｳｨﾝｸ２右"][fno].ratio -= min_blink
    motion.morphs["ウィンク２"][fno].ratio -= min_blink

    for morph_name in ["ウィンク右", "ウィンク", "ｳｨﾝｸ２右", "ウィンク２"]:
        motion.morphs[morph_name][fno].ratio = max(0, min(1, motion.morphs[morph_name][fno].ratio))

    mf = VmdMorphFrame(fno)
    mf.set_name("笑い")
    mf.ratio = max(0, min(1, min_smile))
    motion.regist_mf(mf, mf.name, mf.fno)

    mf = VmdMorphFrame(fno)
    mf.set_name("まばたき")
    mf.ratio = max(0, min(1, min_blink))
    motion.regist_mf(mf, mf.name, mf.fno)

    # # 両目の平均とする
    # mean_eye_euler = (right_eye_euler + left_eye_euler) / 2
    # eye_bf = motion.calc_bf("両目", fno)
    # eye_bf.rotation = MQuaternion.fromEulerAngles(mean_eye_euler.x(), mean_eye_euler.y(), 0)
    # motion.regist_bf(eye_bf, "両目", fno)
    
def get_blink_ratio(fno: int, eye_top1: MVector3D, eye_top3: MVector3D, eye_top6: MVector3D, eye_top9: MVector3D, eye_bottom3: MVector3D, eye_bottom6: MVector3D):
    #loading all the required points
    corner_left  = eye_top1
    corner_right = eye_top9
    corner_center = (eye_top1 + eye_top9) / 2
    
    center_top = (eye_top3 + eye_top6) / 2
    center_bottom = (eye_bottom3 + eye_bottom6) / 2

    #calculating distance
    horizontal_length = euclidean_distance(corner_left, corner_right)
    vertical_length = euclidean_distance(center_top, center_bottom)

    ratio = horizontal_length / vertical_length
    new_ratio = min(1, math.sin(calc_ratio(ratio, 0, 20, 0, 1)))

    # 笑いの比率(日本人用に比率半分)
    smile_ratio = (center_bottom.y() - corner_center.y()) / (center_bottom.y() - center_top.y())

    if smile_ratio > 1:
        # １より大きい場合、目頭よりも上瞼が下にあるという事なので、通常瞬きと見なす
        return 1, 0
    
    return new_ratio * (1 - smile_ratio), new_ratio * smile_ratio

def euclidean_distance(point1: MVector3D, point2: MVector3D):
    return math.sqrt((point1.x() - point2.x())**2 + (point1.y() - point2.y())**2)

def create_bone_leg_ik(pmx: PmxModel, direction: str):
    leg_name = f'{direction}足'
    knee_name = f'{direction}ひざ'
    ankle_name = f'{direction}足首'
    toe_name = f'{direction}つま先'
    leg_ik_name = f'{direction}足ＩＫ'
    toe_ik_name = f'{direction}つま先ＩＫ'

    if leg_name in pmx.bones and knee_name in pmx.bones and ankle_name in pmx.bones:
        # 足ＩＫ
        flag = 0x0000 | 0x0002 | 0x0004 | 0x0008 | 0x0010 | 0x0020
        leg_ik_link = []
        leg_ik_link.append(IkLink(pmx.bones[knee_name].index, 1, MVector3D(math.radians(-180), 0, 0), MVector3D(math.radians(-0.5), 0, 0)))
        leg_ik_link.append(IkLink(pmx.bones[leg_name].index, 0))
        leg_ik = Ik(pmx.bones[ankle_name].index, 40, 1, leg_ik_link)
        leg_ik_bone = Bone(leg_ik_name, leg_ik_name, pmx.bones[ankle_name].position, 0, 0, flag, MVector3D(0, 0, 1), -1, ik=leg_ik)
        leg_ik_bone.index = len(pmx.bones)
        pmx.bones[leg_ik_bone.name] = leg_ik_bone
        pmx.bone_indexes[leg_ik_bone.index] = leg_ik_bone.name

        toe_ik_link = []
        toe_ik_link.append(IkLink(pmx.bones[ankle_name].index, 0))
        toe_ik = Ik(pmx.bones[toe_name].index, 40, 1, toe_ik_link)
        toe_ik_bone = Bone(toe_ik_name, toe_ik_name, pmx.bones[ankle_name].position, 0, 0, flag, MVector3D(0, -1, 0), -1, ik=toe_ik)
        toe_ik_bone.index = len(pmx.bones)
        pmx.bones[toe_ik_bone.name] = toe_ik_bone
        pmx.bone_indexes[toe_ik_bone.index] = toe_ik_bone.name

MP_PMX_CONNECTIONS = {
    "mp_pelvis": {"mmd": "下半身", "parent": "Groove", "tail": "mp_pelvis2", "display": "体幹", "axis": None},
    "mp_pelvis2": {"mmd": "下半身先", "parent": "mp_pelvis", "tail": "", "display": "体幹", "axis": None},
    "mp_left_hip": {"mmd": "左足", "parent": "mp_pelvis", "tail": "mp_left_knee", "display": "左足", "axis": None},
    "mp_right_hip": {"mmd": "右足", "parent": "mp_pelvis", "tail": "mp_right_knee", "display": "右足", "axis": None},
    "mp_left_knee": {"mmd": "左ひざ", "parent": "mp_left_hip", "tail": "mp_left_ankle", "display": "左足", "axis": None},
    "mp_right_knee": {"mmd": "右ひざ", "parent": "mp_right_hip", "tail": "mp_right_ankle", "display": "右足", "axis": None},
    "mp_left_ankle": {"mmd": "左足首", "parent": "mp_left_knee", "tail": "mp_left_foot", "display": "左足", "axis": None},
    "mp_right_ankle": {"mmd": "右足首", "parent": "mp_right_knee", "tail": "mp_right_foot", "display": "右足", "axis": None},
    "mp_left_foot": {"mmd": "左つま先", "parent": "mp_left_ankle", "tail": "", "display": "左足", "axis": None},
    "mp_right_foot": {"mmd": "右つま先", "parent": "mp_right_ankle", "tail": "", "display": "右足", "axis": None},
    "mp_spine1": {"mmd": "上半身", "parent": "Groove", "tail": "mp_neck", "display": "体幹", "axis": None},
    "mp_neck": {"mmd": "首", "parent": "mp_spine1", "tail": "mp_head", "display": "体幹", "axis": None},
    "mp_head": {"mmd": "頭", "parent": "mp_neck", "tail": "mp_head_tail", "display": "体幹", "axis": MVector3D(1, 0, 0), "parent_axis": MVector3D(1, 0, 0)},
    "mp_head_tail": {"mmd": "頭先", "parent": "mp_head", "tail": "", "display": "体幹", "axis": None},
    "mp_left_collar": {"mmd": "左肩", "parent": "mp_spine1", "tail": "mp_left_shoulder", "display": "左手", "axis": MVector3D(1, 0, 0)},
    "mp_right_collar": {"mmd": "右肩", "parent": "mp_spine1", "tail": "mp_right_shoulder", "display": "右手", "axis": MVector3D(-1, 0, 0)},
    "mp_left_shoulder": {"mmd": "左腕", "parent": "mp_left_collar", "tail": "mp_left_elbow", "display": "左手", "axis": MVector3D(1, 0, 0)},
    "mp_right_shoulder": {"mmd": "右腕", "parent": "mp_right_collar", "tail": "mp_right_elbow", "display": "右手", "axis": MVector3D(-1, 0, 0)},
    "mp_left_elbow": {"mmd": "左ひじ", "parent": "mp_left_shoulder", "tail": "mp_body_left_wrist", "display": "左手", "axis": MVector3D(1, 0, 0)},
    "mp_right_elbow": {"mmd": "右ひじ", "parent": "mp_right_shoulder", "tail": "mp_body_right_wrist", "display": "右手", "axis": MVector3D(-1, 0, 0)},
    "mp_body_left_wrist": {"mmd": "左手首", "parent": "mp_left_elbow", "tail": "", "display": "左手", "axis": MVector3D(1, 0, 0)},
    "mp_body_right_wrist": {"mmd": "右手首", "parent": "mp_right_elbow", "tail": "", "display": "右手", "axis": MVector3D(-1, 0, 0)},
    "mp_right_eye": {"mmd": "右目", "parent": "mp_head", "tail": "", "display": "顔", "axis": None},
    "mp_left_eye": {"mmd": "左目", "parent": "mp_head", "tail": "", "display": "顔", "axis": None},
    "mp_left_foot_index": {"mmd": "左つま先", "parent": "mp_left_ankle", "tail": "", "display": "左足", "axis": None},
    "mp_left_heel": {"mmd": "左かかと", "parent": "mp_left_ankle", "tail": "", "display": "左足", "axis": None},
    "mp_right_foot_index": {"mmd": "右つま先", "parent": "mp_right_ankle", "tail": "", "display": "右足", "axis": None},
    "mp_right_heel": {"mmd": "右かかと", "parent": "mp_right_ankle", "tail": "", "display": "右足", "axis": None},
}


PMX_CONNECTIONS = {
    "pelvis": {"mmd": "下半身", "parent": "グルーブ", "tail": "pelvis2", "display": "体幹", "axis": None},
    "pelvis2": {"mmd": "下半身先", "parent": "pelvis", "tail": "", "display": "体幹", "axis": None},
    "left_hip": {"mmd": "左足", "parent": "pelvis", "tail": "left_knee", "display": "左足", "axis": None},
    "right_hip": {"mmd": "右足", "parent": "pelvis", "tail": "right_knee", "display": "右足", "axis": None},
    "left_knee": {"mmd": "左ひざ", "parent": "left_hip", "tail": "left_ankle", "display": "左足", "axis": None},
    "right_knee": {"mmd": "右ひざ", "parent": "right_hip", "tail": "right_ankle", "display": "右足", "axis": None},
    "left_ankle": {"mmd": "左足首", "parent": "left_knee", "tail": "left_foot", "display": "左足", "axis": None},
    "right_ankle": {"mmd": "右足首", "parent": "right_knee", "tail": "right_foot", "display": "右足", "axis": None},
    "left_foot": {"mmd": "左つま先", "parent": "left_ankle", "tail": "", "display": "左足", "axis": None},
    "right_foot": {"mmd": "右つま先", "parent": "right_ankle", "tail": "", "display": "右足", "axis": None},
    "spine1": {"mmd": "上半身", "parent": "グルーブ", "tail": "spine2", "display": "体幹", "axis": None},
    "spine2": {"mmd": "上半身2", "parent": "spine1", "tail": "neck", "display": "体幹", "axis": None},
    "neck": {"mmd": "首", "parent": "spine2", "tail": "head", "display": "体幹", "axis": None},
    "head": {"mmd": "頭", "parent": "neck", "tail": "head_tail", "display": "体幹", "axis": MVector3D(1, 0, 0), "parent_axis": MVector3D(1, 0, 0)},
    "left_collar": {"mmd": "左肩", "parent": "spine2", "tail": "left_shoulder", "display": "左手", "axis": MVector3D(1, 0, 0)},
    "right_collar": {"mmd": "右肩", "parent": "spine2", "tail": "right_shoulder", "display": "右手", "axis": MVector3D(-1, 0, 0)},
    "head_tail": {"mmd": "頭先", "parent": "head", "tail": "", "display": "体幹", "axis": None},
    "left_shoulder": {"mmd": "左腕", "parent": "left_collar", "tail": "left_elbow", "display": "左手", "axis": MVector3D(1, 0, 0)},
    "right_shoulder": {"mmd": "右腕", "parent": "right_collar", "tail": "right_elbow", "display": "右手", "axis": MVector3D(-1, 0, 0)},
    "left_elbow": {"mmd": "左ひじ", "parent": "left_shoulder", "tail": "left_wrist", "display": "左手", "axis": MVector3D(1, 0, 0)},
    "right_elbow": {"mmd": "右ひじ", "parent": "right_shoulder", "tail": "right_wrist", "display": "右手", "axis": MVector3D(-1, 0, 0)},
    "left_wrist": {"mmd": "左手首", "parent": "left_elbow", "tail": "left_wrist_tail", "display": "左手", "axis": MVector3D(1, 0, 0)},
    "right_wrist": {"mmd": "右手首", "parent": "right_elbow", "tail": "right_wrist_tail", "display": "右手", "axis": MVector3D(-1, 0, 0)},
    "left_wrist_tail": {"mmd": "左手先", "parent": "left_wrist", "tail": "", "display": "左手", "axis": MVector3D(1, 0, 0)},
    "right_wrist_tail": {"mmd": "右手先", "parent": "right_wrist", "tail": "", "display": "右手", "axis": MVector3D(-1, 0, 0)},
    "jaw": {"mmd": "jaw", "parent": "head", "tail": "", "display": "顔", "axis": None},
    "left_eye_smplx": {"mmd": "left_eye_smplx", "parent": "head", "tail": "", "display": "顔", "axis": None},
    "right_eye_smplx": {"mmd": "right_eye_smplx", "parent": "head", "tail": "", "display": "顔", "axis": None},
    "nose": {"mmd": "nose", "parent": "head", "tail": "", "display": "顔", "axis": None},
    "right_eye": {"mmd": "右目", "parent": "nose", "tail": "", "display": "顔", "axis": None},
    "left_eye": {"mmd": "左目", "parent": "nose", "tail": "", "display": "顔", "axis": None},
    "right_ear": {"mmd": "right_ear", "parent": "nose", "tail": "", "display": "顔", "axis": None},
    "left_ear": {"mmd": "left_ear", "parent": "nose", "tail": "", "display": "顔", "axis": None},
    "left_big_toe": {"mmd": "左足親指", "parent": "left_ankle", "tail": "", "display": "左足", "axis": None},
    "left_small_toe": {"mmd": "左足小指", "parent": "left_ankle", "tail": "", "display": "左足", "axis": None},
    "left_heel": {"mmd": "左かかと", "parent": "left_ankle", "tail": "", "display": "左足", "axis": None},
    "right_big_toe": {"mmd": "右足親指", "parent": "right_ankle", "tail": "", "display": "右足", "axis": None},
    "right_small_toe": {"mmd": "右足小指", "parent": "right_ankle", "tail": "", "display": "右足", "axis": None},
    "right_heel": {"mmd": "右かかと", "parent": "right_ankle", "tail": "", "display": "右足", "axis": None},

    "mp_left_wrist": {"mmd": "左手首2", "parent": "left_wrist", "tail": "mp_left_middle1", "display": "左手", "axis": MVector3D(1, 0, 0), "hand": True},
    "mp_right_wrist": {"mmd": "右手首2", "parent": "right_wrist", "tail": "mp_right_middle1", "display": "右手", "axis": MVector3D(-1, 0, 0), "hand": True},
    "mp_left_index1": {"mmd": "左人指１", "parent": "mp_left_wrist", "tail": "mp_left_index2", "display": "左指", "axis": MVector3D(1, 0, 0), "hand": True},
    "mp_left_index2": {"mmd": "左人指２", "parent": "mp_left_index1", "tail": "mp_left_index3", "display": "左指", "axis": MVector3D(1, 0, 0), "hand": True},
    "mp_left_index3": {"mmd": "左人指３", "parent": "mp_left_index2", "tail": "mp_left_index", "display": "左指", "axis": MVector3D(1, 0, 0), "hand": True},
    "mp_left_middle1": {"mmd": "左中指１", "parent": "mp_left_wrist", "tail": "mp_left_middle2", "display": "左指", "axis": MVector3D(1, 0, 0), "hand": True},
    "mp_left_middle2": {"mmd": "左中指２", "parent": "mp_left_middle1", "tail": "mp_left_middle3", "display": "左指", "axis": MVector3D(1, 0, 0), "hand": True},
    "mp_left_middle3": {"mmd": "左中指３", "parent": "mp_left_middle2", "tail": "mp_left_middle", "display": "左指", "axis": MVector3D(1, 0, 0), "hand": True},
    "mp_left_pinky1": {"mmd": "左小指１", "parent": "mp_left_wrist", "tail": "mp_left_pinky2", "display": "左指", "axis": MVector3D(1, 0, 0), "hand": True},
    "mp_left_pinky2": {"mmd": "左小指２", "parent": "mp_left_pinky1", "tail": "mp_left_pinky3", "display": "左指", "axis": MVector3D(1, 0, 0), "hand": True},
    "mp_left_pinky3": {"mmd": "左小指３", "parent": "mp_left_pinky2", "tail": "mp_left_pinky", "display": "左指", "axis": MVector3D(1, 0, 0), "hand": True},
    "mp_left_ring1": {"mmd": "左薬指１", "parent": "mp_left_wrist", "tail": "mp_left_ring2", "display": "左指", "axis": MVector3D(1, 0, 0), "hand": True},
    "mp_left_ring2": {"mmd": "左薬指２", "parent": "mp_left_ring1", "tail": "mp_left_ring3", "display": "左指", "axis": MVector3D(1, 0, 0), "hand": True},
    "mp_left_ring3": {"mmd": "左薬指３", "parent": "mp_left_ring2", "tail": "mp_left_ring", "display": "左指", "axis": MVector3D(1, 0, 0), "hand": True},
    "mp_left_thumb1": {"mmd": "左親指０", "parent": "mp_left_wrist", "tail": "mp_left_thumb2", "display": "左指", "axis": MVector3D(1, 0, 0), "hand": True},
    "mp_left_thumb2": {"mmd": "左親指１", "parent": "mp_left_thumb1", "tail": "mp_left_thumb3", "display": "左指", "axis": MVector3D(1, 0, 0), "hand": True},
    "mp_left_thumb3": {"mmd": "左親指２", "parent": "mp_left_thumb2", "tail": "mp_left_thumb", "display": "左指", "axis": MVector3D(1, 0, 0), "hand": True},
    "mp_right_index1": {"mmd": "右人指１", "parent": "mp_right_wrist", "tail": "mp_right_index2", "display": "右指", "axis": MVector3D(-1, 0, 0), "hand": True},
    "mp_right_index2": {"mmd": "右人指２", "parent": "mp_right_index1", "tail": "mp_right_index3", "display": "右指", "axis": MVector3D(-1, 0, 0), "hand": True},
    "mp_right_index3": {"mmd": "右人指３", "parent": "mp_right_index2", "tail": "mp_right_index", "display": "右指", "axis": MVector3D(-1, 0, 0), "hand": True},
    "mp_right_middle1": {"mmd": "右中指１", "parent": "mp_right_wrist", "tail": "mp_right_middle2", "display": "右指", "axis": MVector3D(-1, 0, 0), "hand": True},
    "mp_right_middle2": {"mmd": "右中指２", "parent": "mp_right_middle1", "tail": "mp_right_middle3", "display": "右指", "axis": MVector3D(-1, 0, 0), "hand": True},
    "mp_right_middle3": {"mmd": "右中指３", "parent": "mp_right_middle2", "tail": "mp_right_middle", "display": "右指", "axis": MVector3D(-1, 0, 0), "hand": True},
    "mp_right_pinky1": {"mmd": "右小指１", "parent": "mp_right_wrist", "tail": "mp_right_pinky2", "display": "右指", "axis": MVector3D(-1, 0, 0), "hand": True},
    "mp_right_pinky2": {"mmd": "右小指２", "parent": "mp_right_pinky1", "tail": "mp_right_pinky3", "display": "右指", "axis": MVector3D(-1, 0, 0), "hand": True},
    "mp_right_pinky3": {"mmd": "右小指３", "parent": "mp_right_pinky2", "tail": "mp_right_pinky", "display": "右指", "axis": MVector3D(-1, 0, 0), "hand": True},
    "mp_right_ring1": {"mmd": "右薬指１", "parent": "mp_right_wrist", "tail": "mp_right_ring2", "display": "右指", "axis": MVector3D(-1, 0, 0), "hand": True},
    "mp_right_ring2": {"mmd": "右薬指２", "parent": "mp_right_ring1", "tail": "mp_right_ring3", "display": "右指", "axis": MVector3D(-1, 0, 0), "hand": True},
    "mp_right_ring3": {"mmd": "右薬指３", "parent": "mp_right_ring2", "tail": "mp_right_ring", "display": "右指", "axis": MVector3D(-1, 0, 0), "hand": True},
    "mp_right_thumb1": {"mmd": "右親指０", "parent": "mp_right_wrist", "tail": "mp_right_thumb2", "display": "右指", "axis": MVector3D(-1, 0, 0), "hand": True},
    "mp_right_thumb2": {"mmd": "右親指１", "parent": "mp_right_thumb1", "tail": "mp_right_thumb3", "display": "右指", "axis": MVector3D(-1, 0, 0), "hand": True},
    "mp_right_thumb3": {"mmd": "右親指２", "parent": "mp_right_thumb2", "tail": "mp_right_thumb", "display": "右指", "axis": MVector3D(-1, 0, 0), "hand": True},
    "mp_left_thumb": {"mmd": "左親指先", "parent": "mp_left_thumb3", "tail": "", "display": "左指", "axis": MVector3D(1, 0, 0), "hand": True},
    "mp_left_index": {"mmd": "左人差指先", "parent": "mp_left_index3", "tail": "", "display": "左指", "axis": MVector3D(1, 0, 0), "hand": True},
    "mp_left_middle": {"mmd": "左中指先", "parent": "mp_left_middle3", "tail": "", "display": "左指", "axis": MVector3D(1, 0, 0), "hand": True},
    "mp_left_ring": {"mmd": "左薬指先", "parent": "mp_left_ring3", "tail": "", "display": "左指", "axis": MVector3D(1, 0, 0), "hand": True},
    "mp_left_pinky": {"mmd": "左小指先", "parent": "mp_left_pinky3", "tail": "", "display": "左指", "axis": MVector3D(1, 0, 0), "hand": True},
    "mp_right_thumb": {"mmd": "右親指先", "parent": "mp_right_thumb3", "tail": "", "display": "右指", "axis": MVector3D(-1, 0, 0), "hand": True},
    "mp_right_index": {"mmd": "右人差指先", "parent": "mp_right_index3", "tail": "", "display": "右指", "axis": MVector3D(-1, 0, 0), "hand": True},
    "mp_right_middle": {"mmd": "右中指先", "parent": "mp_right_middle3", "tail": "", "display": "右指", "axis": MVector3D(-1, 0, 0), "hand": True},
    "mp_right_ring": {"mmd": "右薬指先", "parent": "mp_right_ring3", "tail": "", "display": "右指", "axis": MVector3D(-1, 0, 0), "hand": True},
    "mp_right_pinky": {"mmd": "右小指先", "parent": "mp_right_pinky3", "tail": "", "display": "右指", "axis": MVector3D(-1, 0, 0), "hand": True},
    
    # https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    "right_eye_brow1": {"mmd": "right_eye_brow1", "parent": "head", "tail": "", "display": "眉", "axis": None, "mp_index": "221"},
    "right_eye_brow2": {"mmd": "right_eye_brow2", "parent": "right_eye_brow1", "tail": "", "display": "眉", "axis": None, "mp_index": "222"},
    "right_eye_brow3": {"mmd": "right_eye_brow3", "parent": "right_eye_brow2", "tail": "", "display": "眉", "axis": None, "mp_index": "223"},
    "right_eye_brow4": {"mmd": "right_eye_brow4", "parent": "right_eye_brow3", "tail": "", "display": "眉", "axis": None, "mp_index": "224"},
    "right_eye_brow5": {"mmd": "right_eye_brow5", "parent": "right_eye_brow4", "tail": "", "display": "眉", "axis": None, "mp_index": "225"},
    "right_eye_brow6": {"mmd": "right_eye_brow5", "parent": "right_eye_brow4", "tail": "", "display": "眉", "axis": None, "mp_index": "226"},
    "left_eye_brow1": {"mmd": "left_eye_brow1", "parent": "head", "tail": "", "display": "眉", "axis": None, "mp_index": "441"},
    "left_eye_brow2": {"mmd": "left_eye_brow2", "parent": "left_eye_brow1", "tail": "", "display": "眉", "axis": None, "mp_index": "442"},
    "left_eye_brow3": {"mmd": "left_eye_brow3", "parent": "left_eye_brow2", "tail": "", "display": "眉", "axis": None, "mp_index": "443"},
    "left_eye_brow4": {"mmd": "left_eye_brow4", "parent": "left_eye_brow3", "tail": "", "display": "眉", "axis": None, "mp_index": "444"},
    "left_eye_brow5": {"mmd": "left_eye_brow5", "parent": "left_eye_brow4", "tail": "", "display": "眉", "axis": None, "mp_index": "445"},
    "left_eye_brow6": {"mmd": "left_eye_brow5", "parent": "left_eye_brow4", "tail": "", "display": "眉", "axis": None, "mp_index": "342"},

    "right_eye_top1": {"mmd": "right_eye_top1", "parent": "right_eye", "tail": "", "display": "目", "axis": None, "mp_index": "133"},
    "right_eye_top2": {"mmd": "right_eye_top2", "parent": "right_eye_top1", "tail": "", "display": "目", "axis": None, "mp_index": "173"},
    "right_eye_top3": {"mmd": "right_eye_top3", "parent": "right_eye_top2", "tail": "", "display": "目", "axis": None, "mp_index": "157"},
    "right_eye_top4": {"mmd": "right_eye_top4", "parent": "right_eye_top3", "tail": "", "display": "目", "axis": None, "mp_index": "158"},
    "right_eye_top5": {"mmd": "right_eye_top5", "parent": "right_eye_top4", "tail": "", "display": "目", "axis": None, "mp_index": "159"},
    "right_eye_top6": {"mmd": "right_eye_top6", "parent": "right_eye_top5", "tail": "", "display": "目", "axis": None, "mp_index": "160"},
    "right_eye_top7": {"mmd": "right_eye_top7", "parent": "right_eye_top6", "tail": "", "display": "目", "axis": None, "mp_index": "161"},
    "right_eye_top8": {"mmd": "right_eye_top8", "parent": "right_eye_top7", "tail": "", "display": "目", "axis": None, "mp_index": "246"},
    "right_eye_top9": {"mmd": "right_eye_top9", "parent": "right_eye_top8", "tail": "", "display": "目", "axis": None, "mp_index": "33"},
    "right_eye_bottom1": {"mmd": "right_eye_bottom1", "parent": "right_eye", "tail": "", "display": "目", "axis": None, "mp_index": "133"},
    "right_eye_bottom2": {"mmd": "right_eye_bottom2", "parent": "right_eye_bottom1", "tail": "", "display": "目", "axis": None, "mp_index": "155"},
    "right_eye_bottom3": {"mmd": "right_eye_bottom3", "parent": "right_eye_bottom2", "tail": "", "display": "目", "axis": None, "mp_index": "154"},
    "right_eye_bottom4": {"mmd": "right_eye_bottom4", "parent": "right_eye_bottom3", "tail": "", "display": "目", "axis": None, "mp_index": "153"},
    "right_eye_bottom5": {"mmd": "right_eye_bottom5", "parent": "right_eye_bottom4", "tail": "", "display": "目", "axis": None, "mp_index": "145"},
    "right_eye_bottom6": {"mmd": "right_eye_bottom6", "parent": "right_eye_bottom5", "tail": "", "display": "目", "axis": None, "mp_index": "144"},
    "right_eye_bottom7": {"mmd": "right_eye_bottom7", "parent": "right_eye_bottom6", "tail": "", "display": "目", "axis": None, "mp_index": "163"},
    "right_eye_bottom8": {"mmd": "right_eye_bottom8", "parent": "right_eye_bottom7", "tail": "", "display": "目", "axis": None, "mp_index": "7"},
    "right_eye_bottom9": {"mmd": "right_eye_bottom9", "parent": "right_eye_bottom8", "tail": "", "display": "目", "axis": None, "mp_index": "33"},
    "left_eye_top1": {"mmd": "left_eye_top1", "parent": "left_eye", "tail": "", "display": "目", "axis": None, "mp_index": "362"},
    "left_eye_top2": {"mmd": "left_eye_top2", "parent": "left_eye_top1", "tail": "", "display": "目", "axis": None, "mp_index": "398"},
    "left_eye_top3": {"mmd": "left_eye_top3", "parent": "left_eye_top2", "tail": "", "display": "目", "axis": None, "mp_index": "384"},
    "left_eye_top4": {"mmd": "left_eye_top4", "parent": "left_eye_top3", "tail": "", "display": "目", "axis": None, "mp_index": "385"},
    "left_eye_top5": {"mmd": "left_eye_top5", "parent": "left_eye_top4", "tail": "", "display": "目", "axis": None, "mp_index": "386"},
    "left_eye_top6": {"mmd": "left_eye_top6", "parent": "left_eye_top5", "tail": "", "display": "目", "axis": None, "mp_index": "387"},
    "left_eye_top7": {"mmd": "left_eye_top7", "parent": "left_eye_top6", "tail": "", "display": "目", "axis": None, "mp_index": "388"},
    "left_eye_top8": {"mmd": "left_eye_top8", "parent": "left_eye_top7", "tail": "", "display": "目", "axis": None, "mp_index": "466"},
    "left_eye_top9": {"mmd": "left_eye_top9", "parent": "left_eye_top8", "tail": "", "display": "目", "axis": None, "mp_index": "263"},
    "left_eye_bottom1": {"mmd": "left_eye_bottom1", "parent": "left_eye", "tail": "", "display": "目", "axis": None, "mp_index": "362"},
    "left_eye_bottom2": {"mmd": "left_eye_bottom2", "parent": "left_eye_bottom1", "tail": "", "display": "目", "axis": None, "mp_index": "382"},
    "left_eye_bottom3": {"mmd": "left_eye_bottom3", "parent": "left_eye_bottom2", "tail": "", "display": "目", "axis": None, "mp_index": "381"},
    "left_eye_bottom4": {"mmd": "left_eye_bottom4", "parent": "left_eye_bottom3", "tail": "", "display": "目", "axis": None, "mp_index": "380"},
    "left_eye_bottom5": {"mmd": "left_eye_bottom5", "parent": "left_eye_bottom4", "tail": "", "display": "目", "axis": None, "mp_index": "374"},
    "left_eye_bottom6": {"mmd": "left_eye_bottom6", "parent": "left_eye_bottom5", "tail": "", "display": "目", "axis": None, "mp_index": "373"},
    "left_eye_bottom7": {"mmd": "left_eye_bottom7", "parent": "left_eye_bottom6", "tail": "", "display": "目", "axis": None, "mp_index": "390"},
    "left_eye_bottom8": {"mmd": "left_eye_bottom8", "parent": "left_eye_bottom7", "tail": "", "display": "目", "axis": None, "mp_index": "249"},
    "left_eye_bottom9": {"mmd": "left_eye_bottom9", "parent": "left_eye_bottom8", "tail": "", "display": "目", "axis": None, "mp_index": "263"},

    "nose1": {"mmd": "nose1", "parent": "nose", "tail": "", "display": "鼻", "axis": None, "mp_index": "6"},
    "nose2": {"mmd": "nose2", "parent": "nose1", "tail": "", "display": "鼻", "axis": None, "mp_index": "197"},
    "nose3": {"mmd": "nose3", "parent": "nose2", "tail": "", "display": "鼻", "axis": None, "mp_index": "195"},
    "nose4": {"mmd": "nose4", "parent": "nose3", "tail": "", "display": "鼻", "axis": None, "mp_index": "5"},
    "right_nose_1": {"mmd": "right_nose_1", "parent": "nose", "tail": "", "display": "鼻", "axis": None, "mp_index": "49"},
    "right_nose_2": {"mmd": "right_nose_2", "parent": "nose", "tail": "", "display": "鼻", "axis": None, "mp_index": "45"},
    "nose_middle": {"mmd": "nose_middle", "parent": "nose", "tail": "", "display": "鼻", "axis": None, "mp_index": "4"},
    "left_nose_1": {"mmd": "left_nose_1", "parent": "nose", "tail": "", "display": "鼻", "axis": None, "mp_index": "279"},
    "left_nose_2": {"mmd": "left_nose_2", "parent": "nose", "tail": "", "display": "鼻", "axis": None, "mp_index": "278"},

    "right_lip_top_1": {"mmd": "right_lip_top_1", "parent": "head", "tail": "", "display": "口", "axis": None, "mp_index": "78"},
    "right_lip_top_2": {"mmd": "right_lip_top_2", "parent": "right_lip_top_1", "tail": "", "display": "口", "axis": None, "mp_index": "191"},
    "right_lip_top_3": {"mmd": "right_lip_top_3", "parent": "right_lip_top_2", "tail": "", "display": "口", "axis": None, "mp_index": "80"},
    "right_lip_top_4": {"mmd": "right_lip_top_4", "parent": "right_lip_top_3", "tail": "", "display": "口", "axis": None, "mp_index": "81"},
    "right_lip_top_5": {"mmd": "right_lip_top_5", "parent": "right_lip_top_4", "tail": "", "display": "口", "axis": None, "mp_index": "82"},
    "lip_top": {"mmd": "lip_top", "parent": "head", "tail": "", "display": "口", "axis": None, "mp_index": "13"},
    "left_lip_top_1": {"mmd": "left_lip_top_1", "parent": "head", "tail": "", "display": "口", "axis": None, "mp_index": "308"},
    "left_lip_top_2": {"mmd": "left_lip_top_2", "parent": "left_lip_top_1", "tail": "", "display": "口", "axis": None, "mp_index": "415"},
    "left_lip_top_3": {"mmd": "left_lip_top_3", "parent": "left_lip_top_2", "tail": "", "display": "口", "axis": None, "mp_index": "310"},
    "left_lip_top_4": {"mmd": "left_lip_top_4", "parent": "left_lip_top_3", "tail": "", "display": "口", "axis": None, "mp_index": "311"},
    "left_lip_top_5": {"mmd": "left_lip_top_5", "parent": "left_lip_top_4", "tail": "", "display": "口", "axis": None, "mp_index": "312"},
    
    "right_lip_bottom_1": {"mmd": "right_lip_bottom_1", "parent": "head", "tail": "", "display": "口", "axis": None, "mp_index": "78"},
    "right_lip_bottom_2": {"mmd": "right_lip_bottom_2", "parent": "right_lip_bottom_1", "tail": "", "display": "口", "axis": None, "mp_index": "95"},
    "right_lip_bottom_3": {"mmd": "right_lip_bottom_3", "parent": "right_lip_bottom_2", "tail": "", "display": "口", "axis": None, "mp_index": "88"},
    "right_lip_bottom_4": {"mmd": "right_lip_bottom_4", "parent": "right_lip_bottom_3", "tail": "", "display": "口", "axis": None, "mp_index": "178"},
    "right_lip_bottom_5": {"mmd": "right_lip_bottom_5", "parent": "right_lip_bottom_4", "tail": "", "display": "口", "axis": None, "mp_index": "87"},
    "lip_bottom": {"mmd": "lip_bottom", "parent": "head", "tail": "", "display": "口", "axis": None, "mp_index": "14"},
    "left_lip_bottom_1": {"mmd": "left_lip_bottom_1", "parent": "head", "tail": "", "display": "口", "axis": None, "mp_index": "308"},
    "left_lip_bottom_2": {"mmd": "left_lip_bottom_2", "parent": "left_lip_bottom_1", "tail": "", "display": "口", "axis": None, "mp_index": "324"},
    "left_lip_bottom_3": {"mmd": "left_lip_bottom_3", "parent": "left_lip_bottom_2", "tail": "", "display": "口", "axis": None, "mp_index": "318"},
    "left_lip_bottom_4": {"mmd": "left_lip_bottom_4", "parent": "left_lip_bottom_3", "tail": "", "display": "口", "axis": None, "mp_index": "402"},
    "left_lip_bottom_5": {"mmd": "left_lip_bottom_5", "parent": "left_lip_bottom_4", "tail": "", "display": "口", "axis": None, "mp_index": "317"},

    "right_mouth_top_1": {"mmd": "right_mouth_top_1", "parent": "head", "tail": "", "display": "口", "axis": None, "mp_index": "61"},
    "right_mouth_top_2": {"mmd": "right_mouth_top_2", "parent": "right_mouth_top_1", "tail": "", "display": "口", "axis": None, "mp_index": "195"},
    "right_mouth_top_3": {"mmd": "right_mouth_top_3", "parent": "right_mouth_top_2", "tail": "", "display": "口", "axis": None, "mp_index": "40"},
    "right_mouth_top_4": {"mmd": "right_mouth_top_4", "parent": "right_mouth_top_3", "tail": "", "display": "口", "axis": None, "mp_index": "39"},
    "right_mouth_top_5": {"mmd": "right_mouth_top_5", "parent": "right_mouth_top_4", "tail": "", "display": "口", "axis": None, "mp_index": "37"},
    "mouth_top": {"mmd": "mouth_top", "parent": "head", "tail": "", "display": "口", "axis": None, "mp_index": "0"},
    "left_mouth_top_1": {"mmd": "left_mouth_top_1", "parent": "head", "tail": "", "display": "口", "axis": None, "mp_index": "291"},
    "left_mouth_top_2": {"mmd": "left_mouth_top_2", "parent": "left_mouth_top_1", "tail": "", "display": "口", "axis": None, "mp_index": "409"},
    "left_mouth_top_3": {"mmd": "left_mouth_top_3", "parent": "left_mouth_top_2", "tail": "", "display": "口", "axis": None, "mp_index": "270"},
    "left_mouth_top_4": {"mmd": "left_mouth_top_4", "parent": "left_mouth_top_3", "tail": "", "display": "口", "axis": None, "mp_index": "269"},
    "left_mouth_top_5": {"mmd": "left_mouth_top_5", "parent": "left_mouth_top_4", "tail": "", "display": "口", "axis": None, "mp_index": "267"},
    
    "right_mouth_bottom_1": {"mmd": "right_mouth_bottom_1", "parent": "head", "tail": "", "display": "口", "axis": None, "mp_index": "61"},
    "right_mouth_bottom_2": {"mmd": "right_mouth_bottom_2", "parent": "right_mouth_bottom_1", "tail": "", "display": "口", "axis": None, "mp_index": "146"},
    "right_mouth_bottom_3": {"mmd": "right_mouth_bottom_3", "parent": "right_mouth_bottom_2", "tail": "", "display": "口", "axis": None, "mp_index": "91"},
    "right_mouth_bottom_4": {"mmd": "right_mouth_bottom_4", "parent": "right_mouth_bottom_3", "tail": "", "display": "口", "axis": None, "mp_index": "181"},
    "right_mouth_bottom_5": {"mmd": "right_mouth_bottom_5", "parent": "right_mouth_bottom_4", "tail": "", "display": "口", "axis": None, "mp_index": "84"},
    "mouth_bottom": {"mmd": "mouth_bottom", "parent": "head", "tail": "", "display": "口", "axis": None, "mp_index": "17"},
    "left_mouth_bottom_1": {"mmd": "left_mouth_bottom_1", "parent": "head", "tail": "", "display": "口", "axis": None, "mp_index": "291"},
    "left_mouth_bottom_2": {"mmd": "left_mouth_bottom_2", "parent": "left_mouth_bottom_1", "tail": "", "display": "口", "axis": None, "mp_index": "375"},
    "left_mouth_bottom_3": {"mmd": "left_mouth_bottom_3", "parent": "left_mouth_bottom_2", "tail": "", "display": "口", "axis": None, "mp_index": "321"},
    "left_mouth_bottom_4": {"mmd": "left_mouth_bottom_4", "parent": "left_mouth_bottom_3", "tail": "", "display": "口", "axis": None, "mp_index": "405"},
    "left_mouth_bottom_5": {"mmd": "left_mouth_bottom_5", "parent": "left_mouth_bottom_4", "tail": "", "display": "口", "axis": None, "mp_index": "314"},
}

MP_VMD_CONNECTIONS = {
    'mp_pelvis': {'direction': ('mp_pelvis', 'mp_pelvis2'), 'up': ('mp_left_hip', 'mp_right_hip'), 'cancel': []},
    'mp_spine1': {'direction': ('mp_spine1', 'mp_neck'), 'up': ('mp_left_shoulder', 'mp_right_shoulder'), 'cancel': []},
    # 'mp_neck': {'direction': ('mp_neck', 'mp_head'), 'up': ('mp_left_shoulder', 'mp_right_shoulder'), 'cancel': ['mp_spine1']},
    # 'mp_head': {'direction': ('mp_head', 'mp_head_tail'), 'up': ('mp_left_eye', 'mp_right_eye'), 'cancel': ['mp_spine1', 'mp_neck']},

    # 'mp_left_collar': {'direction': ('mp_left_collar', 'mp_left_shoulder'), 'up': ('mp_spine1', 'mp_neck'), 'cross': ('mp_left_shoulder', 'mp_right_shoulder'), 'cancel': ['mp_spine1']},
    # 'mp_left_shoulder': {'direction': ('mp_left_shoulder', 'mp_left_elbow'), 'up': ('mp_left_collar', 'mp_left_shoulder'), 'cancel': ['mp_spine1', 'mp_left_collar']},
    # 'mp_left_elbow': {'direction': ('mp_left_elbow', 'mp_body_left_wrist'), 'up': ('mp_left_shoulder', 'mp_left_elbow'), 'cancel': ['mp_spine1', 'mp_left_collar', 'mp_left_shoulder']},
    'mp_left_hip': {'direction': ('mp_left_hip', 'mp_left_knee'), 'up': ('mp_left_hip', 'mp_right_hip'), 'cancel': ['mp_pelvis']},
    # 'mp_left_knee': {'direction': ('mp_left_knee', 'mp_left_ankle'), 'up': ('mp_left_hip', 'mp_right_hip'), 'cancel': ['mp_pelvis', 'mp_left_hip']},
    # 'mp_left_ankle': {'direction': ('mp_left_ankle', 'mp_left_foot'), 'up': ('mp_left_hip', 'mp_right_hip'), 'cancel': ['mp_pelvis', 'mp_left_hip', 'mp_left_knee']},

    # 'mp_right_collar': {'direction': ('mp_right_collar', 'mp_right_shoulder'), 'up': ('mp_spine1', 'mp_neck'), 'cross': ('mp_right_shoulder', 'mp_left_shoulder'), 'cancel': ['mp_spine1']},
    # 'mp_right_shoulder': {'direction': ('mp_right_shoulder', 'mp_right_elbow'), 'up': ('mp_right_collar', 'mp_right_shoulder'), 'cancel': ['mp_spine1', 'mp_right_collar']},
    # 'mp_right_elbow': {'direction': ('mp_right_elbow', 'mp_body_right_wrist'), 'up': ('mp_right_shoulder', 'mp_right_elbow'), 'cancel': ['mp_spine1', 'mp_right_collar', 'mp_right_shoulder']},
    'mp_right_hip': {'direction': ('mp_right_hip', 'mp_right_knee'), 'up': ('mp_right_hip', 'mp_left_hip'), 'cancel': ['mp_pelvis']},
    # 'mp_right_knee': {'direction': ('mp_right_knee', 'mp_right_ankle'), 'up': ('mp_right_hip', 'mp_left_hip'), 'cancel': ['mp_pelvis', 'mp_right_hip']},
    # 'mp_right_ankle': {'direction': ('mp_right_ankle', 'mp_right_foot'), 'up': ('mp_right_hip', 'mp_left_hip'), 'cancel': ['mp_pelvis', 'mp_right_hip', 'mp_right_knee']},
}


VMD_CONNECTIONS = {
    'pelvis': {'direction': ('pelvis', 'pelvis2'), 'up': ('left_hip', 'right_hip'), 'cancel': []},
    'spine1': {'direction': ('spine1', 'spine2'), 'up': ('left_shoulder', 'right_shoulder'), 'cancel': []},
    'spine2': {'direction': ('spine2', 'neck'), 'up': ('left_shoulder', 'right_shoulder'), 'cancel': ['spine1']},
    'neck': {'direction': ('neck', 'head'), 'up': ('left_shoulder', 'right_shoulder'), 'cancel': ['spine1', 'spine2']},
    'head': {'direction': ('head', 'head_tail'), 'up': ('left_ear', 'right_ear'), 'cancel': ['spine1', 'spine2', 'neck']},

    'left_collar': {'direction': ('left_collar', 'left_shoulder'), 'up': ('spine2', 'neck'), 'cross': ('left_shoulder', 'right_shoulder'), 'cancel': ['spine1', 'spine2']},
    'left_shoulder': {'direction': ('left_shoulder', 'left_elbow'), 'up': ('left_collar', 'left_shoulder'), 'cancel': ['spine1', 'spine2', 'left_collar']},
    'left_elbow': {'direction': ('left_elbow', 'left_wrist'), 'up': ('left_shoulder', 'left_elbow'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder']},
    'left_hip': {'direction': ('left_hip', 'left_knee'), 'up': ('left_hip', 'right_hip'), 'cancel': ['pelvis']},
    'left_knee': {'direction': ('left_knee', 'left_ankle'), 'up': ('left_hip', 'right_hip'), 'cancel': ['pelvis', 'left_hip']},
    'left_ankle': {'direction': ('left_ankle', 'left_foot'), 'up': ('left_hip', 'right_hip'), 'cancel': ['pelvis', 'left_hip', 'left_knee']},

    'right_collar': {'direction': ('right_collar', 'right_shoulder'), 'up': ('spine2', 'neck'), 'cross': ('right_shoulder', 'left_shoulder'), 'cancel': ['spine1', 'spine2']},
    'right_shoulder': {'direction': ('right_shoulder', 'right_elbow'), 'up': ('right_collar', 'right_shoulder'), 'cancel': ['spine1', 'spine2', 'right_collar']},
    'right_elbow': {'direction': ('right_elbow', 'right_wrist'), 'up': ('right_shoulder', 'right_elbow'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder']},
    'right_hip': {'direction': ('right_hip', 'right_knee'), 'up': ('right_hip', 'left_hip'), 'cancel': ['pelvis']},
    'right_knee': {'direction': ('right_knee', 'right_ankle'), 'up': ('right_hip', 'left_hip'), 'cancel': ['pelvis', 'right_hip']},
    'right_ankle': {'direction': ('right_ankle', 'right_foot'), 'up': ('right_hip', 'left_hip'), 'cancel': ['pelvis', 'right_hip', 'right_knee']},

    'left_wrist': {'direction': ('mp_left_wrist', 'mp_left_middle1'), 'up': ('mp_left_index1', 'mp_left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow']},
    # 'mp_left_thumb1': {'direction': ('mp_left_thumb1', 'mp_left_thumb2'), 'up': ('mp_left_index1', 'mp_left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist']},
    'mp_left_thumb2': {'direction': ('mp_left_thumb2', 'mp_left_thumb3'), 'up': ('mp_left_index1', 'mp_left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist', 'mp_left_thumb1']},
    'mp_left_thumb3': {'direction': ('mp_left_thumb3', 'mp_left_thumb'), 'up': ('mp_left_index1', 'mp_left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist', 'mp_left_thumb1', 'mp_left_thumb2']},
    'mp_left_index1': {'direction': ('mp_left_index1', 'mp_left_index2'), 'up': ('mp_left_index1', 'mp_left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist']},
    'mp_left_index2': {'direction': ('mp_left_index2', 'mp_left_index3'), 'up': ('mp_left_index1', 'mp_left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist', 'mp_left_index1']},
    'mp_left_index3': {'direction': ('mp_left_index3', 'mp_left_index'), 'up': ('mp_left_index1', 'mp_left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist', 'mp_left_index1', 'mp_left_index2']},
    'mp_left_middle1': {'direction': ('mp_left_middle1', 'mp_left_middle2'), 'up': ('mp_left_index1', 'mp_left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist']},
    'mp_left_middle2': {'direction': ('mp_left_middle2', 'mp_left_middle3'), 'up': ('mp_left_index1', 'mp_left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist', 'mp_left_middle1']},
    'mp_left_middle3': {'direction': ('mp_left_middle3', 'mp_left_middle'), 'up': ('mp_left_index1', 'mp_left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist', 'mp_left_middle1', 'mp_left_middle2']},
    'mp_left_ring1': {'direction': ('mp_left_ring1', 'mp_left_ring2'), 'up': ('mp_left_index1', 'mp_left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist']},
    'mp_left_ring2': {'direction': ('mp_left_ring2', 'mp_left_ring3'), 'up': ('mp_left_index1', 'mp_left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist', 'mp_left_ring1']},
    'mp_left_ring3': {'direction': ('mp_left_ring3', 'mp_left_ring'), 'up': ('mp_left_index1', 'mp_left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist', 'mp_left_ring1', 'mp_left_ring2']},
    'mp_left_pinky1': {'direction': ('mp_left_pinky1', 'mp_left_pinky2'), 'up': ('mp_left_index1', 'mp_left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist']},
    'mp_left_pinky2': {'direction': ('mp_left_pinky2', 'mp_left_pinky3'), 'up': ('mp_left_index1', 'mp_left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist', 'mp_left_pinky1']},
    'mp_left_pinky3': {'direction': ('mp_left_pinky3', 'mp_left_pinky'), 'up': ('mp_left_index1', 'mp_left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist', 'mp_left_pinky1', 'mp_left_pinky2']},

    'right_wrist': {'direction': ('mp_right_wrist', 'mp_right_middle1'), 'up': ('mp_right_index1', 'mp_right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow']},
    # 'mp_right_thumb1': {'direction': ('mp_right_thumb1', 'mp_right_thumb2'), 'up': ('mp_right_index1', 'mp_right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist']},
    'mp_right_thumb2': {'direction': ('mp_right_thumb2', 'mp_right_thumb3'), 'up': ('mp_right_index1', 'mp_right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist', 'mp_right_thumb1']},
    'mp_right_thumb3': {'direction': ('mp_right_thumb3', 'mp_right_thumb'), 'up': ('mp_right_index1', 'mp_right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist', 'mp_right_thumb1', 'mp_right_thumb2']},
    'mp_right_index1': {'direction': ('mp_right_index1', 'mp_right_index2'), 'up': ('mp_right_index1', 'mp_right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist']},
    'mp_right_index2': {'direction': ('mp_right_index2', 'mp_right_index3'), 'up': ('mp_right_index1', 'mp_right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist', 'mp_right_index1']},
    'mp_right_index3': {'direction': ('mp_right_index3', 'mp_right_index'), 'up': ('mp_right_index1', 'mp_right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist', 'mp_right_index1', 'mp_right_index2']},
    'mp_right_middle1': {'direction': ('mp_right_middle1', 'mp_right_middle2'), 'up': ('mp_right_index1', 'mp_right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist']},
    'mp_right_middle2': {'direction': ('mp_right_middle2', 'mp_right_middle3'), 'up': ('mp_right_index1', 'mp_right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist', 'mp_right_middle1']},
    'mp_right_middle3': {'direction': ('mp_right_middle3', 'mp_right_middle'), 'up': ('mp_right_index1', 'mp_right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist', 'mp_right_middle1', 'mp_right_middle2']},
    'mp_right_ring1': {'direction': ('mp_right_ring1', 'mp_right_ring2'), 'up': ('mp_right_index1', 'mp_right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist']},
    'mp_right_ring2': {'direction': ('mp_right_ring2', 'mp_right_ring3'), 'up': ('mp_right_index1', 'mp_right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist', 'mp_right_ring1']},
    'mp_right_ring3': {'direction': ('mp_right_ring3', 'mp_right_ring'), 'up': ('mp_right_index1', 'mp_right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist', 'mp_right_ring1', 'mp_right_ring2']},
    'mp_right_pinky1': {'direction': ('mp_right_pinky1', 'mp_right_pinky2'), 'up': ('mp_right_index1', 'mp_right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist']},
    'mp_right_pinky2': {'direction': ('mp_right_pinky2', 'mp_right_pinky3'), 'up': ('mp_right_index1', 'mp_right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist', 'mp_right_pinky1']},
    'mp_right_pinky3': {'direction': ('mp_right_pinky3', 'mp_right_pinky'), 'up': ('mp_right_index1', 'mp_right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist', 'mp_right_pinky1', 'mp_right_pinky2']},
}

