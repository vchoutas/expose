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
        start_z = 9999999999

        pmx_writer = PmxWriter()
        vmd_writer = VmdWriter()

        for oidx, ordered_person_dir_path in enumerate(ordered_person_dir_pathes):    
            smooth_json_pathes = sorted(glob.glob(os.path.join(ordered_person_dir_path, "smooth_*.json")), key=sort_by_numeric)

            trace_mov_motion = VmdMotion()
            trace_rot_motion = VmdMotion()
            trace_miku_motion = VmdMotion()

            # KEY: 処理対象ボーン名, VALUE: vecリスト
            target_bone_vecs = {}
            fnos = []
        
            for mbname in ["全ての親", "センター", "グルーブ", "pelvis2", "head_tail", "right_wrist_tail", "left_wrist_tail"]:
                target_bone_vecs[mbname] = {}
            
            logger.info("【No.{0}】モーション結果位置計算開始", f"{oidx:02d}", decoration=MLogger.DECORATION_LINE)

            with tqdm(total=(len(smooth_json_pathes) * (len(PMX_CONNECTIONS.keys()))), desc=f'No.{oidx:02d}') as pchar:
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

                            # # 上半身（下半身と同じ位置に生成）
                            # target_bone_vecs["spine1"][fno] = np.array([target_bone_vecs["pelvis"][fnos[-1]][0], target_bone_vecs["pelvis"][fnos[-1]][1], target_bone_vecs["pelvis"][fnos[-1]][2]])

                            # 下半身先
                            target_bone_vecs["pelvis2"][fno] = np.mean([target_bone_vecs["left_hip"][fno], target_bone_vecs["right_hip"][fno]], axis=0)
                            pchar.update(1)

                            # 頭先
                            target_bone_vecs["head_tail"][fno] = np.mean([target_bone_vecs["left_eye"][fno], target_bone_vecs["right_eye"][fno]], axis=0) + ((target_bone_vecs["head"][fno] - target_bone_vecs["neck"][fno]) / 2)
                            pchar.update(1)

                            # 手首先
                            for direction in ["left", "right"]:
                                target_bone_vecs[f"{direction}_wrist_tail"][fno] = np.mean([target_bone_vecs[f"{direction}_index1"][fno], target_bone_vecs[f"{direction}_middle1"][fno], target_bone_vecs[f"{direction}_ring1"][fno], target_bone_vecs[f"{direction}_pinky1"][fno]], axis=0)
                                pchar.update(1)

                        if "depth" in frame_joints and "camera" in frame_joints:
                            logger.debug("fno: {0}, depth: {1}, camera: {2}, ratio: {3}", fno, frame_joints["depth"]["depth"], frame_joints["camera"]["scale"], frame_joints["depth"]["depth"] / frame_joints["camera"]["scale"])
                            # pelvis_vec = calc_pelvis_vec(frame_joints, fno, args)
                            target_bone_vecs["センター"][fno] = MVector3D()

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
            
            logger.info("【No.{0}】モーション(移動)計算開始", f"{oidx:02d}", decoration=MLogger.DECORATION_LINE)

            with tqdm(total=(len(PMX_CONNECTIONS.keys()) * (len(fnos) + 1)), desc=f'No.{oidx:02d}') as pchar:
                for fno in fnos:
                    joint = target_bone_vecs["センター"][fno]                    
                    center_bf = trace_mov_motion.calc_bf("センター", fno)
                    groove_bf = trace_mov_motion.calc_bf("グルーブ", fno)
                    # groove_bf.position.setY(joint.y() + trace_model.bones["グルーブ"].position.y())
                    groove_bf.position.setY(joint.y())
                    groove_bf.key = True
                    trace_mov_motion.bones[groove_bf.name][fno] = groove_bf

                    center_bf.position.setX(joint.x())
                    center_bf.key = True
                    trace_mov_motion.bones[center_bf.name][fno] = center_bf
                    pchar.update(1)

                for jidx, (jname, pconn) in enumerate(PMX_CONNECTIONS.items()):
                    # ボーン登録        
                    create_bone(trace_model, jname, pconn, target_bone_vecs, miku_model)

                    mname = pconn['mmd']
                    pmname = PMX_CONNECTIONS[pconn['parent']]['mmd'] if pconn['parent'] in PMX_CONNECTIONS else pconn['parent']
                    
                    # モーションも登録
                    for fno in fnos:
                        joint = target_bone_vecs[jname][fno]
                        parent_joint = target_bone_vecs[pconn['parent']][fno]
                        
                        trace_bf = trace_mov_motion.calc_bf(mname, fno)

                        # # parentが全ての親の場合
                        # trace_bf.position = MVector3D(joint) - trace_model.bones[mname].position
                        # parentが親ボーンの場合
                        trace_bf.position = MVector3D(joint) - MVector3D(parent_joint) \
                                            - (trace_model.bones[mname].position - trace_model.bones[pmname].position)
                        trace_bf.key = True
                        trace_mov_motion.bones[mname][fno] = trace_bf
                        pchar.update(1)

            for bidx, bone in trace_model.bones.items():
                # 表示先の設定
                if bone.tail_index and bone.tail_index in trace_model.bones:
                    bone.flag |= 0x0001
                    bone.tail_index = trace_model.bones[bone.tail_index].index
                else:
                    bone.tail_index = -1

            trace_model_path = os.path.join(motion_dir_path, f"trace_{process_datetime}_mov_model_no{oidx:02d}.pmx")
            logger.info("【No.{0}】トレース(移動)モデル生成開始【{1}】", f"{oidx:02d}", os.path.basename(trace_model_path), decoration=MLogger.DECORATION_LINE)
            trace_model.name = f"Trace結果 {oidx:02d} (移動用)"
            trace_model.english_name = f"TraceModel {oidx:02d} (Move)"
            trace_model.comment = f"Trace結果 {oidx:02d} 表示用モデル (移動用)\n足ＩＫがありません"
            pmx_writer.write(trace_model, trace_model_path)
            
            trace_mov_motion_path = os.path.join(motion_dir_path, f"trace_{process_datetime}_mov_no{oidx:02d}.vmd")
            logger.info("【No.{0}】モーション(移動)生成開始【{1}】", f"{oidx:02d}", os.path.basename(trace_mov_motion_path), decoration=MLogger.DECORATION_LINE)
            vmd_writer.write(trace_model, trace_mov_motion, trace_mov_motion_path)

            # 足ＩＫ ------------------------
            ik_show = VmdShowIkFrame()
            ik_show.fno = 0
            ik_show.show = 1

            for direction in ["左", "右"]:
                create_bone_leg_ik(trace_model, direction)
                
                ik_show.ik.append(VmdInfoIk(f'{direction}足ＩＫ', 0))
                ik_show.ik.append(VmdInfoIk(f'{direction}つま先ＩＫ', 0))

            trace_rot_motion.showiks.append(ik_show)
            trace_miku_motion.showiks.append(ik_show)

            trace_model_path = os.path.join(motion_dir_path, f"trace_{process_datetime}_rot_model_no{oidx:02d}.pmx")
            logger.info("【No.{0}】トレース(移動)モデル生成開始【{1}】", f"{oidx:02d}", os.path.basename(trace_model_path), decoration=MLogger.DECORATION_LINE)
            trace_model.name = f"Trace結果 {oidx:02d} (回転用)"
            trace_model.english_name = f"TraceModel {oidx:02d} (Rot)"
            trace_model.comment = f"Trace結果 {oidx:02d} 表示用モデル (回転用)\n足ＩＫはモーション側でOFFにしています"
            pmx_writer.write(trace_model, trace_model_path)
            
            logger.info("【No.{0}】モーション(回転)計算開始", f"{oidx:02d}", decoration=MLogger.DECORATION_LINE)

            with tqdm(total=((len(VMD_CONNECTIONS.keys()) + 2) * (len(fnos))), desc=f'No.{oidx:02d}') as pchar:
                for mname in ["センター", "グルーブ"]:
                    for fno in fnos:
                        mov_bf = trace_mov_motion.calc_bf(mname, fno)
                        now_bf = trace_rot_motion.calc_bf(mname, fno)
                        now_bf.position = mov_bf.position
                        now_bf.key = True
                        trace_rot_motion.bones[mname][fno] = now_bf

                        pchar.update(1)

                for jname, jconn in VMD_CONNECTIONS.items():
                    mname = PMX_CONNECTIONS[jname]['mmd']
                    direction_from_mname = PMX_CONNECTIONS[jconn["direction"][0]]['mmd']
                    direction_to_mname = PMX_CONNECTIONS[jconn["direction"][1]]['mmd']
                    up_from_mname = PMX_CONNECTIONS[jconn["up"][0]]['mmd']
                    up_to_mname = PMX_CONNECTIONS[jconn["up"][1]]['mmd']
                    cross_from_mname = PMX_CONNECTIONS[jconn["cross"][0]]['mmd'] if 'cross' in jconn else direction_from_mname
                    cross_to_mname = PMX_CONNECTIONS[jconn["cross"][1]]['mmd'] if 'cross' in jconn else direction_to_mname

                    # トレースモデルの初期姿勢
                    trace_direction_from_vec = trace_model.bones[direction_from_mname].position
                    trace_direction_to_vec = trace_model.bones[direction_to_mname].position
                    trace_direction = (trace_direction_to_vec - trace_direction_from_vec).normalized()

                    trace_up_from_vec = trace_model.bones[up_from_mname].position
                    trace_up_to_vec = trace_model.bones[up_to_mname].position
                    trace_up = (trace_up_to_vec - trace_up_from_vec).normalized()

                    trace_cross_from_vec = trace_model.bones[cross_from_mname].position
                    trace_cross_to_vec = trace_model.bones[cross_to_mname].position
                    trace_cross = (trace_cross_to_vec - trace_cross_from_vec).normalized()

                    trace_up_cross = MVector3D.crossProduct(trace_up, trace_cross).normalized()
                    trace_stance_qq = MQuaternion.fromDirection(trace_direction, trace_up_cross)

                    direction_from_links = trace_model.create_link_2_top_one(direction_from_mname, is_defined=False)
                    direction_to_links = trace_model.create_link_2_top_one(direction_to_mname, is_defined=False)
                    up_from_links = trace_model.create_link_2_top_one(up_from_mname, is_defined=False)
                    up_to_links = trace_model.create_link_2_top_one(up_to_mname, is_defined=False)
                    cross_from_links = trace_model.create_link_2_top_one(cross_from_mname, is_defined=False)
                    cross_to_links = trace_model.create_link_2_top_one(cross_to_mname, is_defined=False)

                    for fno in fnos:
                        now_direction_from_vec = calc_global_pos_from_mov(trace_model, direction_from_links, trace_mov_motion, fno)
                        now_direction_to_vec = calc_global_pos_from_mov(trace_model, direction_to_links, trace_mov_motion, fno)
                        now_up_from_vec = calc_global_pos_from_mov(trace_model, up_from_links, trace_mov_motion, fno)
                        now_up_to_vec = calc_global_pos_from_mov(trace_model, up_to_links, trace_mov_motion, fno)
                        now_cross_from_vec = calc_global_pos_from_mov(trace_model, cross_from_links, trace_mov_motion, fno)
                        now_cross_to_vec = calc_global_pos_from_mov(trace_model, cross_to_links, trace_mov_motion, fno)

                        # トレースモデルの回転量 ------------
                        now_direction = (now_direction_to_vec - now_direction_from_vec).normalized()
                        now_up = (now_up_to_vec - now_up_from_vec).normalized()
                        now_cross = (now_cross_to_vec - now_cross_from_vec).normalized()

                        now_up_cross = MVector3D.crossProduct(now_up, now_cross).normalized()
                        now_stance_qq = MQuaternion.fromDirection(now_direction, now_up_cross)

                        cancel_qq = MQuaternion()
                        for cancel_jname in jconn["cancel"]:
                            cancel_qq *= trace_rot_motion.calc_bf(PMX_CONNECTIONS[cancel_jname]['mmd'], fno).rotation

                        now_qq = cancel_qq.inverted() * now_stance_qq * trace_stance_qq.inverted()

                        now_bf = trace_rot_motion.calc_bf(mname, fno)
                        now_bf.rotation = now_qq
                        now_bf.key = True
                        trace_rot_motion.bones[mname][fno] = now_bf

                        pchar.update(1)

            trace_rot_motion_path = os.path.join(motion_dir_path, f"trace_{process_datetime}_rot_no{oidx:02d}.vmd")
            logger.info("【No.{0}】トレースモーション(回転)生成開始【{1}】", f"{oidx:02d}", os.path.basename(trace_rot_motion_path), decoration=MLogger.DECORATION_LINE)
            vmd_writer.write(trace_model, trace_rot_motion, trace_rot_motion_path)

            logger.info("【No.{0}】モーション(あにまさ式ミク)計算開始", f"{oidx:02d}", decoration=MLogger.DECORATION_LINE)

            with tqdm(total=(len(trace_rot_motion.bones.keys()) * len(fnos)), desc=f'No.{oidx:02d}') as pchar:
                for mname in ["センター", "グルーブ"]:
                    for fno in fnos:
                        rot_bf = trace_rot_motion.calc_bf(mname, fno)
                        miku_bf = trace_miku_motion.calc_bf(mname, fno)
                        miku_bf.position = rot_bf.position.copy()
                        miku_bf.rotation = rot_bf.rotation.copy()
                        miku_bf.key = True

                        trace_miku_motion.bones[mname][fno] = miku_bf
                        pchar.update(1)
                        continue
                    
                for jname, jconn in VMD_CONNECTIONS.items():
                    mname = PMX_CONNECTIONS[jname]['mmd']
                    parent_jnmae = PMX_CONNECTIONS[jname]['parent']

                    parent_name = "センター"
                    if mname not in ["全ての親", "センター", "グルーブ"] and parent_jnmae in PMX_CONNECTIONS:
                        parent_name = PMX_CONNECTIONS[parent_jnmae]['mmd']
                    
                    base_axis = PMX_CONNECTIONS[jname]["axis"] if jname in PMX_CONNECTIONS else None
                    parent_axis = PMX_CONNECTIONS[jname]["parent_axis"] if jname in PMX_CONNECTIONS and "parent_axis" in PMX_CONNECTIONS[jname] else PMX_CONNECTIONS[parent_jnmae]["axis"] if parent_jnmae in PMX_CONNECTIONS else None
                    trace_parent_local_x_qq = trace_model.get_local_x_qq(parent_name, parent_axis)
                    trace_target_local_x_qq = trace_model.get_local_x_qq(mname, base_axis)
                    miku_parent_local_x_qq = miku_model.get_local_x_qq(parent_name, parent_axis)
                    miku_target_local_x_qq = miku_model.get_local_x_qq(mname, base_axis)

                    parent_local_x_qq = miku_parent_local_x_qq.inverted() * trace_parent_local_x_qq
                    target_local_x_qq = miku_target_local_x_qq.inverted() * trace_target_local_x_qq

                    miku_local_x_axis = miku_model.get_local_x_axis(mname)
                    miku_local_y_axis = MVector3D.crossProduct(miku_local_x_axis, MVector3D(0, 0, 1))

                    for fno in fnos:
                        rot_bf = trace_rot_motion.calc_bf(mname, fno)
                        miku_bf = trace_miku_motion.calc_bf(mname, fno)

                        miku_bf.position = rot_bf.position.copy()
                        new_miku_qq = rot_bf.rotation.copy()
                        
                        if (len(mname) > 2 and mname[2] == "指") or mname[1:] in ["ひざ"]:
                            # 指・ひざは念のためX捩り除去
                            _, _, _, now_yz_qq = MServiceUtils.separate_local_qq(fno, mname, new_miku_qq, miku_local_x_axis)
                            new_miku_qq = now_yz_qq
                        
                        if (len(mname) > 2 and mname[2] == "指"):
                            pass
                        # elif mname[1:] in ["手首"]:
                        #     new_miku_qq = parent_local_x_qq.inverted() * new_miku_qq * target_local_x_qq
                        # elif mname[1:] in ["ひじ"]:
                        #     new_miku_qq = parent_local_x_qq.inverted() * new_miku_qq * target_local_x_qq
                        # elif mname[1:] in ["腕"]:
                        #     new_miku_qq = trace_parent_local_x_qq.inverted() * new_miku_qq * target_local_x_qq
                        elif mname[1:] in ["肩", "足"]:
                            new_miku_qq = new_miku_qq * target_local_x_qq
                        else:
                            new_miku_qq = parent_local_x_qq.inverted() * new_miku_qq * target_local_x_qq

                        if len(mname) > 2 and mname[2] == "指":
                            # 指は正方向にしか曲がらない
                            _, _, _, now_yz_qq = MServiceUtils.separate_local_qq(fno, mname, new_miku_qq, miku_local_x_axis)
                            new_miku_qq = MQuaternion.fromAxisAndAngle((MVector3D(0, 0, -1) if mname[0] == "左" else MVector3D(0, 0, 1)), now_yz_qq.toDegree())
                        elif mname[1:] in ["ひざ"]:
                            # ひざは足IK用にYのみ
                            _, _, _, now_yz_qq = MServiceUtils.separate_local_qq(fno, mname, new_miku_qq, miku_local_x_axis)
                            new_miku_qq = MQuaternion.fromAxisAndAngle(miku_local_y_axis, now_yz_qq.toDegree())

                        miku_bf.rotation = new_miku_qq
                        miku_bf.key = True

                        trace_miku_motion.bones[mname][fno] = miku_bf
                        pchar.update(1)

            trace_miku_motion_path = os.path.join(motion_dir_path, f"trace_{process_datetime}_miku_no{oidx:02d}.vmd")
            logger.info("【No.{0}】トレースモーション(あにまさ式ミク)生成開始【{1}】", f"{oidx:02d}", os.path.basename(trace_miku_motion_path), decoration=MLogger.DECORATION_LINE)
            vmd_writer.write(miku_model, trace_miku_motion, trace_miku_motion_path)

            logger.info("【No.{0}】モーション(あにまさ式ミク)外れ値削除開始", f"{oidx:02d}", decoration=MLogger.DECORATION_LINE)

            loop_cnt = 2
            with tqdm(total=(len(trace_miku_motion.bones.keys()) * len(fnos) * loop_cnt), desc=f'No.{oidx:02d}') as pchar:
                for n in range(loop_cnt):
                    bone_names = list(trace_miku_motion.bones.keys())
                    for mname in bone_names:
                        if mname in ["センター", "グルーブ"]:
                            for prev_fno, fno in zip(fnos[:-1:2], fnos[1::2]):
                                prev_bf = trace_miku_motion.calc_bf(mname, prev_fno)
                                now_bf = trace_miku_motion.calc_bf(mname, fno)

                                distance = prev_bf.position.distanceToPoint(now_bf.position)
                                if 0.005 + (n * 0.001) > distance:
                                    # 離れすぎてるのは削除(回数を重ねるほどに変化量の検知を鈍くする)
                                    if fno in trace_miku_motion.bones[mname]:
                                        del trace_miku_motion.bones[mname][fno]
                                pchar.update(2)
                            continue
                        
                        for prev_fno, fno in zip(fnos[:-1:2], fnos[1::2]):
                            prev_bf = trace_miku_motion.calc_bf(mname, prev_fno)
                            now_bf = trace_miku_motion.calc_bf(mname, fno)

                            dot = MQuaternion.dotProduct(prev_bf.rotation, now_bf.rotation)
                            if 0.8 - (n * 0.1) > dot:
                                # 離れすぎてるのは削除(回数を重ねるほどに変化量の検知を鈍くする)
                                if fno in trace_miku_motion.bones[mname]:
                                    del trace_miku_motion.bones[mname][fno]
                            pchar.update(2)

            logger.info("【No.{0}】モーション(あにまさ式ミク)スムージング開始", f"{oidx:02d}", decoration=MLogger.DECORATION_LINE)

            loop_cnt = 1
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
                                # まず前後の中間をそのまま求める
                                filterd_qq = MQuaternion.slerp(prev_bf.rotation, next_bf.rotation, (fno - prev_fno) / (next_fno - prev_fno))
                                # 現在の回転にも少し近づける
                                bf.rotation = MQuaternion.slerp(filterd_qq, bf.rotation, 0.6)
                                bf.key = True
                                trace_miku_motion.bones[mname][fno] = bf

                            pchar.update(1)

            trace_miku_motion_path = os.path.join(motion_dir_path, f"trace_{process_datetime}_miku_smooth_no{oidx:02d}.vmd")
            logger.info("【No.{0}】スムージングトレースモーション(あにまさ式ミク)生成開始【{1}】", f"{oidx:02d}", os.path.basename(trace_miku_motion_path), decoration=MLogger.DECORATION_LINE)
            vmd_writer.write(miku_model, trace_miku_motion, trace_miku_motion_path)

            # logger.info("【No.{0}】モーション(回転)足首計算開始", f"{oidx:02d}", decoration=MLogger.DECORATION_LINE)

            # with tqdm(total=(2 * (len(fnos))), desc=f'No.{oidx:02d}') as pchar:
            #     for direction in ["左", "右"]:
            #         trace_big_toe_links = trace_model.create_link_2_top_one(f'{direction}足親指', is_defined=False)
            #         trace_small_toe_links = trace_model.create_link_2_top_one(f'{direction}足小指', is_defined=False)
            #         trace_heel_links = trace_model.create_link_2_top_one(f'{direction}かかと', is_defined=False)

            #         _, trace_big_toe_mats = MServiceUtils.calc_global_pos(trace_model, trace_big_toe_links, VmdMotion(), fno, return_matrix=True)
            #         _, trace_small_toe_mats = MServiceUtils.calc_global_pos(trace_model, trace_small_toe_links, VmdMotion(), fno, return_matrix=True)
            #         _, trace_heel_mats = MServiceUtils.calc_global_pos(trace_model, trace_heel_links, VmdMotion(), fno, return_matrix=True)

            #         for fno in fnos:
            #             now_trace_big_toe_global_vec = trace_big_toe_mats[trace_big_toe_links.last_name()] * trace_mov_motion.calc_bf(trace_big_toe_links.last_name(), fno).position
            #             now_trace_small_toe_global_vec = trace_small_toe_mats[trace_small_toe_links.last_name()] * trace_mov_motion.calc_bf(trace_small_toe_links.last_name(), fno).position
            #             now_trace_big_toe_relative_vec = trace_heel_mats[trace_heel_links.last_name()].inverted() * now_trace_big_toe_global_vec
            #             now_trace_small_toe_relative_vec = trace_heel_mats[trace_heel_links.last_name()].inverted() * now_trace_small_toe_global_vec

            #             now_trace_big_slope = abs(MVector3D.dotProduct(MVector3D(now_trace_big_toe_relative_vec.x(), 0, now_trace_big_toe_relative_vec.z()).normalized(), now_trace_big_toe_relative_vec.normalized()))
            #             now_trace_small_slope = abs(MVector3D.dotProduct(MVector3D(now_trace_small_toe_relative_vec.x(), 0, now_trace_small_toe_relative_vec.z()).normalized(), now_trace_small_toe_relative_vec.normalized()))

            #             if (np.average([now_trace_big_slope, now_trace_small_slope]) > 0.92):
            #                 # 大体水平の場合、足首の角度を初期化する
            #                 ankle_bf = trace_rot_motion.calc_bf(f'{direction}足首', fno)
            #                 ankle_bf.rotation = MQuaternion()
            #                 trace_rot_motion.bones[ankle_bf.name][fno] = ankle_bf
            #             else:
            #                 logger.debug("fno: {0}, now_trace_big_slope: {1}, now_trace_small_slope: {2}", fno, now_trace_big_slope, now_trace_small_slope)

            #             pchar.update(1)

                    # # ミクモデルの初期姿勢
                    # miku_direction_from_vec = miku_model.bones[direction_from_mname].position
                    # miku_direction_to_vec = miku_model.bones[direction_to_mname].position
                    # miku_direction = (miku_direction_to_vec - miku_direction_from_vec)

                    # miku_up_from_vec = miku_model.bones[up_from_mname].position
                    # miku_up_to_vec = miku_model.bones[up_to_mname].position
                    # miku_up = (miku_up_to_vec - miku_up_from_vec)

                    # miku_cross = MVector3D.crossProduct(miku_direction.normalized(), miku_up.normalized()).normalized()
                    # miku_stance_qq = MQuaternion.fromDirection(miku_direction.normalized(), miku_cross.normalized())

                    # miku_trace_stance_diff_qq = MQuaternion.rotationTo(miku_direction.normalized(), trace_direction.normalized())

                    # miku_direction_from_links = miku_model.create_link_2_top_one(direction_from_mname, is_defined=False)
                    # miku_direction_to_links = miku_model.create_link_2_top_one(direction_to_mname, is_defined=False)
                    # miku_up_from_links = miku_model.create_link_2_top_one(up_from_mname, is_defined=False)
                    # miku_up_to_links = miku_model.create_link_2_top_one(up_to_mname, is_defined=False)

                    # _, miku_direction_from_mats = MServiceUtils.calc_global_pos(miku_model, miku_direction_from_links, VmdMotion(), fno, return_matrix=True)
                    # _, miku_direction_to_mats = MServiceUtils.calc_global_pos(miku_model, miku_direction_to_links, VmdMotion(), fno, return_matrix=True)
                    # _, miku_up_from_mats = MServiceUtils.calc_global_pos(miku_model, miku_up_from_links, VmdMotion(), fno, return_matrix=True)
                    # _, miku_up_to_mats = MServiceUtils.calc_global_pos(miku_model, miku_up_to_links, VmdMotion(), fno, return_matrix=True)

                        # # ミクの回転量 ---------------------
                        # now_miku_direction_from_vec = miku_direction_from_mats[direction_from_mname] * trace_mov_motion.calc_bf(direction_from_mname, fno).position
                        # now_miku_direction_to_vec = miku_direction_to_mats[direction_to_mname] * trace_mov_motion.calc_bf(direction_to_mname, fno).position
                        # now_miku_up_from_vec = miku_up_from_mats[up_from_mname] * trace_mov_motion.calc_bf(up_from_mname, fno).position
                        # now_miku_up_to_vec = miku_up_to_mats[up_to_mname] * trace_mov_motion.calc_bf(up_to_mname, fno).position

                        # now_miku_direction = MVector3D(now_miku_direction_to_vec - now_miku_direction_from_vec).normalized()
                        # now_miku_up = MVector3D(now_miku_up_to_vec - now_miku_up_from_vec).normalized()
                        # now_miku_cross = MVector3D.crossProduct(now_miku_direction, now_miku_up).normalized()
                        # now_miku_direction_qq = MQuaternion.fromDirection(now_miku_direction, now_miku_cross)

                        # miku_cancel_qq = MQuaternion()
                        # for miku_cancel_jname in jconn["cancel"]:
                        #     miku_cancel_qq *= miku_motion.calc_bf(PMX_CONNECTIONS[miku_cancel_jname]['mmd'], fno).rotation

                        # now_miku_qq = now_qq * 

                        # now_miku_bf = miku_motion.calc_bf(mname, fno)
                        # now_miku_bf.rotation = now_miku_qq
                        # now_miku_bf.key = True
                        # miku_motion.bones[mname][fno] = now_miku_bf

                        # pchar.update(1)

            # logger.info("【No.{0}】ミクモーション結果回転計算開始", f"{oidx:02d}", decoration=MLogger.DECORATION_LINE)

            # with tqdm(total=(len(VMD_CONNECTIONS.keys()) * (len(fnos))), desc=f'No.{oidx:02d}') as pchar:
            #     for jname, jconn in VMD_CONNECTIONS.items():
            #         direction_from_mname = PMX_CONNECTIONS[jconn["direction"][0]]['mmd']
            #         direction_to_mname = PMX_CONNECTIONS[jconn["direction"][1]]['mmd']
            #         up_from_mname = PMX_CONNECTIONS[jconn["up"][0]]['mmd']
            #         up_to_mname = PMX_CONNECTIONS[jconn["up"][1]]['mmd']

            #         # トレースモデルの初期姿勢
            #         trace_direction_from_vec = trace_model.bones[direction_from_mname].position
            #         trace_direction_to_vec = trace_model.bones[direction_to_mname].position
            #         trace_direction = (trace_direction_to_vec - trace_direction_from_vec)

            #         trace_up_from_vec = trace_model.bones[up_from_mname].position
            #         trace_up_to_vec = trace_model.bones[up_to_mname].position
            #         trace_up = (trace_up_to_vec - trace_up_from_vec)

            #         trace_cross = MVector3D.crossProduct(trace_direction.normalized(), trace_up.normalized()).normalized()
            #         trace_stance_qq = MQuaternion.fromDirection(trace_direction.normalized(), trace_cross.normalized())

            #         # ミクモデルの初期姿勢
            #         miku_direction_from_vec = miku_model.bones[direction_from_mname].position
            #         miku_direction_to_vec = miku_model.bones[direction_to_mname].position
            #         miku_direction = (miku_direction_to_vec - miku_direction_from_vec)

            #         miku_up_from_vec = miku_model.bones[up_from_mname].position
            #         miku_up_to_vec = miku_model.bones[up_to_mname].position
            #         miku_up = (miku_up_to_vec - miku_up_from_vec)

            #         miku_cross = MVector3D.crossProduct(miku_direction.normalized(), miku_up.normalized()).normalized()
            #         miku_stance_qq = MQuaternion.fromDirection(miku_direction.normalized(), miku_cross.normalized())

            #         mname = PMX_CONNECTIONS[jname]['mmd']

            #         trace_direction_from_links = trace_model.create_link_2_top_one(direction_from_mname, is_defined=False)
            #         trace_direction_to_links = trace_model.create_link_2_top_one(direction_to_mname, is_defined=False)
            #         trace_up_from_links = trace_model.create_link_2_top_one(up_from_mname, is_defined=False)
            #         trace_up_to_links = trace_model.create_link_2_top_one(up_to_mname, is_defined=False)

            #         _, trace_direction_from_mats = MServiceUtils.calc_global_pos(trace_model, trace_direction_from_links, VmdMotion(), fno, return_matrix=True)
            #         _, trace_direction_to_mats = MServiceUtils.calc_global_pos(trace_model, trace_direction_to_links, VmdMotion(), fno, return_matrix=True)
            #         _, trace_up_from_mats = MServiceUtils.calc_global_pos(trace_model, trace_up_from_links, VmdMotion(), fno, return_matrix=True)
            #         _, trace_up_to_mats = MServiceUtils.calc_global_pos(trace_model, trace_up_to_links, VmdMotion(), fno, return_matrix=True)

            #         miku_direction_from_links = miku_model.create_link_2_top_one(direction_from_mname, is_defined=False)
            #         miku_direction_to_links = miku_model.create_link_2_top_one(direction_to_mname, is_defined=False)
            #         miku_up_from_links = miku_model.create_link_2_top_one(up_from_mname, is_defined=False)
            #         miku_up_to_links = miku_model.create_link_2_top_one(up_to_mname, is_defined=False)

            #         _, miku_direction_from_mats = MServiceUtils.calc_global_pos(miku_model, miku_direction_from_links, VmdMotion(), fno, return_matrix=True)
            #         _, miku_direction_to_mats = MServiceUtils.calc_global_pos(miku_model, miku_direction_to_links, VmdMotion(), fno, return_matrix=True)
            #         _, miku_up_from_mats = MServiceUtils.calc_global_pos(miku_model, miku_up_from_links, VmdMotion(), fno, return_matrix=True)
            #         _, miku_up_to_mats = MServiceUtils.calc_global_pos(miku_model, miku_up_to_links, VmdMotion(), fno, return_matrix=True)

            #         for fno in fnos:
            #             # ミクのグローバル位置
            #             now_miku_direction_from_vec = miku_direction_from_mats[direction_from_mname] * trace_mov_motion.calc_bf(direction_from_mname, fno).position
            #             now_miku_direction_to_vec = miku_direction_to_mats[direction_to_mname] * trace_mov_motion.calc_bf(direction_to_mname, fno).position
            #             now_miku_up_from_vec = miku_up_from_mats[up_from_mname] * trace_mov_motion.calc_bf(up_from_mname, fno).position
            #             now_miku_up_to_vec = miku_up_to_mats[up_to_mname] * trace_mov_motion.calc_bf(up_to_mname, fno).position
                        




            #             # ミク用モーションの生成
                    
            #             miku_cancel_qq = MQuaternion()
            #             for cancel_jname in jconn["cancel"]:
            #                 miku_cancel_qq *= miku_motion.calc_bf(PMX_CONNECTIONS[cancel_jname]['mmd'], fno).rotation

            #             miku_qq = miku_cancel_qq.inverted() * now_direction_qq * miku_stance_qq.inverted()

            #             miku_bf = miku_motion.calc_bf(mname, fno)
            #             miku_bf.rotation = miku_qq
            #             miku_bf.key = True
            #             miku_motion.bones[mname][fno] = miku_bf

            #             pchar.update(1)

                    # # ミクモデルの初期姿勢
                    # miku_direction_from_vec = miku_model.bones[direction_from_mname].position
                    # miku_direction_to_vec = miku_model.bones[direction_to_mname].position
                    # miku_direction = (miku_direction_to_vec - miku_direction_from_vec).normalized()

                    # miku_up_from_vec = miku_model.bones[up_from_mname].position
                    # miku_up_to_vec = miku_model.bones[up_to_mname].position
                    # miku_up = (miku_up_to_vec - miku_up_from_vec).normalized()

                    # miku_cross = MVector3D.crossProduct(miku_direction, miku_up).normalized()
                    # miku_stance_qq = MQuaternion.fromDirection(miku_direction, miku_cross)

                    # if PMX_CONNECTIONS[jconn["direction"][0]]['parent'] in PMX_CONNECTIONS:
                    #     direction_from_parent_name = PMX_CONNECTIONS[PMX_CONNECTIONS[jconn["direction"][0]]['parent']]['mmd']
                    #     direction_from_parent_axis = PMX_CONNECTIONS[PMX_CONNECTIONS[jconn["direction"][0]]['parent']]['axis']

                    #     trace_parent_stance = MQuaternion.rotationTo(direction_from_parent_axis, trace_model.get_local_x_axis(direction_from_parent_name))
                    #     miku_parent_stance = MQuaternion.rotationTo(direction_from_parent_axis, miku_model.get_local_x_axis(direction_from_parent_name))
                    #     parent_stance_qq = miku_parent_stance.inverted() * trace_parent_stance
                    # else:
                    #     parent_stance_qq = MQuaternion()

                    # # スタンスの差
                    # trace_from_stance = MQuaternion.rotationTo(PMX_CONNECTIONS[jconn["direction"][0]]['axis'], trace_model.get_local_x_axis(direction_from_mname))
                    # miku_from_stance = MQuaternion.rotationTo(PMX_CONNECTIONS[jconn["direction"][0]]['axis'], miku_model.get_local_x_axis(direction_from_mname))
                    # from_stance_qq = miku_from_stance.inverted() * trace_from_stance

                    # trace_direction_stance_qq = MQuaternion.rotationTo(PMX_CONNECTIONS[jconn["direction"][0]]['axis'], trace_direction)
                    # miku_direction_stance_qq = MQuaternion.rotationTo(PMX_CONNECTIONS[jconn["direction"][0]]['axis'], miku_direction)
                    # direction_stance_qq = miku_direction_stance_qq.inverted() * trace_direction_stance_qq

                    # trace_direction_from_links = trace_model.create_link_2_top_one(direction_from_mname, is_defined=False)
                    # trace_direction_to_links = trace_model.create_link_2_top_one(direction_to_mname, is_defined=False)
                    # trace_up_from_links = trace_model.create_link_2_top_one(up_from_mname, is_defined=False)
                    # trace_up_to_links = trace_model.create_link_2_top_one(up_to_mname, is_defined=False)

                    # _, trace_direction_from_mats = MServiceUtils.calc_global_pos(trace_model, trace_direction_from_links, VmdMotion(), fno, return_matrix=True)
                    # _, trace_direction_to_mats = MServiceUtils.calc_global_pos(trace_model, trace_direction_to_links, VmdMotion(), fno, return_matrix=True)
                    # _, trace_up_from_mats = MServiceUtils.calc_global_pos(trace_model, trace_up_from_links, VmdMotion(), fno, return_matrix=True)
                    # _, trace_up_to_mats = MServiceUtils.calc_global_pos(trace_model, trace_up_to_links, VmdMotion(), fno, return_matrix=True)

                    # miku_direction_from_links = miku_model.create_link_2_top_one(direction_from_mname, is_defined=False)
                    # miku_direction_to_links = miku_model.create_link_2_top_one(direction_to_mname, is_defined=False)
                    # miku_up_from_links = miku_model.create_link_2_top_one(up_from_mname, is_defined=False)
                    # miku_up_to_links = miku_model.create_link_2_top_one(up_to_mname, is_defined=False)

                    # _, miku_direction_from_mats = MServiceUtils.calc_global_pos(miku_model, miku_direction_from_links, VmdMotion(), fno, return_matrix=True)
                    # _, miku_direction_to_mats = MServiceUtils.calc_global_pos(miku_model, miku_direction_to_links, VmdMotion(), fno, return_matrix=True)
                    # _, miku_up_from_mats = MServiceUtils.calc_global_pos(miku_model, miku_up_from_links, VmdMotion(), fno, return_matrix=True)
                    # _, miku_up_to_mats = MServiceUtils.calc_global_pos(miku_model, miku_up_to_links, VmdMotion(), fno, return_matrix=True)

                        # miku_direction_from_vec = miku_direction_from_mats[direction_from_mname] * trace_mov_motion.calc_bf(direction_from_mname, fno).position
                        # miku_direction_to_vec = miku_direction_to_mats[direction_to_mname] * trace_mov_motion.calc_bf(direction_to_mname, fno).position
                        # miku_up_from_vec = miku_up_from_mats[up_from_mname] * trace_mov_motion.calc_bf(up_from_mname, fno).position
                        # miku_up_to_vec = miku_up_to_mats[up_to_mname] * trace_mov_motion.calc_bf(up_to_mname, fno).position

                        # miku_direction = MVector3D(miku_direction_to_vec - miku_direction_from_vec).normalized()
                        # miku_up = MVector3D(miku_up_to_vec - miku_up_from_vec).normalized()
                        # miku_cross = MVector3D.crossProduct(miku_direction, miku_up).normalized()
                        # miku_direction_qq = MQuaternion.fromDirection(miku_direction, miku_cross)

                        # miku_cancel_qq = MQuaternion()
                        # for cancel_jname in jconn["cancel"]:
                        #     miku_cancel_qq *= motion.calc_bf(PMX_CONNECTIONS[cancel_jname]['mmd'], fno).rotation

                    # miku_direction_from_links = miku_model.create_link_2_top_one(direction_from_mname, is_defined=False)
                    # miku_direction_to_links = miku_model.create_link_2_top_one(direction_to_mname, is_defined=False)
                    # miku_up_from_links = miku_model.create_link_2_top_one(up_from_mname, is_defined=False)
                    # miku_up_to_links = miku_model.create_link_2_top_one(up_to_mname, is_defined=False)

                    # _, miku_direction_from_mats = MServiceUtils.calc_global_pos(model, miku_direction_from_links, VmdMotion(), fno, return_matrix=True)
                    # _, miku_direction_to_mats = MServiceUtils.calc_global_pos(model, miku_direction_to_links, VmdMotion(), fno, return_matrix=True)
                    # _, miku_up_from_mats = MServiceUtils.calc_global_pos(model, miku_up_from_links, VmdMotion(), fno, return_matrix=True)
                    # _, miku_up_to_mats = MServiceUtils.calc_global_pos(model, miku_up_to_links, VmdMotion(), fno, return_matrix=True)


            # for jname, jconn in VMD_CONNECTIONS.items():
            #     direction_from_mname = PMX_CONNECTIONS[jconn["direction"][0]]['mmd']
            #     direction_to_mname = PMX_CONNECTIONS[jconn["direction"][1]]['mmd']
            #     up_from_mname = PMX_CONNECTIONS[jconn["up"][0]]['mmd']
            #     up_to_mname = PMX_CONNECTIONS[jconn["up"][1]]['mmd']

            #     # トレースモデルの初期姿勢
            #     trace_direction_from_vec = trace_model.bones[direction_from_mname].position
            #     trace_direction_to_vec = trace_model.bones[direction_to_mname].position
            #     trace_direction = (trace_direction_to_vec - trace_direction_from_vec)

            #     trace_up_from_vec = trace_model.bones[up_from_mname].position
            #     trace_up_to_vec = trace_model.bones[up_to_mname].position
            #     trace_up = (trace_up_to_vec - trace_up_from_vec)

            #     trace_cross = MVector3D.crossProduct(trace_direction.normalized(), trace_up.normalized()).normalized()
            #     trace_stance_qq = MQuaternion.fromDirection(trace_direction.normalized(), trace_cross.normalized())

            #     # ミクモデルの初期姿勢
            #     miku_direction_from_vec = miku_model.bones[direction_from_mname].position
            #     miku_direction_to_vec = miku_model.bones[direction_to_mname].position
            #     miku_direction = (miku_direction_to_vec - miku_direction_from_vec)

            #     miku_up_from_vec = miku_model.bones[up_from_mname].position
            #     miku_up_to_vec = miku_model.bones[up_to_mname].position
            #     miku_up = (miku_up_to_vec - miku_up_from_vec)

            #     miku_cross = MVector3D.crossProduct(miku_direction.normalized(), miku_up.normalized()).normalized()
            #     miku_stance_qq = MQuaternion.fromDirection(miku_direction.normalized(), miku_cross.normalized())

            #     mname = PMX_CONNECTIONS[jname]['mmd']

            #     trace_direction_from_links = trace_model.create_link_2_top_one(direction_from_mname, is_defined=False)
            #     trace_direction_to_links = trace_model.create_link_2_top_one(direction_to_mname, is_defined=False)
            #     trace_up_from_links = trace_model.create_link_2_top_one(up_from_mname, is_defined=False)
            #     trace_up_to_links = trace_model.create_link_2_top_one(up_to_mname, is_defined=False)

            #     trace_direction_from_3ds, trace_direction_from_mats = MServiceUtils.calc_global_pos(trace_model, trace_direction_from_links, VmdMotion(), fno, return_matrix=True)
            #     trace_direction_to_3ds, trace_direction_to_mats = MServiceUtils.calc_global_pos(trace_model, trace_direction_to_links, VmdMotion(), fno, return_matrix=True)
            #     trace_up_from_3ds, trace_up_from_mats = MServiceUtils.calc_global_pos(trace_model, trace_up_from_links, VmdMotion(), fno, return_matrix=True)
            #     trace_up_to_3ds, trace_up_to_mats = MServiceUtils.calc_global_pos(trace_model, trace_up_to_links, VmdMotion(), fno, return_matrix=True)

            #     miku_direction_from_links = miku_model.create_link_2_top_one(direction_from_mname, is_defined=False)
            #     miku_direction_to_links = miku_model.create_link_2_top_one(direction_to_mname, is_defined=False)
            #     miku_up_from_links = miku_model.create_link_2_top_one(up_from_mname, is_defined=False)
            #     miku_up_to_links = miku_model.create_link_2_top_one(up_to_mname, is_defined=False)

            #     for fno in tqdm(fnos, desc=f"No.{oidx:02d}: {mname} ... "):
            #         # オリジナルのグローバル位置
            #         now_direction_from_vec = trace_direction_from_mats[direction_from_mname] * trace_mov_motion.calc_bf(direction_from_mname, fno).position
            #         now_direction_to_vec = trace_direction_to_mats[direction_to_mname] * trace_mov_motion.calc_bf(direction_to_mname, fno).position
            #         now_up_from_vec = trace_up_from_mats[up_from_mname] * trace_mov_motion.calc_bf(up_from_mname, fno).position
            #         now_up_to_vec = trace_up_to_mats[up_to_mname] * trace_mov_motion.calc_bf(up_to_mname, fno).position
                    
            #         # ミク用モーションの生成
                    






















                    # trace_direction_from_links = trace_model.create_link_2_top_one(direction_from_mname, is_defined=False)
                    # trace_direction_to_links = trace_model.create_link_2_top_one(direction_to_mname, is_defined=False)
                    # trace_up_from_links = trace_model.create_link_2_top_one(up_from_mname, is_defined=False)
                    # trace_up_to_links = trace_model.create_link_2_top_one(up_to_mname, is_defined=False)

                    # trace_direction_from_3ds, trace_direction_from_mats = MServiceUtils.calc_global_pos(trace_model, trace_direction_from_links, VmdMotion(), fno, return_matrix=True)
                    # trace_direction_to_3ds, trace_direction_to_mats = MServiceUtils.calc_global_pos(trace_model, trace_direction_to_links, VmdMotion(), fno, return_matrix=True)
                    # trace_up_from_3ds, trace_up_from_mats = MServiceUtils.calc_global_pos(trace_model, trace_up_from_links, VmdMotion(), fno, return_matrix=True)
                    # trace_up_to_3ds, trace_up_to_mats = MServiceUtils.calc_global_pos(trace_model, trace_up_to_links, VmdMotion(), fno, return_matrix=True)

                    # for fidx, fno in enumerate(fnos):

                    #     now_direction_from_vec = trace_direction_from_mats[direction_from_mname] * trace_mov_motion.calc_bf(direction_from_mname, fno).position
                    #     now_direction_to_vec = trace_direction_to_mats[direction_to_mname] * trace_mov_motion.calc_bf(direction_to_mname, fno).position
                    #     now_up_from_vec = trace_up_from_mats[up_from_mname] * trace_mov_motion.calc_bf(up_from_mname, fno).position
                    #     now_up_to_vec = trace_up_to_mats[up_to_mname] * trace_mov_motion.calc_bf(up_to_mname, fno).position
                       
                    #     now_direction = (now_direction_to_vec - now_direction_from_vec).normalized()
                    #     now_up = (now_up_to_vec - now_up_from_vec).normalized()
                    #     now_cross = MVector3D.crossProduct(now_direction, now_up).normalized()
                    #     now_direction_qq = MQuaternion.fromDirection(now_direction, now_cross)

                    #     # 親ボーンのキャンセル分
                    #     cancel_qq = MQuaternion()
                    #     for cancel_jname in jconn["cancel"]:
                    #         cancel_qq *= trace_rot_motion.calc_bf(PMX_CONNECTIONS[cancel_jname]['mmd'], fno).rotation

                    #     now_qq = cancel_qq.inverted() * now_direction_qq * trace_stance_qq.inverted()

                    #     bf = trace_rot_motion.calc_bf(mname, fno)
                    #     bf.rotation = now_qq
                    #     bf.key = True
                    #     trace_rot_motion.bones[mname][fno] = bf

                    #     # ---------------
                    #     now2_direction_from_vec = MVector3D(target_bone_vecs[jconn['direction'][0]][fidx])
                    #     now2_direction_to_vec = MVector3D(target_bone_vecs[jconn['direction'][1]][fidx])
                    #     now2_up_from_vec = MVector3D(target_bone_vecs[jconn['up'][0]][fidx])
                    #     now2_up_to_vec = MVector3D(target_bone_vecs[jconn['up'][1]][fidx])

                    #     now2_direction = (now2_direction_to_vec - now2_direction_from_vec).normalized()
                    #     now2_up = (now2_up_to_vec - now2_up_from_vec).normalized()
                    #     now2_cross = MVector3D.crossProduct(now2_direction, now2_up).normalized()
                    #     now2_direction_qq = MQuaternion.fromDirection(now2_direction, now2_cross)

                    #     # 親ボーンのキャンセル分
                    #     cancel2_qq = MQuaternion()
                    #     for cancel_jname in jconn["cancel"]:
                    #         cancel2_qq *= motion.calc_bf(PMX_CONNECTIONS[cancel_jname]['mmd'], fno).rotation

                    #     now2_qq = cancel2_qq.inverted() * now2_direction_qq * trace_stance_qq.inverted()

                    #     now2_bf = motion.calc_bf(mname, fno)
                    #     now2_bf.rotation = now2_qq
                    #     now2_bf.key = True
                    #     motion.bones[mname][fno] = now2_bf
                    #     pchar.update(1)







                    # for fno, direction_from_vec, direction_to_vec, up_from_vec, up_to_vec in zip(fnos, target_bone_vecs[jconn['direction'][0]], target_bone_vecs[jconn['direction'][1]], \
                    #                                                                              target_bone_vecs[jconn['up'][0]], target_bone_vecs[jconn['up'][1]]):                    
                    #     now_direction_from_vec = trace_direction_from_mats[direction_from_mname] * MVector3D(direction_from_vec)
                    #     now_direction_to_vec = trace_direction_to_mats[] * MVector3D(direction_to_vec)
                    #     now_up_from_vec = MVector3D(up_from_vec) + trace_model.bones[up_from_mname].position
                    #     now_up_to_vec = MVector3D(up_to_vec) + trace_model.bones[up_to_mname].position
                       
                    #     now_direction = (now_direction_to_vec - now_direction_from_vec).normalized()
                    #     now_up = (now_up_to_vec - now_up_from_vec).normalized()
                    #     now_cross = MVector3D.crossProduct(now_direction, now_up).normalized()
                    #     now_direction_qq = MQuaternion.fromDirection(now_direction, now_cross)

                    #     # 親ボーンのキャンセル分
                    #     cancel_qq = MQuaternion()
                    #     for cancel_jname in reversed(jconn["cancel"]):
                    #         cancel_qq *= trace_rot_motion.calc_bf(PMX_CONNECTIONS[cancel_jname]['mmd'], fno).rotation

                    #     now_qq = cancel_qq.inverted() * now_direction_qq * trace_stance_qq.inverted()

                    #     bf = trace_rot_motion.calc_bf(mname, fno)
                    #     bf.rotation = now_qq
                    #     bf.key = True
                    #     trace_rot_motion.bones[mname][fno] = bf
                    #     pchar.update(1)

                    # trace_direction_from_links = trace_model.create_link_2_top_one(direction_from_mname, is_defined=False)
                    # trace_direction_to_links = trace_model.create_link_2_top_one(direction_to_mname, is_defined=False)
                    # trace_up_from_links = trace_model.create_link_2_top_one(up_from_mname, is_defined=False)
                    # trace_up_to_links = trace_model.create_link_2_top_one(up_to_mname, is_defined=False)

                    # for fno in fnos:
                    #     trace_direction_from_3ds = MServiceUtils.calc_global_pos(trace_model, trace_direction_from_links, trace_mov_motion, fno)
                    #     trace_direction_to_3ds = MServiceUtils.calc_global_pos(trace_model, trace_direction_to_links, trace_mov_motion, fno)
                    #     trace_up_from_3ds = MServiceUtils.calc_global_pos(trace_model, trace_up_from_links, trace_mov_motion, fno)
                    #     trace_up_to_3ds = MServiceUtils.calc_global_pos(trace_model, trace_up_to_links, trace_mov_motion, fno)

                    #     now_direction_from_vec = trace_direction_from_3ds[direction_from_mname]
                    #     now_direction_to_vec = trace_direction_to_3ds[direction_to_mname]
                    #     now_up_from_vec = trace_up_from_3ds[up_from_mname]
                    #     now_up_to_vec = trace_up_to_3ds[up_to_mname]
                       
                    #     now_direction = (now_direction_to_vec - now_direction_from_vec).normalized()
                    #     now_up = (now_up_to_vec - now_up_from_vec).normalized()
                    #     now_cross = MVector3D.crossProduct(now_direction, now_up).normalized()
                    #     now_direction_qq = MQuaternion.fromDirection(now_direction, now_cross)

                    #     # 親ボーンのキャンセル分
                    #     cancel_qq = MQuaternion()
                    #     for cancel_jname in reversed(jconn["cancel"]):
                    #         cancel_qq *= trace_rot_motion.calc_bf(PMX_CONNECTIONS[cancel_jname]['mmd'], fno).rotation

                    #     now_qq = now_direction_qq * trace_stance_qq.inverted()

                    #     bf = trace_rot_motion.calc_bf(mname, fno)
                    #     bf.rotation = now_qq
                    #     bf.key = True
                    #     trace_rot_motion.bones[mname][fno] = bf
                    #     pchar.update(1)

            # logger.info("【No.{0}】ミクモーション結果回転計算開始", f"{oidx:02d}", decoration=MLogger.DECORATION_LINE)

            # for jname, jconn in VMD_CONNECTIONS.items():
            #     direction_from_mname = PMX_CONNECTIONS[jconn["direction"][0]]['mmd']
            #     direction_to_mname = PMX_CONNECTIONS[jconn["direction"][1]]['mmd']
            #     up_from_mname = PMX_CONNECTIONS[jconn["up"][0]]['mmd']
            #     up_to_mname = PMX_CONNECTIONS[jconn["up"][1]]['mmd']

            #     # トレースモデルの初期姿勢
            #     trace_direction_from_vec = trace_model.bones[direction_from_mname].position
            #     trace_direction_to_vec = trace_model.bones[direction_to_mname].position
            #     trace_direction = (trace_direction_to_vec - trace_direction_from_vec)

            #     trace_up_from_vec = trace_model.bones[up_from_mname].position
            #     trace_up_to_vec = trace_model.bones[up_to_mname].position
            #     trace_up = (trace_up_to_vec - trace_up_from_vec)

            #     trace_cross = MVector3D.crossProduct(trace_direction.normalized(), trace_up.normalized()).normalized()
            #     trace_stance_qq = MQuaternion.fromDirection(trace_direction.normalized(), trace_cross.normalized())

            #     # ミクモデルの初期姿勢
            #     miku_direction_from_vec = miku_model.bones[direction_from_mname].position
            #     miku_direction_to_vec = miku_model.bones[direction_to_mname].position
            #     miku_direction = (miku_direction_to_vec - miku_direction_from_vec)

            #     miku_up_from_vec = miku_model.bones[up_from_mname].position
            #     miku_up_to_vec = miku_model.bones[up_to_mname].position
            #     miku_up = (miku_up_to_vec - miku_up_from_vec)

            #     miku_cross = MVector3D.crossProduct(miku_direction.normalized(), miku_up.normalized()).normalized()
            #     miku_stance_qq = MQuaternion.fromDirection(miku_direction.normalized(), miku_cross.normalized())

            #     mname = PMX_CONNECTIONS[jname]['mmd']

            #     trace_direction_from_links = trace_model.create_link_2_top_one(direction_from_mname, is_defined=False)
            #     trace_direction_to_links = trace_model.create_link_2_top_one(direction_to_mname, is_defined=False)
            #     trace_up_from_links = trace_model.create_link_2_top_one(up_from_mname, is_defined=False)
            #     trace_up_to_links = trace_model.create_link_2_top_one(up_to_mname, is_defined=False)

            #     miku_direction_from_links = miku_model.create_link_2_top_one(direction_from_mname, is_defined=False)
            #     miku_direction_to_links = miku_model.create_link_2_top_one(direction_to_mname, is_defined=False)
            #     miku_up_from_links = miku_model.create_link_2_top_one(up_from_mname, is_defined=False)
            #     miku_up_to_links = miku_model.create_link_2_top_one(up_to_mname, is_defined=False)

            #     for fno in tqdm(fnos, desc=f"No.{oidx:02d}: {mname}(ミク) ... "):
            #         # ミク用モーションの生成
                    
            #         _, trace_direction_from_mats = MServiceUtils.calc_global_pos(trace_model, trace_direction_from_links, trace_mov_motion, fno, return_matrix=True)
            #         trace_direction_to_3ds = MServiceUtils.calc_global_pos(trace_model, trace_direction_to_links, trace_mov_motion, fno)
            #         # directionの初期姿勢to絶対位置
            #         trace_direction_to_initial_vec = trace_direction_from_mats[direction_from_mname] * trace_direction
            #         # 初期姿勢からの差分
            #         trace_direction_to_diff_vec = trace_direction_to_3ds[direction_to_mname] - trace_direction_to_initial_vec
            #         # toの相対位置をミクモデルに適用
            #         miku_direction_from = miku_direction_from_vec
            #         miku_direction_to = miku_direction_to_vec + trace_direction_to_diff_vec

            #         _, trace_up_from_mats = MServiceUtils.calc_global_pos(trace_model, trace_up_from_links, trace_mov_motion, fno, return_matrix=True)
            #         trace_up_to_3ds = MServiceUtils.calc_global_pos(trace_model, trace_up_to_links, trace_mov_motion, fno)
            #         # upの初期姿勢to絶対位置
            #         trace_up_to_initial_vec = trace_up_from_mats[up_from_mname] * trace_up
            #         # 初期姿勢からの差分
            #         trace_up_to_diff_vec = trace_up_to_3ds[up_to_mname] - trace_up_to_initial_vec
            #         # toの相対位置をミクモデルに適用
            #         miku_up_from = miku_up_from_vec
            #         miku_up_to = miku_up_to_vec + trace_up_to_diff_vec

            #         miku_direction = (miku_direction_to - miku_direction_from).normalized()
            #         miku_up = (miku_up_to - miku_up_from).normalized()
            #         miku_cross = MVector3D.crossProduct(miku_direction, miku_up).normalized()
            #         miku_direction_qq = MQuaternion.fromDirection(miku_direction, miku_cross)

            #         # 親ボーンのキャンセル分
            #         miku_cancel_qq = MQuaternion()
            #         for cancel_jname in reversed(jconn["cancel"]):
            #             miku_cancel_qq *= motion.calc_bf(PMX_CONNECTIONS[cancel_jname]['mmd'], fno).rotation

            #         miku_qq = miku_cancel_qq.inverted() * miku_direction_qq * miku_stance_qq.inverted()

            #         miku_bf = motion.calc_bf(mname, fno)
            #         miku_bf.rotation = miku_qq
            #         miku_bf.key = True
            #         motion.bones[mname][fno] = miku_bf
                    




                # for fno in tqdm(fnos, desc=f"No.{oidx:02d}: {mname} ... "):
                #     # ミク用モーションの生成
                #     _, trace_direction_from_mats = MServiceUtils.calc_global_pos(trace_model, trace_direction_from_links, trace_mov_motion, fno, return_matrix=True)
                #     trace_direction_to_3ds = MServiceUtils.calc_global_pos(trace_model, trace_direction_to_links, trace_mov_motion, fno)
                #     # directionのfromから見たtoの相対位置
                #     trace_direction_to_relative_vec = trace_direction_from_mats[direction_from_mname].inverted() * trace_direction_to_3ds[direction_to_mname]
                #     # trace_direction_diff_vec = trace_direction_to_relative_vec - trace_direction
                #     now_direction = trace_direction_to_relative_vec.normalized()


                # for fno in tqdm(fnos, desc=f"No.{oidx:02d}: {mname} ... "):
                #     # ミク用モーションの生成
                #     _, trace_direction_from_mats = MServiceUtils.calc_global_pos(trace_model, trace_direction_from_links, trace_mov_motion, fno, return_matrix=True)
                #     trace_direction_to_3ds = MServiceUtils.calc_global_pos(trace_model, trace_direction_to_links, trace_mov_motion, fno)
                #     # directionのfromから見たtoの相対位置
                #     trace_direction_to_relative_vec = trace_direction_from_mats[direction_from_mname].inverted() * trace_direction_to_3ds[direction_to_mname]
                #     # trace_direction_diff_vec = trace_direction_to_relative_vec - trace_direction
                #     now_direction = trace_direction_to_relative_vec.normalized()

                #     _, trace_up_from_mats = MServiceUtils.calc_global_pos(trace_model, trace_up_from_links, trace_mov_motion, fno, return_matrix=True)
                #     trace_up_to_3ds = MServiceUtils.calc_global_pos(trace_model, trace_up_to_links, trace_mov_motion, fno)
                #     # upのfromから見たtoの相対位置
                #     trace_up_to_relative_vec = trace_up_from_mats[up_from_mname].inverted() * trace_up_to_3ds[up_to_mname]
                #     # trace_up_diff_vec = trace_up_to_relative_vec - trace_up
                #     now_up = trace_up_to_relative_vec.normalized()
                    
                #     now_cross = MVector3D.crossProduct(now_direction, now_up).normalized()
                #     now_direction_qq = MQuaternion.fromDirection(now_direction, now_cross)

                #     # 親ボーンのキャンセル分
                #     cancel_qq = MQuaternion()
                #     for cancel_jname in reversed(jconn["cancel"]):
                #         cancel_qq *= motion.calc_bf(PMX_CONNECTIONS[cancel_jname]['mmd'], fno).rotation

                #     now_qq = cancel_qq.inverted() * now_direction_qq * miku_stance_qq.inverted()

                #     bf = motion.calc_bf(mname, fno)
                #     bf.rotation = now_qq
                #     bf.key = True
                #     motion.bones[mname][fno] = bf


                #     # # toの相対位置をミクモデルに適用
                #     # miku_direction_from = trace_direction_from_mats[direction_from_mname] * MVector3D()
                #     # miku_direction_to = trace_direction_from_mats[direction_from_mname] * trace_direction_to_relative_vec

                # for fno, direction_from_vec, direction_to_vec, up_from_vec, up_to_vec in tqdm(zip(fnos, target_bone_vecs[jconn['direction'][0]], target_bone_vecs[jconn['direction'][1]], \
                #                                                                                   target_bone_vecs[jconn['up'][0]], target_bone_vecs[jconn['up'][1]]), desc=f"No.{oidx:02d}: {mname}(トレース) ... "):
                    
                #     now_direction = MVector3D(direction_to_vec - direction_from_vec) - trace_direction + miku_direction
                #     now_up = MVector3D(up_to_vec - up_from_vec) - trace_up + miku_up
                #     now_cross = MVector3D.crossProduct(now_direction.normalized(), now_up.normalized())
                #     now_direction_qq = MQuaternion.fromDirection(now_direction.normalized(), now_cross.normalized())

                #     # 親ボーンのキャンセル分
                #     cancel_qq = MQuaternion()
                #     for cancel_jname in reversed(jconn["cancel"]):
                #         cancel_qq *= motion.calc_bf(PMX_CONNECTIONS[cancel_jname]['mmd'], fno).rotation

                #     now_qq = cancel_qq.inverted() * now_direction_qq * miku_stance_qq.inverted()

                #     bf = motion.calc_bf(mname, fno)
                #     bf.rotation = now_qq
                #     bf.key = True
                #     motion.bones[mname][fno] = bf


                # for fno, direction_from_vec, direction_to_vec, up_from_vec, up_to_vec in tqdm(zip(fnos, target_bone_vecs[jconn['direction'][0]], target_bone_vecs[jconn['direction'][1]], \
                #                                                                                   target_bone_vecs[jconn['up'][0]], target_bone_vecs[jconn['up'][1]]), desc=f"No.{oidx:02d}: {mname}(トレース) ... "):
                #     now_direction = (MVector3D(direction_to_vec) - MVector3D(direction_from_vec)).normalized()
                #     now_up = (MVector3D(up_to_vec) - MVector3D(up_from_vec)).normalized()
                #     now_cross = MVector3D.crossProduct(now_direction, now_up).normalized()
                #     now_direction_qq = MQuaternion.fromDirection(now_direction, now_cross)

                #     # 親ボーンのキャンセル分
                #     cancel_qq = MQuaternion()
                #     for cancel_jname in jconn["cancel"]:
                #         cancel_qq *= trace_rot_motion.calc_bf(PMX_CONNECTIONS[cancel_jname]['mmd'], fno).rotation

                #     now_qq = cancel_qq.inverted() * now_direction_qq * trace_stance_qq.inverted()

                #     bf = trace_rot_motion.calc_bf(mname, fno)
                #     bf.rotation = now_qq
                #     bf.key = True
                #     trace_rot_motion.bones[mname][fno] = bf

                # for fno in tqdm(fnos, desc=f"No.{oidx:02d}: {mname}(ミク) ... "):
                #     # ミク用モーションの生成
                #     _, trace_direction_from_mats = MServiceUtils.calc_global_pos(trace_model, trace_direction_from_links, trace_rot_motion, fno, return_matrix=True)
                #     trace_direction_to_3ds = MServiceUtils.calc_global_pos(trace_model, trace_direction_to_links, trace_rot_motion, fno)
                #     _, miku_direction_from_mats = MServiceUtils.calc_global_pos(model, miku_direction_from_links, motion, fno, return_matrix=True)
                #     # directionのfromから見たtoの相対位置
                #     trace_direction_to_relative_vec = trace_direction_from_mats[direction_from_mname].inverted() * trace_direction_to_3ds[direction_to_mname]
                #     # toの相対位置をミクモデルに適用
                #     miku_direction_from = miku_direction_from_mats[direction_from_mname] * MVector3D()
                #     miku_direction_to = miku_direction_from_mats[direction_from_mname] * trace_direction_to_relative_vec

                #     _, trace_up_from_mats = MServiceUtils.calc_global_pos(trace_model, trace_up_from_links, trace_rot_motion, fno, return_matrix=True)
                #     trace_up_to_3ds = MServiceUtils.calc_global_pos(trace_model, trace_up_to_links, trace_rot_motion, fno)
                #     _, miku_up_from_mats = MServiceUtils.calc_global_pos(model, miku_up_from_links, motion, fno, return_matrix=True)
                #     # upのfromから見たtoの相対位置
                #     trace_up_to_relative_vec = trace_up_from_mats[up_from_mname].inverted() * trace_up_to_3ds[up_to_mname]
                #     # toの相対位置をミクモデルに適用
                #     miku_up_from = miku_up_from_mats[up_from_mname] * MVector3D()
                #     miku_up_to = miku_up_from_mats[up_from_mname] * trace_up_to_relative_vec

                #     miku_direction = (miku_direction_to - miku_direction_from).normalized()
                #     miku_up = (miku_up_to - miku_up_from).normalized()
                #     miku_cross = MVector3D.crossProduct(miku_direction, miku_up).normalized()
                #     miku_direction_qq = MQuaternion.fromDirection(miku_direction, miku_cross)

                #     # 親ボーンのキャンセル分
                #     miku_cancel_qq = MQuaternion()
                #     for cancel_jname in jconn["cancel"]:
                #         miku_cancel_qq *= motion.calc_bf(PMX_CONNECTIONS[cancel_jname]['mmd'], fno).rotation

                #     miku_qq = miku_cancel_qq.inverted() * miku_direction_qq * miku_stance_qq.inverted()

                #     miku_bf = motion.calc_bf(mname, fno)
                #     miku_bf.rotation = miku_qq
                #     miku_bf.key = True
                #     motion.bones[mname][fno] = miku_bf

                    
                    # trace_bf.position = (MVector3D(joint) * MIKU_METER) - (MVector3D(parent_joint) * MIKU_METER) \
                    #                     - (trace_model.bones[mname].position - trace_model.bones[pmname].position)


                    # # directionのtoの位置を、ミクに合わせて再計算
                    # direction_from_ = trace_mov_motion.calc_bf(direction_from_mname, fno)
                    # trace_bf.position

                    # now_trace_direction_from = trace_direction_from_vec + now_trace_direction_diff
                    # now_trace_direction_to = trace_direction_to_vec + now_trace_direction_diff




                    # now_trace_direction = trace_direction + now_trace_direction_diff
                    # now_miku_direction_to = miku_direction_from_vec + now_trace_direction
                    # now_direction = (MVector3D(now_miku_direction_to) - MVector3D(now_trace_direction)).normalized()

                    # now_trace_up_diff = MVector3D(up_to_vec) - MVector3D(up_from_vec)
                    # now_trace_up = trace_up + now_trace_up_diff
                    # now_miku_up_to = miku_up_from_vec + now_trace_up
                    # now_up = (MVector3D(now_miku_up_to) - MVector3D(now_trace_up)).normalized()

                    # trace_bf = trace_rot_motion.calc_bf(mname, fno)
                    # trace_bf.rotation = cancel_qq.inverted() * now_qq * initial_stance_qq.inverted()
                    # trace_bf.key = True
                    # trace_rot_motion.bones[mname][fno] = trace_bf

                    # # 軍曹との差分
                    # now_trace_vec = (trace_bf.rotation * now_direction)
                    # now_trace_up = MVector3D.crossProduct(miku_direction_vec.normalized(), now_trace_vec.normalized())
                    # now_trace_cross = MVector3D.crossProduct(miku_direction_vec.normalized(), now_trace_up.normalized())
                    # now_trace_dot = MVector3D.dotProduct(miku_direction_vec.normalized(), now_trace_vec.normalized())
                    # now_trace_degree = math.degrees(max(-1, min(1, math.acos(now_trace_dot))))
                    # diff_stance_qq = MQuaternion.fromAxisAndAngle(now_trace_cross, now_trace_degree)

                    # now_target_vec = now_qq * initial_direction
                    # correct_qq = MQuaternion.rotationTo(miku_direction, now_target_vec)

                    # if jconn['diff']:
                    #     if 'y' in jconn['diff']:
                    #         now_qq *= miku_trace_diff_y_qq.inverted()
                    #     if 'x' in jconn['diff']:
                    #         now_qq *= miku_trace_diff_x_qq.inverted()
                    #     if 'z' in jconn['diff']:
                    #         now_qq *= miku_trace_diff_z_qq.inverted()

                    # now_initial_vec = trace_bf.rotation * initial_direction_vec
                    # now_miku_vec = trace_bf.rotation * miku_direction_vec
                    # now_collect_qq = MQuaternion.rotationTo(now_miku_vec, now_initial_vec)

                    # mbname = PMX_CONNECTIONS[jname]["mmd"]
                    # miku_bf = motion.calc_bf(mbname, fno)
                    # miku_bf.rotation = bf.rotation
                    # miku_bf.key = True
                    # motion.bones[mbname][fno] = miku_bf

            # for jname, pconn in PMX_CONNECTIONS.items():
            #     if pconn["mmd"] in miku_model.bones:
            #         # 対応するボーンのみ登録
            #         if pconn["mmd"] not in motion.bones and jname in trace_rot_motion.bones:
            #             jconn = VMD_CONNECTIONS[jname]
            #             parent_default_vec = MVector3D(1, 0, 0) if "左" in pconn["parent"] else MVector3D(-1, 0, 0)
            #             default_vec = MVector3D(1, 0, 0) if "左" in pconn["mmd"] else MVector3D(-1, 0, 0)
                        
            #             if pconn["parent"] in VMD_CONNECTIONS:
            #                 pjconn = VMD_CONNECTIONS[pconn["parent"]]

            #                 # 親ボーンの初期姿勢
            #                 parent_initial_direction_from_vec = trace_model.bones[pjconn["direction"][0]].position
            #                 parent_initial_direction_to_vec = trace_model.bones[pjconn["direction"][1]].position
            #                 parent_initial_direction = (parent_initial_direction_to_vec - parent_initial_direction_from_vec).normalized()

            #                 parent_initial_up_from_vec = trace_model.bones[pjconn["up"][0]].position
            #                 parent_initial_up_to_vec = trace_model.bones[pjconn["up"][1]].position
            #                 parent_initial_up = (parent_initial_up_to_vec - parent_initial_up_from_vec).normalized()
                            
            #                 parent_initial_cross = MVector3D.crossProduct(parent_initial_direction, parent_initial_up).normalized()
            #                 parent_initial_stance_qq = MQuaternion.fromDirection(parent_initial_direction, parent_initial_cross)

            #                 # 軍曹の親ボーンの初期姿勢
            #                 parent_miku_direction_from_vec = miku_model.bones[PMX_CONNECTIONS[pjconn["direction"][0]]["mmd"]].position
            #                 parent_miku_direction_to_vec = miku_model.bones[PMX_CONNECTIONS[pjconn["direction"][1]]["mmd"]].position
            #                 parent_miku_direction = (parent_miku_direction_to_vec - parent_miku_direction_from_vec).normalized()

            #                 parent_miku_up_from_vec = miku_model.bones[PMX_CONNECTIONS[pjconn["up"][0]]["mmd"]].position
            #                 parent_miku_up_to_vec = miku_model.bones[PMX_CONNECTIONS[pjconn["up"][1]]["mmd"]].position
            #                 parent_miku_up = (parent_miku_up_to_vec - parent_miku_up_from_vec).normalized()

            #                 parent_miku_cross = MVector3D.crossProduct(parent_miku_direction, parent_miku_up).normalized()
            #                 parent_miku_stance_qq = MQuaternion.fromDirection(parent_miku_direction, parent_miku_cross)

            #                 # 親ボーンのスタンス補正角度
            #                 parent_stance_qq = parent_miku_stance_qq.inverted() * parent_initial_stance_qq
            #             else:
            #                 parent_stance_qq = MQuaternion()

            #             # # 初期姿勢
            #             # initial_direction_from_vec = trace_model.bones[jconn["direction"][0]].position
            #             # initial_direction_to_vec = trace_model.bones[jconn["direction"][1]].position
            #             # initial_direction = (initial_direction_to_vec - initial_direction_from_vec).normalized()

            #             # initial_up_from_vec = trace_model.bones[jconn["up"][0]].position
            #             # initial_up_to_vec = trace_model.bones[jconn["up"][1]].position
            #             # initial_up = (initial_up_to_vec - initial_up_from_vec).normalized()

            #             # initial_cross = MVector3D.crossProduct(initial_direction, initial_up).normalized()
            #             # initial_stance_qq = MQuaternion.fromDirection(initial_direction, initial_cross)

            #             # # 軍曹の初期姿勢
            #             # miku_direction_from_vec = miku_model.bones[PMX_CONNECTIONS[jconn["direction"][0]]["mmd"]].position
            #             # miku_direction_to_vec = miku_model.bones[PMX_CONNECTIONS[jconn["direction"][1]]["mmd"]].position
            #             # miku_direction = (miku_direction_to_vec - miku_direction_from_vec).normalized()

            #             # miku_up_from_vec = miku_model.bones[PMX_CONNECTIONS[jconn["up"][0]]["mmd"]].position
            #             # miku_up_to_vec = miku_model.bones[PMX_CONNECTIONS[jconn["up"][1]]["mmd"]].position
            #             # miku_up = (miku_up_to_vec - miku_up_from_vec).normalized()

            #             # miku_cross = MVector3D.crossProduct(miku_direction, miku_up).normalized()
            #             # miku_stance_qq = MQuaternion.fromDirection(miku_direction, miku_cross)

            #             # # 対象ボーンのスタンス補正角度
            #             # stance_qq = miku_stance_qq.inverted() * initial_stance_qq

            #             # 初期姿勢
            #             initial_direction_from_vec = trace_model.bones[jconn["direction"][0]].position
            #             initial_direction_to_vec = trace_model.bones[jconn["direction"][1]].position
            #             initial_direction = (initial_direction_to_vec - initial_direction_from_vec).normalized()

            #             initial_up_from_vec = trace_model.bones[jconn["up"][0]].position
            #             initial_up_to_vec = trace_model.bones[jconn["up"][1]].position
            #             initial_up = (initial_up_to_vec - initial_up_from_vec).normalized()
                        
            #             initial_cross = MVector3D.crossProduct(initial_direction, initial_up).normalized()
            #             initial_stance_qq = MQuaternion.fromDirection(initial_direction, initial_cross)

            #             # 軍曹の初期姿勢
            #             miku_direction_from_vec = miku_model.bones[PMX_CONNECTIONS[jconn["direction"][0]]["mmd"]].position
            #             miku_direction_to_vec = miku_model.bones[PMX_CONNECTIONS[jconn["direction"][1]]["mmd"]].position
            #             miku_direction = (miku_direction_to_vec - miku_direction_from_vec).normalized()

            #             miku_up_from_vec = miku_model.bones[PMX_CONNECTIONS[jconn["up"][0]]["mmd"]].position
            #             miku_up_to_vec = miku_model.bones[PMX_CONNECTIONS[jconn["up"][1]]["mmd"]].position
            #             miku_up = (miku_up_to_vec - miku_up_from_vec).normalized()

            #             miku_cross = MVector3D.crossProduct(miku_direction, miku_up).normalized()
            #             miku_stance_qq = MQuaternion.fromDirection(miku_direction, miku_cross)

            #             # スタンス補正角度
            #             stance_qq = miku_stance_qq.inverted() * initial_stance_qq

            #             for fno in tqdm(fnos, desc=f"No.{oidx:02d}:{pconn['mmd']} ... "):
            #                 trace_bf = trace_rot_motion.calc_bf(jname, fno)
            #                 bf = motion.calc_bf(pconn["mmd"], fno)
            #                 bf.rotation = parent_stance_qq.inverted() * trace_bf.rotation.copy() * stance_qq
            #                 bf.key = True
            #                 motion.bones[pconn["mmd"]][fno] = bf
            
            # for jidx, (jname, joints) in enumerate(target_bone_vecs.items()):
            #     if jname in VMD_CONNECTIONS:
            #         bname = VMD_CONNECTIONS[jname]["bname"]
            #         default_vec = MVector3D(1, 0, 0) if "左" in bname else MVector3D(-1, 0, 0) if "右" in bname else MVector3D(0, 1, 0)

            #         # 初期姿勢
            #         initial_direction_from_vec = trace_model.bones[VMD_CONNECTIONS[jname]["direction"][0]].position
            #         initial_direction_to_vec = trace_model.bones[VMD_CONNECTIONS[jname]["direction"][1]].position
            #         initial_direction = (initial_direction_to_vec - initial_direction_from_vec).normalized()

            #         initial_up_from_vec = trace_model.bones[VMD_CONNECTIONS[jname]["up"][0]].position
            #         initial_up_to_vec = trace_model.bones[VMD_CONNECTIONS[jname]["up"][1]].position
            #         initial_up = (initial_up_to_vec - initial_up_from_vec).normalized()

            #         initial_cross = MVector3D.crossProduct(initial_direction, initial_up).normalized()
            #         initial_qq = MQuaternion.fromDirection(initial_direction, initial_cross)
            #         initial_stance_qq = MQuaternion.rotationTo(default_vec, initial_direction)

            #         # 軍曹の初期姿勢
            #         miku_direction_from_vec = miku_model.bones[PMX_CONNECTIONS[VMD_CONNECTIONS[jname]["direction"][0]]["mmd"]].position
            #         miku_direction_to_vec = miku_model.bones[PMX_CONNECTIONS[VMD_CONNECTIONS[jname]["direction"][1]]["mmd"]].position
            #         miku_direction = (miku_direction_to_vec - miku_direction_from_vec).normalized()

            #         miku_up_from_vec = miku_model.bones[PMX_CONNECTIONS[VMD_CONNECTIONS[jname]["up"][0]]["mmd"]].position
            #         miku_up_to_vec = miku_model.bones[PMX_CONNECTIONS[VMD_CONNECTIONS[jname]["up"][1]]["mmd"]].position
            #         miku_up = (miku_up_to_vec - miku_up_from_vec).normalized()

            #         miku_cross = MVector3D.crossProduct(miku_direction, miku_up).normalized()
            #         miku_qq = MQuaternion.fromDirection(miku_direction, miku_cross)
            #         miku_stance_qq = MQuaternion.rotationTo(default_vec, miku_direction)
            #         stance_qq = MQuaternion.rotationTo(initial_direction, miku_direction)

            #         for fno, joint in tqdm(zip(fnos, joints), desc=f"No.{oidx:02d}:{jname} ... "):
            #             if jname in VMD_CONNECTIONS:
            #                 # 現在の姿勢
            #                 now_direction_from_vec = trace_mov_motion.calc_bf(VMD_CONNECTIONS[jname]["direction"][0], fno).position + initial_direction_from_vec
            #                 now_direction_to_vec = trace_mov_motion.calc_bf(VMD_CONNECTIONS[jname]["direction"][1], fno).position + initial_direction_to_vec
            #                 now_direction = (now_direction_to_vec - now_direction_from_vec).normalized()

            #                 now_up_from_vec = trace_mov_motion.calc_bf(VMD_CONNECTIONS[jname]["up"][0], fno).position + initial_up_from_vec
            #                 now_up_to_vec = trace_mov_motion.calc_bf(VMD_CONNECTIONS[jname]["up"][1], fno).position + initial_up_to_vec
            #                 now_up = (now_up_to_vec - now_up_from_vec).normalized()

            #                 now_cross = MVector3D.crossProduct(now_direction, now_up).normalized()
            #                 now_qq = MQuaternion.fromDirection(now_direction, now_cross)

            #                 # 親ボーンのキャンセル分
            #                 cancel_qq = MQuaternion()
            #                 for cancel_bname in VMD_CONNECTIONS[jname]["cancel"]:
            #                     cancel_qq *= motion.calc_bf(cancel_bname, fno).rotation

            #                 bf = motion.calc_bf(bname, fno)
            #                 trace_qq = cancel_qq.inverted() * now_qq * initial_qq.inverted()
            #                 # 現時点の回転に応じたdirectionの位置
            #                 now_initial_vec = trace_qq * initial_direction
            #                 miku_initial_vec = trace_qq * miku_direction
            #                 # 回転の差分
            #                 diff_qq = MQuaternion.rotationTo(now_initial_vec, miku_initial_vec)
            #                 bf.rotation = trace_qq #* stance_qq.inverted() * miku_stance_qq
            #                 bf.key = True
            #                 motion.bones[bname][fno] = bf

            # trace_miku_motion_path = os.path.join(motion_dir_path, f"trace_{process_datetime}_miku_no{oidx:02d}.vmd")
            # logger.info("【No.{0}】トレースモーション(あにまさ式ミク)生成開始【{1}】", f"{oidx:02d}", os.path.basename(trace_miku_motion_path), decoration=MLogger.DECORATION_LINE)
            # vmd_writer.write(miku_model, miku_motion, trace_miku_motion_path)

            # logger.info("【No.{0}】トレースモーション(回転)生成開始", f"{oidx:02d}", decoration=MLogger.DECORATION_LINE)
            # vmd_writer.write(model, motion, os.path.join(motion_dir_path, f"trace_{process_datetime}_rot_no{oidx:02d}.vmd"))


        # for oidx, ordered_person_dir_path in enumerate(ordered_person_dir_pathes):    
        #     logger.info("【No.{0}】FKボーン角度計算開始", f"{oidx:02d}", decoration=MLogger.DECORATION_LINE)

        #     smooth_json_pathes = sorted(glob.glob(os.path.join(ordered_person_dir_path, "smooth_*.json")), key=sort_by_numeric)

        #     motion = VmdMotion()
        #     all_frame_joints = {}
        #     prev_fno = 9999999999

        #     right_leg_lengths = []
        #     left_leg_lengths = []
        #     leg_lengths = []
        #     # foot_ys = []
        #     leg_degrees = []
        #     flip_fnos = []
        #     # KEY: 処理対象ボーン名, VALUE: 誤差許容範囲
        #     target_bone_names = {}
        
        #     for sidx, smooth_json_path in enumerate(tqdm(smooth_json_pathes, desc=f"No.{oidx:02d} ... ")):
        #         m = smooth_pattern.match(os.path.basename(smooth_json_path))
        #         if m:
        #             # キーフレの場所を確定（間が空く場合もある）
        #             fno = int(m.groups()[0])

        #             frame_joints = {}
        #             with open(smooth_json_path, 'r', encoding='utf-8') as f:
        #                 frame_joints = json.load(f)
        #             all_frame_joints[fno] = frame_joints

        #             left_hip_vec = get_vec3(all_frame_joints[fno]["joints"], "left_hip")
        #             left_foot_vec = get_vec3(all_frame_joints[fno]["joints"], "left_foot")
        #             right_hip_vec = get_vec3(all_frame_joints[fno]["joints"], "right_hip")
        #             right_foot_vec = get_vec3(all_frame_joints[fno]["joints"], "right_foot")

        #             right_leg_lengths.append(right_hip_vec.distanceToPoint(right_foot_vec))
        #             left_leg_lengths.append(left_hip_vec.distanceToPoint(left_foot_vec))
        #             # 両足の長さを平均
        #             leg_lengths.append(np.mean([left_hip_vec.distanceToPoint(left_foot_vec), right_hip_vec.distanceToPoint(right_foot_vec)]))
        #             # # 踵の位置を平均
        #             # foot_ys.append(np.mean([left_foot_vec.y(), right_foot_vec.y()]))

        #             if args.body_motion == 1 or args.face_motion == 1:
                        
        #                 is_flip = False
        #                 if prev_fno < fno and sorted(all_frame_joints.keys())[-1] <= sorted(all_frame_joints.keys())[-2] + 2:
        #                     # 前のキーフレがある場合、体幹に近いボーンが反転していたらスルー(直近が3フレーム以上離れていたらスルーなし)
        #                     for ljname in ['left_hip', 'left_shoulder']:
        #                         rjname = ljname.replace('left', 'right')
        #                         prev_lsign = np.sign(all_frame_joints[prev_fno]["joints"][ljname]["x"] - all_frame_joints[prev_fno]["joints"]["pelvis"]["x"])
        #                         prev_rsign = np.sign(all_frame_joints[prev_fno]["joints"][rjname]["x"] - all_frame_joints[prev_fno]["joints"]["pelvis"]["x"])
        #                         lsign = np.sign(all_frame_joints[fno]["joints"][ljname]["x"] - all_frame_joints[fno]["joints"]["pelvis"]["x"])
        #                         rsign = np.sign(all_frame_joints[fno]["joints"][rjname]["x"] - all_frame_joints[fno]["joints"]["pelvis"]["x"])
        #                         ldiff = abs(np.diff([all_frame_joints[prev_fno]["joints"][ljname]["x"], all_frame_joints[fno]["joints"][ljname]["x"]]))
        #                         rdiff = abs(np.diff([all_frame_joints[prev_fno]["joints"][rjname]["x"], all_frame_joints[fno]["joints"][rjname]["x"]]))
        #                         lrdiff = abs(np.diff([all_frame_joints[fno]["joints"][ljname]["x"], all_frame_joints[fno]["joints"][rjname]["x"]]))

        #                         if prev_lsign != lsign and prev_rsign != rsign and ldiff > 0.055 and rdiff > 0.055 and lrdiff > 0.055:
        #                             is_flip = True
        #                             break
                        
        #                 if is_flip:
        #                     flip_fnos.append(fno)
        #                     continue

        #                 for jname, (bone_name, name_list, parent_list, initial_qq, ranges, diff_limits, is_hand, is_head) in VMD_CONNECTIONS.items():
        #                     if name_list is None:
        #                         continue

        #                     if not args.hand_motion == 1 and is_hand:
        #                         # 手トレースは手ON時のみ
        #                         continue
                            
        #                     if args.body_motion == 0 and args.face_motion == 1 and not is_head:
        #                         continue
                            
        #                     # 前のキーフレから大幅に離れていたらスルー
        #                     if prev_fno < fno and jname in all_frame_joints[prev_fno]["joints"]:
        #                         if abs(all_frame_joints[prev_fno]["joints"][jname]["x"] - all_frame_joints[fno]["joints"][jname]["x"]) > 0.05:
        #                             continue

        #                     bf = VmdBoneFrame(fno)
        #                     bf.set_name(bone_name)
                            
        #                     if len(name_list) == 4:
        #                         rotation = calc_direction_qq(bf.fno, motion, frame_joints, *name_list)
        #                         initial = calc_bone_direction_qq(bf, motion, model, jname, *name_list)
        #                     else:
        #                         rotation = calc_direction_qq2(bf.fno, motion, frame_joints, *name_list)
        #                         initial = calc_bone_direction_qq2(bf, motion, model, jname, *name_list)

        #                     qq = MQuaternion()
        #                     for parent_name in reversed(parent_list):
        #                         qq *= motion.calc_bf(parent_name, bf.fno).rotation.inverted()
        #                     bf.rotation = qq * initial_qq * rotation * initial.inverted()

        #                     motion.regist_bf(bf, bf.name, bf.fno, is_key=(not is_flip))
        #                     target_bone_names[bf.name] = diff_limits
                                
        #             if "faces" in frame_joints and args.face_motion == 1:
        #                 # 表情がある場合出力
        #                 # まばたき・視線の向き
        #                 left_eye_euler = calc_left_eye(fno, motion, frame_joints)
        #                 right_eye_euler = calc_right_eye(fno, motion, frame_joints)
        #                 blend_eye(fno, motion, left_eye_euler, right_eye_euler)
        #                 target_bone_names["両目"] = VMD_CONNECTIONS["nose"][5]

        #                 # 口
        #                 calc_lip(fno, motion, frame_joints)

        #                 # 眉
        #                 calc_eyebrow(fno, motion, frame_joints)
                    
        #             prev_fno = fno

        #     start_fno = sorted(all_frame_joints.keys())[0]
        #     last_fno = sorted(all_frame_joints.keys())[-1]
        #     fnos = list(range(start_fno, last_fno + 1))

        #     if args.body_motion == 1:

        #         if args.smooth_key == 1:
        #             logger.info("【No.{0}】スムージング開始", f"{oidx:02d}", decoration=MLogger.DECORATION_LINE)

        #             with tqdm(total=(((len(list(target_bone_names.keys())) - 3) * len(fnos) * 3) + (3 * len(fnos)))) as pchar:
        #                 for bone_name in target_bone_names.keys():
        #                     xvalues = []
        #                     yvalues = []
        #                     zvalues = []

        #                     for fidx, fno in enumerate(fnos):
        #                         pchar.update(1)
        #                         prev_fno, next_fno = motion.get_bone_prev_next_fno(bone_name, fno=fno, is_key=True)

        #                         prev_bf = motion.calc_bf(bone_name, prev_fno)
        #                         next_bf = motion.calc_bf(bone_name, next_fno)

        #                         if fno in flip_fnos or fno not in all_frame_joints:
        #                             # キーフレがないフレームの場合、前後の線形補間
        #                             if fidx == 0:
        #                                 xvalues.append(0)
        #                                 yvalues.append(0)
        #                                 zvalues.append(0)
        #                             else:
        #                                 now_rot = MQuaternion.slerp(prev_bf.rotation, next_bf.rotation, ((fno - prev_fno) / (next_fno - prev_fno)))
        #                                 now_euler = now_rot.toEulerAngles()
        #                                 xvalues.append(now_euler.x())
        #                                 yvalues.append(now_euler.y())
        #                                 zvalues.append(now_euler.z())
        #                             continue

        #                         now_bf = motion.calc_bf(bone_name, fno)

        #                         # 前のキーフレから大きく変化しすぎてる場合、前後の線形補間をコピーしてスルー
        #                         if fidx > 0:
        #                             dot = MQuaternion.dotProduct(now_bf.rotation, prev_bf.rotation)
        #                             if dot < 1 - ((now_bf.fno - prev_bf.fno) * (0.2 if bone_name in ["上半身", "下半身"] else 0.05)):
        #                                 now_rot = MQuaternion.slerp(prev_bf.rotation, next_bf.rotation, ((fno - prev_fno) / (next_fno - prev_fno)))
        #                                 now_euler = now_rot.toEulerAngles()
        #                                 xvalues.append(now_euler.x())
        #                                 yvalues.append(now_euler.y())
        #                                 zvalues.append(now_euler.z())

        #                                 # フリップに相当している場合、キーフレ削除
        #                                 if fno in motion.bones[bone_name]:
        #                                     del motion.bones[bone_name][fno]

        #                                 if bone_name in ["上半身", "下半身"]:
        #                                     # 体幹の場合、フリップに追加
        #                                     flip_fnos.append(fno)

        #                                 continue

        #                         euler = now_bf.rotation.toEulerAngles()
        #                         xvalues.append(euler.x())
        #                         yvalues.append(euler.y())
        #                         zvalues.append(euler.z())

        #                     smooth_xs = []                            
        #                     smooth_ys = []
        #                     smooth_zs = []


        #                     if bone_name in ["上半身2"]:
        #                         # 強制的に平滑化
        #                         smooth_xs = smooth_values(9, xvalues)
        #                         smooth_ys = smooth_values(9, yvalues)
        #                         smooth_zs = smooth_values(9, zvalues)

        #                     elif bone_name in ["上半身", "下半身"]:
        #                         smooth_xs = smooth_values(9, xvalues)
        #                         smooth_zs = smooth_values(9, zvalues)

        #                         # 体幹Yは回転を殺さないようフィルタスムージング
        #                         ryfilter = OneEuroFilter(freq=30, mincutoff=1, beta=0.000000000000001, dcutoff=1)
        #                         smooth_ys = []
        #                         for fidx, fno in enumerate(fnos):
        #                             smooth_ys.append(ryfilter(yvalues[fidx], fno))
        #                             pchar.update(1)

        #                     elif bone_name in ["左手首", "右手首"]:
        #                         # 他は強制的に平滑化
        #                         smooth_xs = smooth_values(17, xvalues)
        #                         smooth_ys = smooth_values(17, yvalues)
        #                         smooth_zs = smooth_values(17, zvalues)

        #                     else:
        #                         rxfilter = OneEuroFilter(freq=30, mincutoff=1, beta=0.000000000000001, dcutoff=1)
        #                         ryfilter = OneEuroFilter(freq=30, mincutoff=1, beta=0.000000000000001, dcutoff=1)
        #                         rzfilter = OneEuroFilter(freq=30, mincutoff=1, beta=0.000000000000001, dcutoff=1)
        #                         for fidx, fno in enumerate(fnos):
        #                             smooth_xs.append(rxfilter(xvalues[fidx], fno))
        #                             smooth_ys.append(ryfilter(yvalues[fidx], fno))
        #                             smooth_zs.append(rzfilter(zvalues[fidx], fno))
        #                             pchar.update(1)

        #                     for fidx, fno in enumerate(fnos):
        #                         # 平滑化したのを登録
        #                         if fno in flip_fnos and fno in motion.bones[bone_name] and bone_name in ["上半身", "下半身"]:
        #                             del motion.bones[bone_name][fno]
        #                         else:
        #                             now_bf = motion.calc_bf(bone_name, fno)
        #                             now_bf.rotation = MQuaternion.fromEulerAngles(smooth_xs[fidx], smooth_ys[fidx], smooth_zs[fidx])
        #                             motion.regist_bf(now_bf, now_bf.name, now_bf.fno, is_key=now_bf.key)
        #                         pchar.update(1)

        #         logger.info("【No.{0}】移動ボーン初期化開始", f"{oidx:02d}", decoration=MLogger.DECORATION_LINE)

        #         for fidx, fno in enumerate(tqdm(fnos, desc=f"{oidx:02d} ... ")):
        #             # 平滑化したのを登録
        #             if fno not in flip_fnos and fno in all_frame_joints:
        #                 # センター・グルーブ・足IKは初期値
        #                 center_bf = VmdBoneFrame(fno)
        #                 center_bf.set_name("センター")
        #                 motion.regist_bf(center_bf, center_bf.name, fno)
        #                 target_bone_names["センター"] = VMD_CONNECTIONS["center"][5]

        #                 groove_bf = VmdBoneFrame(fno)
        #                 groove_bf.set_name("グルーブ")
        #                 motion.regist_bf(groove_bf, groove_bf.name, fno)
        #                 target_bone_names["グルーブ"] = VMD_CONNECTIONS["groove"][5]

        #                 left_leg_ik_bf = VmdBoneFrame(fno)
        #                 left_leg_ik_bf.set_name("左足ＩＫ")
        #                 motion.regist_bf(left_leg_ik_bf, left_leg_ik_bf.name, fno)
        #                 target_bone_names["左足ＩＫ"] = VMD_CONNECTIONS["leg_ik"][5]

        #                 right_leg_ik_bf = VmdBoneFrame(fno)
        #                 right_leg_ik_bf.set_name("右足ＩＫ")
        #                 motion.regist_bf(right_leg_ik_bf, right_leg_ik_bf.name, fno)
        #                 target_bone_names["右足ＩＫ"] = VMD_CONNECTIONS["leg_ik"][5]

        #             if fno in all_frame_joints:
        #                 if fno not in flip_fnos:
        #                     # フリップしてない場合、足の角度
        #                     right_leg_bf = motion.calc_bf("右足", fno)
        #                     right_knee_bf = motion.calc_bf("右ひざ", fno)
        #                     left_leg_bf = motion.calc_bf("左足", fno)
        #                     left_knee_bf = motion.calc_bf("左ひざ", fno)
                            
        #                     total_degree = 0
        #                     right_leg_degree = right_leg_bf.rotation.toDegree()
        #                     total_degree += right_leg_degree if right_leg_degree < 180 else 360 - right_leg_degree
        #                     right_knee_degree = right_knee_bf.rotation.toDegree()
        #                     total_degree += right_knee_degree if right_knee_degree < 180 else 360 - right_knee_degree
        #                     left_leg_degree = left_leg_bf.rotation.toDegree()
        #                     total_degree += left_leg_degree if left_leg_degree < 180 else 360 - left_leg_degree
        #                     left_knee_degree = left_knee_bf.rotation.toDegree()
        #                     total_degree += left_knee_degree if left_knee_degree < 180 else 360 - left_knee_degree
        #                     leg_degrees.append(total_degree)
        #                 else:
        #                     # フリップしてる場合、対象外として最もデカいのを挿入
        #                     leg_degrees.append(99999999)

        #         logger.info("【No.{0}】直立姿勢計算開始", f"{oidx:02d}", decoration=MLogger.DECORATION_LINE)
                
        #         # 足とひざの角度が最も小さい（最も伸びている）を対象とする
        #         degree_fidxs = np.argsort(leg_degrees)
        #         upright_fidx = degree_fidxs[0]
        #         upright_fno = list(all_frame_joints.keys())[upright_fidx]

        #         # 直立キーフレの骨盤は地に足がついているとみなす
        #         upright_pelvis_vec = calc_pelvis_vec(all_frame_joints, upright_fno, args)

        #         logger.info("【No.{0}】直立キーフレ: {1}", f"{oidx:02d}", upright_fno)

        #         # かかと末端までのリンク
        #         right_heel_links = miku_model.create_link_2_top_one("右かかと", is_defined=False)
        #         left_heel_links = miku_model.create_link_2_top_one("左かかと", is_defined=False)

        #         # つま先ＩＫまでのリンク
        #         right_toe_ik_links = miku_model.create_link_2_top_one("右つま先ＩＫ", is_defined=False)
        #         left_toe_ik_links = model.create_link_2_top_one("左つま先ＩＫ", is_defined=False)

        #         logger.info("【No.{0}】センター計算開始", f"{oidx:02d}", decoration=MLogger.DECORATION_LINE)
                
        #         pelvis_xs = []
        #         pelvis_ys = []
        #         pelvis_zs = []
        #         for fidx, fno in enumerate(tqdm(fnos, desc=f"No.{oidx:02d} ... ")):
        #             if fno in flip_fnos or fno not in all_frame_joints:
        #                 # キーフレがないフレームの場合、前のをコピー
        #                 if fidx == 0:
        #                     pelvis_xs.append(0)
        #                     pelvis_ys.append(0)
        #                     pelvis_zs.append(0)
        #                 else:
        #                     pelvis_xs.append(pelvis_xs[-1])
        #                     pelvis_ys.append(pelvis_ys[-1])
        #                     pelvis_zs.append(pelvis_zs[-1])
        #                 continue

        #             pelvis_vec = calc_pelvis_vec(all_frame_joints, fno, args)

        #             if start_z == 9999999999:
        #                 # 最初の人物の深度を0にする
        #                 start_z = pelvis_vec.z()
                    
        #             # Yは先に接地を検討する ----------
        #             now_left_heel_3ds_dic = calc_global_pos(model, left_heel_links, motion, fno)
        #             now_left_toe_vec = now_left_heel_3ds_dic["左つま先"]
        #             now_left_heel_vec = now_left_heel_3ds_dic["左かかと"]
        #             left_foot_vec = (now_left_toe_vec - now_left_heel_vec).normalized()

        #             now_right_heel_3ds_dic = calc_global_pos(model, right_heel_links, motion, fno)
        #             now_right_toe_vec = now_right_heel_3ds_dic["右つま先"]
        #             now_right_heel_vec = now_right_heel_3ds_dic["右かかと"]
        #             right_foot_vec = (now_right_toe_vec - now_right_heel_vec).normalized()

        #             # かかとからつま先の向きが大体水平なら接地
        #             diff_y = 0
        #             if 0.2 > abs(left_foot_vec.y()) and 0.2 > abs(right_foot_vec.y()):
        #                 # 両足水平の場合
        #                 diff_y = min(now_left_toe_vec.y(), now_left_heel_vec.y(), now_right_toe_vec.y(), now_right_heel_vec.y())
        #             elif 0.2 > abs(left_foot_vec.y()):
        #                 # 左足水平の場合
        #                 diff_y = min(now_left_toe_vec.y(), now_left_heel_vec.y())
        #             elif 0.2 > abs(right_foot_vec.y()):
        #                 # 右足水平の場合
        #                 diff_y = min(now_right_toe_vec.y(), now_right_heel_vec.y())

        #             pelvis_xs.append(pelvis_vec.x())
        #             pelvis_ys.append(pelvis_vec.y() - upright_pelvis_vec.y() - diff_y)
        #             pelvis_zs.append(pelvis_vec.z() - start_z)

        #         logger.info("【No.{0}】センター登録開始", f"{oidx:02d}", decoration=MLogger.DECORATION_LINE)

        #         smooth_pelvis_xs = smooth_values(9, pelvis_xs)
        #         smooth_pelvis_ys = smooth_values(11, pelvis_ys)
        #         smooth_pelvis_zs = smooth_values(11, pelvis_zs)

        #         for fidx, fno in enumerate(tqdm(fnos, desc=f"No.{oidx:02d} ... ")):
        #             center_bf = VmdBoneFrame()
        #             center_bf.fno = fno
        #             center_bf.set_name("センター")

        #             # XZはセンター
        #             center_bf.position.setX(smooth_pelvis_xs[fidx])
        #             center_bf.position.setZ(smooth_pelvis_zs[fidx])
        #             motion.regist_bf(center_bf, center_bf.name, fno)

        #             # Yはグルーブ
        #             if args.upper_motion == 0:
        #                 groove_bf = VmdBoneFrame()
        #                 groove_bf.fno = fno
        #                 groove_bf.set_name("グルーブ")
        #                 groove_bf.position.setY(max(-7, smooth_pelvis_ys[fidx]))
        #                 motion.regist_bf(groove_bf, groove_bf.name, fno)

        #         logger.info("【No.{0}】右足IK計算開始", f"{oidx:02d}", decoration=MLogger.DECORATION_LINE)
        #         convert_leg_fk2ik(oidx, all_frame_joints, motion, model, flip_fnos, "右")

        #         logger.info("【No.{0}】左足IK計算開始", f"{oidx:02d}", decoration=MLogger.DECORATION_LINE)
        #         convert_leg_fk2ik(oidx, all_frame_joints, motion, model, flip_fnos, "左")

        #         logger.info("【No.{0}】足IK固定開始", f"{oidx:02d}", decoration=MLogger.DECORATION_LINE)

        #         for fidx, fno in enumerate(tqdm(fnos, desc=f"No.{oidx:02d} ... ")):
        #             prev_fno, _ = motion.get_bone_prev_next_fno("センター", fno=fno, is_key=True)

        #             # 画面内の関節位置から位置調整
        #             check_proj_joints(model, motion, all_frame_joints, right_heel_links, left_heel_links, fidx, fno, prev_fno, True, right_toe_ik_links, left_toe_ik_links)

        #         if args.smooth_key == 1:
        #             logger.info("【No.{0}】足ＩＫスムージング開始", f"{oidx:02d}", decoration=MLogger.DECORATION_LINE)

        #             with tqdm(total=(2 * len(fnos))) as pchar:
        #                 for bone_name in ["左足ＩＫ", "右足ＩＫ"]:
        #                     mxfilter = OneEuroFilter(freq=30, mincutoff=1, beta=0.000000000000001, dcutoff=1)
        #                     myfilter = OneEuroFilter(freq=30, mincutoff=1, beta=0.000000000000001, dcutoff=1)
        #                     mzfilter = OneEuroFilter(freq=30, mincutoff=1, beta=0.000000000000001, dcutoff=1)
        #                     rxfilter = OneEuroFilter(freq=30, mincutoff=1, beta=0.000000000000001, dcutoff=1)
        #                     ryfilter = OneEuroFilter(freq=30, mincutoff=1, beta=0.000000000000001, dcutoff=1)
        #                     rzfilter = OneEuroFilter(freq=30, mincutoff=1, beta=0.000000000000001, dcutoff=1)

        #                     for fidx, fno in enumerate(fnos):
        #                         bf = motion.calc_bf(bone_name, fno)
        #                         pchar.update(1)

        #                         if fno in flip_fnos or fno not in all_frame_joints:
        #                             continue

        #                         if model.bones[bone_name].getRotatable():
        #                             # 回転ありボーンの場合
        #                             euler = bf.rotation.toEulerAngles()
        #                             bf.rotation = MQuaternion.fromEulerAngles(rxfilter(euler.x(), fno), ryfilter(euler.y(), fno), rzfilter(euler.z(), fno))
                                
        #                         if model.bones[bone_name].getTranslatable():
        #                             # 移動ありボーンの場合
        #                             bf.position.setX(mxfilter(bf.position.x(), fno))
        #                             bf.position.setY(myfilter(bf.position.y(), fno))
        #                             bf.position.setZ(mzfilter(bf.position.z(), fno))
                                
        #                         motion.regist_bf(bf, bone_name, fno, is_key=bf.key)

        #     if args.face_motion == 1:
        #         # モーフはキーフレ上限があるので、削除処理を入れておく
        #         logger.info("【No.{0}】モーフスムージング", f"{oidx:02d}", decoration=MLogger.DECORATION_LINE)
        #         for morph_name in tqdm(motion.morphs.keys(), desc=f"No.{oidx:02d} ... "):
        #             motion.smooth_filter_mf(0, morph_name, config={"freq": 30, "mincutoff": 0.05, "beta": 1, "dcutoff": 1})
        
        #         logger.info("【No.{0}】不要モーフ削除処理", f"{oidx:02d}", decoration=MLogger.DECORATION_LINE)
        #         for morph_name in tqdm(motion.morphs.keys(), desc=f"No.{oidx:02d} ... "):
        #             motion.remove_unnecessary_mf(0, morph_name, threshold=0.05)

        #     logger.info("【No.{0}】モーション生成開始", f"{oidx:02d}", decoration=MLogger.DECORATION_LINE)
        #     motion_path = os.path.join(motion_dir_path, "output_{0}_no{1:03}.vmd".format(process_datetime, oidx))
        #     writer = VmdWriter(model, motion, motion_path)
        #     writer.write()

        #     logger.info("【No.{0}】モーション生成終了: {1}", f"{oidx:02d}", motion_path, decoration=MLogger.DECORATION_BOX)

        logger.info('モーション生成処理全件終了', decoration=MLogger.DECORATION_BOX)

        return True
    except Exception as e:
        logger.critical("モーション生成で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        return False

def create_bone(trace_model: PmxModel, jname: str, jconn: dict, target_bone_vecs: dict, miku_model: PmxModel):
    joints = list(target_bone_vecs[jname].values())
    parent_joints = list(target_bone_vecs[jconn["parent"]].values())
    # MMDボーン名
    mname = jconn["mmd"]
    # 親ボーン
    parent_bone = trace_model.bones[PMX_CONNECTIONS[jconn["parent"]]['mmd']] if jconn["parent"] in PMX_CONNECTIONS else trace_model.bones[jconn["parent"]]

    bone_length = np.median(np.linalg.norm(np.array(joints) - np.array(parent_joints), ord=2, axis=1))

    # 親からの相対位置
    # if "指" in jconn["display"]:
    #     # 指は完全にミクに合わせる
    #     bone_relative_pos = miku_model.bones[mname].position - miku_model.bones[parent_bone.name].position
    # elif (mname in ["頭", "首"] or jconn["display"] in ["顔", "眉", "目", "口", "輪郭"]) and mname in miku_model.bones and parent_bone.name in miku_model.bones:
    #     # 頭は方向はミクに合わせる。長さはトレース元
    #     bone_relative_pos = miku_model.bones[mname].position - miku_model.bones[parent_bone.name].position
    #     bone_relative_pos *= bone_length / bone_relative_pos.length()
    # else:
    # トレース元から採取
    bone_axis = MVector3D(np.median(np.array(joints), axis=0) - np.median(np.array(parent_joints), axis=0)).normalized()
    bone_relative_pos = MVector3D(bone_axis * bone_length)
    bone_pos = parent_bone.position + bone_relative_pos
    bone = Bone(mname, mname, bone_pos, parent_bone.index, 0, 0x0000 | 0x0002 | 0x0004 | 0x0008 | 0x0010)
    bone.index = len(list(trace_model.bones.keys()))
    if len(jconn['tail']) > 0:
        bone.tail_index = PMX_CONNECTIONS[jconn["tail"]]['mmd']

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













def check_proj_joints(model: PmxModel, motion: VmdMotion, all_frame_joints: dict, right_heel_links: BoneLinks, left_heel_links: BoneLinks, \
                      fidx: int, fno: int, prev_fno: int, is_leg_ik: bool, right_toe_ik_links: BoneLinks, left_toe_ik_links: BoneLinks):

    # XZ調整 --------------------------
    if is_leg_ik and fidx > 0 and fno in all_frame_joints:
        # 画像サイズからブレ許容量
        # image_max = max(all_frame_joints[fno]["image"]["width"], all_frame_joints[fno]["image"]["height"])
        image_offset = np.array([all_frame_joints[fno]["image"]["width"], all_frame_joints[fno]["image"]["height"]]) * 0.012

        past_avg_right_heel_x = []
        past_avg_right_heel_y = []
        past_avg_right_big_toe_x = []
        past_avg_right_big_toe_y = []
        past_avg_right_small_toe_x = []
        past_avg_right_small_toe_y = []
        past_avg_left_heel_x = []
        past_avg_left_heel_y = []
        past_avg_left_big_toe_x = []
        past_avg_left_big_toe_y = []
        past_avg_left_small_toe_x = []
        past_avg_left_small_toe_y = []

        for f in range(fno - 1, max(0, fno - 11), -1):
            if f in all_frame_joints:
                past_avg_right_heel_x.append(all_frame_joints[f]["proj_joints"]["right_heel"]["x"])
                past_avg_right_heel_y.append(all_frame_joints[f]["proj_joints"]["right_heel"]["y"])
                past_avg_right_big_toe_x.append(all_frame_joints[f]["proj_joints"]["right_big_toe"]["x"])
                past_avg_right_big_toe_y.append(all_frame_joints[f]["proj_joints"]["right_big_toe"]["y"])
                past_avg_right_small_toe_x.append(all_frame_joints[f]["proj_joints"]["right_small_toe"]["x"])
                past_avg_right_small_toe_y.append(all_frame_joints[f]["proj_joints"]["right_small_toe"]["y"])
                past_avg_left_heel_x.append(all_frame_joints[f]["proj_joints"]["left_heel"]["x"])
                past_avg_left_heel_y.append(all_frame_joints[f]["proj_joints"]["left_heel"]["y"])
                past_avg_left_big_toe_x.append(all_frame_joints[f]["proj_joints"]["left_big_toe"]["x"])
                past_avg_left_big_toe_y.append(all_frame_joints[f]["proj_joints"]["left_big_toe"]["y"])
                past_avg_left_small_toe_x.append(all_frame_joints[f]["proj_joints"]["left_small_toe"]["x"])
                past_avg_left_small_toe_y.append(all_frame_joints[f]["proj_joints"]["left_small_toe"]["y"])
    
        past_avg_right_heel_x = np.mean(past_avg_right_heel_x) if len(past_avg_right_heel_x) > 0 else 0
        past_avg_right_heel_y = np.mean(past_avg_right_heel_y) if len(past_avg_right_heel_y) > 0 else 0
        past_avg_right_big_toe_x = np.mean(past_avg_right_big_toe_x) if len(past_avg_right_big_toe_x) > 0 else 0
        past_avg_right_big_toe_y = np.mean(past_avg_right_big_toe_y) if len(past_avg_right_big_toe_y) > 0 else 0
        past_avg_right_small_toe_x = np.mean(past_avg_right_small_toe_x) if len(past_avg_right_small_toe_x) > 0 else 0
        past_avg_right_small_toe_y = np.mean(past_avg_right_small_toe_y) if len(past_avg_right_small_toe_y) > 0 else 0
        past_avg_left_heel_x = np.mean(past_avg_left_heel_x) if len(past_avg_left_heel_x) > 0 else 0
        past_avg_left_heel_y = np.mean(past_avg_left_heel_y) if len(past_avg_left_heel_y) > 0 else 0
        past_avg_left_big_toe_x = np.mean(past_avg_left_big_toe_x) if len(past_avg_left_big_toe_x) > 0 else 0
        past_avg_left_big_toe_y = np.mean(past_avg_left_big_toe_y) if len(past_avg_left_big_toe_y) > 0 else 0
        past_avg_left_small_toe_x = np.mean(past_avg_left_small_toe_x) if len(past_avg_left_small_toe_x) > 0 else 0
        past_avg_left_small_toe_y = np.mean(past_avg_left_small_toe_y) if len(past_avg_left_small_toe_y) > 0 else 0

        # 2F目以降は、前Fからの移動量を計算する
        # 右かかと
        right_heel_x_diff = abs(all_frame_joints[fno]["proj_joints"]["right_heel"]["x"] - past_avg_right_heel_x)
        right_heel_y_diff = abs(all_frame_joints[fno]["proj_joints"]["right_heel"]["y"] - past_avg_right_heel_y)
        # 右足親指
        right_big_toe_x_diff = abs(all_frame_joints[fno]["proj_joints"]["right_big_toe"]["x"] - past_avg_right_big_toe_x)
        right_big_toe_y_diff = abs(all_frame_joints[fno]["proj_joints"]["right_big_toe"]["y"] - past_avg_right_big_toe_y)
        # 右足小指
        right_small_toe_x_diff = abs(all_frame_joints[fno]["proj_joints"]["right_small_toe"]["x"] - past_avg_right_small_toe_x)
        right_small_toe_y_diff = abs(all_frame_joints[fno]["proj_joints"]["right_small_toe"]["y"] - past_avg_right_small_toe_y)
        # 左かかと
        left_heel_x_diff = abs(all_frame_joints[fno]["proj_joints"]["left_heel"]["x"] - past_avg_left_heel_x)
        left_heel_y_diff = abs(all_frame_joints[fno]["proj_joints"]["left_heel"]["y"] - past_avg_left_heel_y)
        # 左足親指
        left_big_toe_x_diff = abs(all_frame_joints[fno]["proj_joints"]["left_big_toe"]["x"] - past_avg_left_big_toe_x)
        left_big_toe_y_diff = abs(all_frame_joints[fno]["proj_joints"]["left_big_toe"]["y"] - past_avg_left_big_toe_y)
        # 左足小指
        left_small_toe_x_diff = abs(all_frame_joints[fno]["proj_joints"]["left_small_toe"]["x"] - past_avg_left_small_toe_x)
        left_small_toe_y_diff = abs(all_frame_joints[fno]["proj_joints"]["left_small_toe"]["y"] - past_avg_left_small_toe_y)
        
        is_fix_right = False

        if right_heel_x_diff < image_offset[0] and right_heel_y_diff < image_offset[1]:
            # 右足ＩＫ固定
            now_right_leg_ik_bf = motion.calc_bf("右足ＩＫ", fno)
            prev_right_leg_ik_bf = motion.calc_bf("右足ＩＫ", prev_fno)
            diff_right_leg_vec = prev_right_leg_ik_bf.position - now_right_leg_ik_bf.position
            now_right_leg_ik_bf.position += diff_right_leg_vec
            motion.regist_bf(now_right_leg_ik_bf, now_right_leg_ik_bf.name, fno)

            # センター調整
            now_center_bf = motion.calc_bf("センター", fno)
            now_center_bf.position.setZ(now_center_bf.position.z() + diff_right_leg_vec.z())
            motion.regist_bf(now_center_bf, now_center_bf.name, fno)

            # 左足ＩＫ調整
            now_left_leg_ik_bf = motion.calc_bf("左足ＩＫ", fno)
            now_left_leg_ik_bf.position.setZ(now_left_leg_ik_bf.position.z() + diff_right_leg_vec.z())
            motion.regist_bf(now_left_leg_ik_bf, now_left_leg_ik_bf.name, fno)

            is_fix_right = True

        elif (right_big_toe_x_diff < image_offset[0] and right_big_toe_y_diff < image_offset[1]) or \
                (right_small_toe_x_diff < image_offset[0] and right_small_toe_y_diff < image_offset[1]):
            # 右つま先ＩＫ固定
            now_right_toe_ik_3ds_dic = calc_global_pos(model, right_toe_ik_links, motion, fno)
            prev_right_toe_ik_3ds_dic = calc_global_pos(model, right_toe_ik_links, motion, prev_fno)
            diff_right_toe_vec = prev_right_toe_ik_3ds_dic["右つま先ＩＫ"] - now_right_toe_ik_3ds_dic["右つま先ＩＫ"]

            now_right_leg_ik_bf = motion.calc_bf("右足ＩＫ", fno)
            now_right_leg_ik_bf.position += diff_right_toe_vec
            motion.regist_bf(now_right_leg_ik_bf, now_right_leg_ik_bf.name, fno)

            # センター調整
            now_center_bf = motion.calc_bf("センター", fno)
            now_center_bf.position.setZ(now_center_bf.position.z() + diff_right_toe_vec.z())
            motion.regist_bf(now_center_bf, now_center_bf.name, fno)

            # 左足ＩＫ調整
            now_left_leg_ik_bf = motion.calc_bf("左足ＩＫ", fno)
            now_left_leg_ik_bf.position.setZ(now_left_leg_ik_bf.position.z() + diff_right_toe_vec.z())
            motion.regist_bf(now_left_leg_ik_bf, now_left_leg_ik_bf.name, fno)

            is_fix_right = True

        if left_heel_x_diff < image_offset[0] and left_heel_y_diff < image_offset[1]:
            # 左足ＩＫ固定
            now_left_leg_ik_bf = motion.calc_bf("左足ＩＫ", fno)
            prev_left_leg_ik_bf = motion.calc_bf("左足ＩＫ", prev_fno)
            diff_left_leg_vec = prev_left_leg_ik_bf.position - now_left_leg_ik_bf.position
            now_left_leg_ik_bf.position += diff_left_leg_vec
            motion.regist_bf(now_left_leg_ik_bf, now_left_leg_ik_bf.name, fno)

            # センター調整
            now_center_bf = motion.calc_bf("センター", fno)
            now_center_bf.position.setZ(now_center_bf.position.z() + diff_left_leg_vec.z())
            motion.regist_bf(now_center_bf, now_center_bf.name, fno)

            if not is_fix_right:
                # 右足ＩＫ調整
                now_right_leg_ik_bf = motion.calc_bf("右足ＩＫ", fno)
                now_right_leg_ik_bf.position.setZ(now_right_leg_ik_bf.position.z() + diff_left_leg_vec.z())
                motion.regist_bf(now_right_leg_ik_bf, now_right_leg_ik_bf.name, fno)

        elif (left_big_toe_x_diff < image_offset[0] and left_big_toe_y_diff < image_offset[1]) or \
                (left_small_toe_x_diff < image_offset[0] and left_small_toe_y_diff < image_offset[1]):
            # 左つま先ＩＫ固定
            now_left_toe_ik_3ds_dic = calc_global_pos(model, left_toe_ik_links, motion, fno)
            prev_left_toe_ik_3ds_dic = calc_global_pos(model, left_toe_ik_links, motion, prev_fno)
            diff_left_toe_vec = prev_left_toe_ik_3ds_dic["左つま先ＩＫ"] - now_left_toe_ik_3ds_dic["左つま先ＩＫ"]

            now_left_leg_ik_bf = motion.calc_bf("左足ＩＫ", fno)
            now_left_leg_ik_bf.position += diff_left_toe_vec
            motion.regist_bf(now_left_leg_ik_bf, now_left_leg_ik_bf.name, fno)

            # センター調整
            now_center_bf = motion.calc_bf("センター", fno)
            now_center_bf.position.setZ(now_center_bf.position.z() + diff_left_toe_vec.z())
            motion.regist_bf(now_center_bf, now_center_bf.name, fno)

            if not is_fix_right:
                # 右足ＩＫ調整
                now_right_leg_ik_bf = motion.calc_bf("右足ＩＫ", fno)
                now_right_leg_ik_bf.position.setZ(now_right_leg_ik_bf.position.z() + diff_left_toe_vec.z())
                motion.regist_bf(now_right_leg_ik_bf, now_right_leg_ik_bf.name, fno)


def calc_pelvis_vec(frame_joints: dict, fno: int, args):
    # 画像サイズ
    image_size = np.array([frame_joints["image"]["width"], frame_joints["image"]["height"]])

    # カメラ中央
    camera_center_pos = np.array([frame_joints["others"]["center"]["x"], frame_joints["others"]["center"]["y"]])
    # カメラ倍率
    camera_scale = frame_joints["camera"]["scale"]
    # センサー幅（画角？）
    sensor_width = frame_joints["others"]["sensor_width"]
    # フォーカスのpx単位
    focal_length_in_px = frame_joints["others"]["focal_length_in_px"]
    # camera_center_pos -= image_size / 2
    # Zはカメラ深度
    depth = frame_joints["depth"]["depth"]
    pelvis_z = depth * args.center_scale

    # # pxからmmへのスケール変換
    # mm_scale = image_size[0] * sensor_width

    #bbox
    # bbox_pos = np.array([frame_joints["bbox"]["x"], frame_joints["bbox"]["y"]])
    bbox_size = np.array([frame_joints["bbox"]["width"], frame_joints["bbox"]["height"]])

    # # 骨盤のカメラの中心からの相対位置
    # pelvis_vec = (get_vec3(frame_joints["joints"], "pelvis") + get_vec3(frame_joints["joints"], "spine1")) / 2
    # # pelvis_pos = np.array([(center_vec.x() / 2) + 0.5, (center_vec.y() / -2) + 0.5])
    # # bbox左上を原点とした相対位置
    # pelvis_pos = np.array([(pelvis_vec.x() / 2) + 0.5, (pelvis_vec.y() / -2) + 0.5])

    # 骨盤の画面内グローバル位置(骨盤と脊椎の間）)
    pelvis_global_pos = np.array([np.average([frame_joints["proj_joints"]["pelvis"]["x"], frame_joints["proj_joints"]["spine1"]["x"]]), \
                                  np.average([frame_joints["proj_joints"]["pelvis"]["y"], frame_joints["proj_joints"]["spine1"]["y"]])])
    pelvis_global_pos[0] -= image_size[0] / 2
    pelvis_global_pos[1] = -pelvis_global_pos[1]

    # 足の画面内グローバル位置(X: 骨盤と脊椎の間、Y: つま先・踵の大きい方（地面に近い方）)
    foot_global_pos = np.array([np.average([frame_joints["proj_joints"]["pelvis"]["x"], frame_joints["proj_joints"]["spine1"]["x"]]), \
                                  np.max([frame_joints["proj_joints"]["right_big_toe"]["y"], frame_joints["proj_joints"]["right_heel"]["y"], \
                                          frame_joints["proj_joints"]["left_big_toe"]["y"], frame_joints["proj_joints"]["left_heel"]["y"]])])
    foot_global_pos[0] -= image_size[0] / 2
    foot_global_pos[1] = -foot_global_pos[1]

    # モデル座標系
    model_view = create_model_view(camera_center_pos, camera_scale, image_size, depth, focal_length_in_px)
    # プロジェクション座標系
    projection_view = create_projection_view(image_size, sensor_width, focal_length_in_px)

    # viewport
    viewport_rect = MRect(0, 0, args.center_scale * 7, args.center_scale * 7)

    # 逆プロジェクション座標位置
    pelvis_project_vec = MVector3D(pelvis_global_pos[0], pelvis_global_pos[1], 1)
    pelvis_global_vec = pelvis_project_vec.unproject(model_view, projection_view, viewport_rect)
    # pelvis_global_vec.setX(pelvis_global_vec.x() - (image_size[0] / 2))
    # pelvis_global_vec /= mmd_scale_vec

    # foot_project_vec = MVector3D(foot_global_pos[0], foot_global_pos[1], 1)
    # foot_global_vec = foot_project_vec.unproject(model_view, projection_view, viewport_rect)
    # foot_global_vec.setX(foot_global_vec.x() - (image_size[0] / 2))
    # foot_global_vec /= mmd_scale_vec
    
    # camera_scale * 
    # upper_bf = motion.calc_bf("上半身", fno)
    # upper_x_qq, _, _, _ = separate_local_qq(upper_bf.fno, upper_bf.name, upper_bf.rotation, upper_bf.rotation * MVector3D(1, 0, 0))

    # upper_dot = MQuaternion.dotProduct(MQuaternion(), upper_x_qq)
    # pelvis_mmd_vec = MVector3D(pelvis_global_vec.x(), pelvis_global_vec.y(), pelvis_z * math.sin(abs(upper_dot)))
    # logger.debug("calc_pelvis_vec fno: {0}, pelvis_z: {1}, upper_x_qq: {2}, upper_dot: {3}, result: {4}", fno, pelvis_z, upper_x_qq.toDegree(), upper_dot, pelvis_mmd_vec.z())

    pelvis_mmd_vec = MVector3D(pelvis_global_vec.x(), pelvis_global_vec.y(), pelvis_z)
    logger.debug("calc_pelvis_vec fno: {0}, pelvis_z: {1}, camera_scale: {2}, depth: {3}, bbox_size: {4}, {5}", fno, pelvis_z, camera_scale, depth, bbox_size[0], bbox_size[1])

    return pelvis_mmd_vec


# モデル座標系作成
def create_model_view(camera_center_pos: list, camera_scale: float, image_size: list, depth: float, focal_length_in_px: float):
    # モデル座標系（原点を見るため、単位行列）
    model_view = MMatrix4x4()
    model_view.setToIdentity()   

    camera_eye = MVector3D(image_size[0] / 2, -(image_size[1] / 2), focal_length_in_px)
    camera_pos = MVector3D(camera_center_pos[0], -camera_center_pos[1], depth)

    # カメラの注視点（グローバル座標）
    mat_pos = MMatrix4x4()
    mat_pos.setToIdentity()
    mat_pos.translate(camera_pos)
    mat_pos.scale(camera_scale)
    camera_center = mat_pos * MVector3D()

    # カメラ座標系の行列
    # eye: カメラの原点（グローバル座標）
    # center: カメラの注視点（グローバル座標）
    # up: カメラの上方向ベクトル
    model_view.lookAt(camera_eye, camera_center, MVector3D(0, 1, 0))

    return model_view

# プロジェクション座標系作成
def create_projection_view(image_size: list, sensor_width: float, focal_length_in_px: float):
    mat = MMatrix4x4()
    mat.setToIdentity()
    mat.perspective(sensor_width, image_size[0] / image_size[1], 0, focal_length_in_px)

    return mat

# # 足ＩＫ変換処理実行
# def convert_leg_fk2ik(oidx: int, all_frame_joints: dict, motion: VmdMotion, model: PmxModel, flip_fnos: list, direction: str):
#     leg_ik_bone_name = "{0}足ＩＫ".format(direction)
#     toe_ik_bone_name = "{0}つま先ＩＫ".format(direction)
#     leg_bone_name = "{0}足".format(direction)
#     knee_bone_name = "{0}ひざ".format(direction)
#     ankle_bone_name = "{0}足首".format(direction)
#     toe_bone_name = "{0}つま先".format(direction)
#     heel_bone_name = "{0}かかと".format(direction)

#     # 足FK末端までのリンク
#     fk_links = model.create_link_2_top_one(heel_bone_name, is_defined=False)
#     # 足IK末端までのリンク
#     ik_links = model.create_link_2_top_one(leg_ik_bone_name, is_defined=False)
#     # つま先IK末端までのリンク
#     toe_ik_links = model.create_link_2_top_one(toe_ik_bone_name, is_defined=False)
#     # つま先（足首の子ボーン）の名前
#     ankle_child_bone_name = model.bone_indexes[model.bones[toe_ik_bone_name].ik.target_index]
#     # つま先末端までのリンク
#     toe_fk_links = model.create_link_2_top_one(ankle_child_bone_name, is_defined=False)

#     fnos = motion.get_bone_fnos(leg_bone_name, knee_bone_name, ankle_bone_name)

#     ik_parent_name = ik_links.get(leg_ik_bone_name, offset=-1).name
    
#     # 足IKの移植
#     # logger.info("【No.{0}】{1}足IK移植", f"{oidx:02d}", direction)
#     fno = 0
#     for fidx, fno in enumerate(tqdm(fnos, desc=f"No.{oidx:02d} ... ")):
#         if fno in flip_fnos or fno not in all_frame_joints:
#             # フリップはセンター計算もおかしくなるのでスキップ
#             continue

#         leg_fk_3ds_dic = calc_global_pos(model, fk_links, motion, fno)
#         _, leg_ik_matrixs = calc_global_pos(model, ik_links, motion, fno, return_matrix=True)

#         # 足首の角度がある状態での、つま先までのグローバル位置
#         leg_toe_fk_3ds_dic = calc_global_pos(model, toe_fk_links, motion, fno)

#         # IKの親から見た相対位置
#         leg_ik_parent_matrix = leg_ik_matrixs[ik_parent_name]

#         bf = motion.calc_bf(leg_ik_bone_name, fno)
#         # 足ＩＫの位置は、足ＩＫの親から見た足首のローカル位置（足首位置マイナス）
#         bf.position = leg_ik_parent_matrix.inverted() * (leg_fk_3ds_dic[ankle_bone_name] - (model.bones[ankle_bone_name].position - model.bones[ik_parent_name].position))
#         bf.rotation = MQuaternion()

#         # 一旦足ＩＫの位置が決まった時点で登録
#         motion.regist_bf(bf, leg_ik_bone_name, fno)
#         # 足ＩＫ回転なし状態でのつま先までのグローバル位置
#         leg_ik_3ds_dic, leg_ik_matrisxs = calc_global_pos(model, toe_ik_links, motion, fno, return_matrix=True)

#         # つま先のローカル位置
#         ankle_child_initial_local_pos = leg_ik_matrisxs[leg_ik_bone_name].inverted() * leg_ik_3ds_dic[toe_ik_bone_name]
#         ankle_child_local_pos = leg_ik_matrisxs[leg_ik_bone_name].inverted() * leg_toe_fk_3ds_dic[ankle_child_bone_name]

#         # 足ＩＫの回転は、足首から見たつま先の方向
#         bf.rotation = MQuaternion.rotationTo(ankle_child_initial_local_pos, ankle_child_local_pos)

#         toe_vec = leg_fk_3ds_dic[toe_bone_name]
#         heel_vec = leg_fk_3ds_dic[heel_bone_name]
#         foot_vec = (toe_vec - heel_vec).normalized()

#         # かかとからつま先の向きが大体水平なら接地
#         if 0.2 > abs(foot_vec.y()):
#             diff_y = max(0, min(toe_vec.y(), heel_vec.y()))
#             bf.position.setY(max(0, bf.position.y() - diff_y))
#         motion.regist_bf(bf, leg_ik_bone_name, fno)

#         # つま先が地面にめり込んでたら上げる
#         leg_ik_3ds_dic = calc_global_pos(model, toe_ik_links, motion, fno)
#         if leg_ik_3ds_dic[toe_ik_bone_name].y() < 0:
#             bf.position.setY(max(0, bf.position.y() - leg_ik_3ds_dic[toe_ik_bone_name].y()))
#         motion.regist_bf(bf, leg_ik_bone_name, fno)

# def calc_direction_qq(bf: VmdBoneFrame, motion: VmdMotion, joints: dict, direction_from_name: str, direction_to_name: str, up_from_name: str, up_to_name: str):
#     direction_from_vec = get_vec3(joints["joints"], direction_from_name)
#     direction_to_vec = get_vec3(joints["joints"], direction_to_name)
#     up_from_vec = get_vec3(joints["joints"], up_from_name)
#     up_to_vec = get_vec3(joints["joints"], up_to_name)

#     direction = (direction_to_vec - direction_from_vec).normalized()
#     up = (up_to_vec - up_from_vec).normalized()
#     cross = MVector3D.crossProduct(direction, up)
#     qq = MQuaternion.fromDirection(direction, cross)

#     return qq

# def calc_direction_qq2(bf: VmdBoneFrame, motion: VmdMotion, joints: dict, direction_from_name: str, direction_to_name: str, up_from_name: str, up_to_name: str, cross_from_name: str, cross_to_name: str):
#     direction_from_vec = get_vec3(joints["joints"], direction_from_name)
#     direction_to_vec = get_vec3(joints["joints"], direction_to_name)
#     up_from_vec = get_vec3(joints["joints"], up_from_name)
#     up_to_vec = get_vec3(joints["joints"], up_to_name)
#     cross_from_vec = get_vec3(joints["joints"], cross_from_name)
#     cross_to_vec = get_vec3(joints["joints"], cross_to_name)

#     direction = (direction_to_vec - direction_from_vec).normalized()
#     up = (up_to_vec - up_from_vec).normalized()
#     cross = (cross_to_vec - cross_from_vec).normalized()
#     qq = MQuaternion.fromDirection(direction, MVector3D.crossProduct(up, cross))

#     return qq

# def calc_bone_direction_qq(model: PmxModel, direction_from_name: str, direction_to_name: str, up_from_name: str, up_to_name: str):
#     direction_from_vec = get_bone_vec3(model, direction_from_name)
#     direction_to_vec = get_bone_vec3(model, direction_to_name)
#     up_from_vec = get_bone_vec3(model, up_from_name)
#     up_to_vec = get_bone_vec3(model, up_to_name)

#     direction = (direction_to_vec - direction_from_vec).normalized()
#     up = (up_to_vec - up_from_vec).normalized()
#     qq = MQuaternion.fromDirection(direction, up)

#     return qq

# def calc_bone_direction_qq2(bf: VmdBoneFrame, motion: VmdMotion, model: PmxModel, jname: str, direction_from_name: str, direction_to_name: str, up_from_name: str, up_to_name: str, cross_from_name: str, cross_to_name: str):
#     direction_from_vec = get_bone_vec3(model, direction_from_name)
#     direction_to_vec = get_bone_vec3(model, direction_to_name)
#     up_from_vec = get_bone_vec3(model, up_from_name)
#     up_to_vec = get_bone_vec3(model, up_to_name)
#     cross_from_vec = get_bone_vec3(model, cross_from_name)
#     cross_to_vec = get_bone_vec3(model, cross_to_name)

#     direction = (direction_to_vec - direction_from_vec).normalized()
#     up = (up_to_vec - up_from_vec).normalized()
#     cross = (cross_to_vec - cross_from_vec).normalized()
#     qq = MQuaternion.fromDirection(direction, MVector3D.crossProduct(up, cross))

#     return qq

# def get_bone_vec3(model: PmxModel, joint_name: str):
#     bone_name = VMD_CONNECTIONS[joint_name]["bname"]
#     if bone_name in model.bones:
#         return model.bones[bone_name].position
    
#     return MVector3D()

# def get_vec3(joints: dict, jname: str):
#     if jname in joints:
#         joint = joints[jname]
#         return MVector3D(joint["x"], joint["y"], joint["z"])
#     else:
#         if jname == "spine1":
#             # 尾てい骨くらい
#             right_hip_vec = get_vec3(joints, "right_hip")
#             left_hip_vec = get_vec3(joints, "left_hip")
#             return (right_hip_vec + left_hip_vec) / 2
    
#     return MVector3D()


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


# # 眉モーフ
# def calc_eyebrow(fno: int, motion: VmdMotion, frame_joints: dict):
#     left_eye_brow1 = get_vec2(frame_joints["faces"], "left_eye_brow1")
#     left_eye_brow2 = get_vec2(frame_joints["faces"], "left_eye_brow2")
#     left_eye_brow3 = get_vec2(frame_joints["faces"], "left_eye_brow3")
#     left_eye_brow4 = get_vec2(frame_joints["faces"], "left_eye_brow4")
#     left_eye_brow5 = get_vec2(frame_joints["faces"], "left_eye_brow5")

#     left_eye1 = get_vec2(frame_joints["faces"], "left_eye1")
#     left_eye2 = get_vec2(frame_joints["faces"], "left_eye2")
#     left_eye3 = get_vec2(frame_joints["faces"], "left_eye3")
#     left_eye4 = get_vec2(frame_joints["faces"], "left_eye4")
#     left_eye5 = get_vec2(frame_joints["faces"], "left_eye5")
#     left_eye6 = get_vec2(frame_joints["faces"], "left_eye6")

#     right_eye_brow1 = get_vec2(frame_joints["faces"], "right_eye_brow1")
#     right_eye_brow2 = get_vec2(frame_joints["faces"], "right_eye_brow2")
#     right_eye_brow3 = get_vec2(frame_joints["faces"], "right_eye_brow3")
#     right_eye_brow4 = get_vec2(frame_joints["faces"], "right_eye_brow4")
#     right_eye_brow5 = get_vec2(frame_joints["faces"], "right_eye_brow5")

#     right_eye1 = get_vec2(frame_joints["faces"], "right_eye1")
#     right_eye2 = get_vec2(frame_joints["faces"], "right_eye2")
#     right_eye3 = get_vec2(frame_joints["faces"], "right_eye3")
#     right_eye4 = get_vec2(frame_joints["faces"], "right_eye4")
#     right_eye5 = get_vec2(frame_joints["faces"], "right_eye5")
#     right_eye6 = get_vec2(frame_joints["faces"], "right_eye6")

#     left_nose_1 = get_vec2(frame_joints["faces"], 'left_nose_1')
#     right_nose_2 = get_vec2(frame_joints["faces"], 'right_nose_2')

#     # 鼻の幅
#     nose_width = abs(left_nose_1.x() - right_nose_2.x())
    
#     # 眉のしかめ具合
#     frown_ratio = abs(left_eye_brow1.x() - right_eye_brow1.x()) / nose_width

#     # 眉の幅
#     eye_brow_length = (euclidean_distance(left_eye_brow1, left_eye_brow5) + euclidean_distance(right_eye_brow1, right_eye_brow5)) / 2
#     # 目の幅
#     eye_length = (euclidean_distance(left_eye1, left_eye4) + euclidean_distance(right_eye1, right_eye4)) / 2

#     # 目と眉の縦幅
#     left_vertical_length = (euclidean_distance(left_eye1, left_eye_brow1) + euclidean_distance(right_eye1, right_eye_brow1)) / 2
#     center_vertical_length = (euclidean_distance(left_eye2, left_eye_brow3) + euclidean_distance(right_eye2, right_eye_brow3)) / 2
#     right_vertical_length = (euclidean_distance(left_eye4, left_eye_brow5) + euclidean_distance(right_eye4, right_eye_brow5)) / 2

#     left_ratio = left_vertical_length / eye_brow_length
#     center_ratio = center_vertical_length / eye_brow_length
#     right_ratio = right_vertical_length / eye_brow_length

#     updown_ratio = center_ratio - 0.5

#     if updown_ratio >= 0.2:
#         # 上
#         mf = VmdMorphFrame(fno)
#         mf.set_name("上")
#         mf.ratio = max(0, min(1, abs(updown_ratio) + 0.3))
#         motion.regist_mf(mf, mf.name, mf.fno)

#         mf = VmdMorphFrame(fno)
#         mf.set_name("下")
#         mf.ratio = 0
#         motion.regist_mf(mf, mf.name, mf.fno)
#     else:
#         # 下
#         mf = VmdMorphFrame(fno)
#         mf.set_name("下")
#         mf.ratio = max(0, min(1, abs(updown_ratio) + 0.3))
#         motion.regist_mf(mf, mf.name, mf.fno)

#         mf = VmdMorphFrame(fno)
#         mf.set_name("上")
#         mf.ratio = 0
#         motion.regist_mf(mf, mf.name, mf.fno)
    
#     mf = VmdMorphFrame(fno)
#     mf.set_name("困る")
#     mf.ratio = max(0, min(1, (0.8 - frown_ratio)))
#     motion.regist_mf(mf, mf.name, mf.fno)

#     if left_ratio >= right_ratio:
#         # 怒る系
#         mf = VmdMorphFrame(fno)
#         mf.set_name("怒り")
#         mf.ratio = max(0, min(1, abs(left_ratio - right_ratio)))
#         motion.regist_mf(mf, mf.name, mf.fno)

#         mf = VmdMorphFrame(fno)
#         mf.set_name("にこり")
#         mf.ratio = 0
#         motion.regist_mf(mf, mf.name, mf.fno)
#     else:
#         # 笑う系
#         mf = VmdMorphFrame(fno)
#         mf.set_name("にこり")
#         mf.ratio = max(0, min(1, abs(right_ratio - left_ratio)))
#         motion.regist_mf(mf, mf.name, mf.fno)

#         mf = VmdMorphFrame(fno)
#         mf.set_name("怒り")
#         mf.ratio = 0
#         motion.regist_mf(mf, mf.name, mf.fno)

# # リップモーフ
# def calc_lip(fno: int, motion: VmdMotion, frame_joints: dict):
#     left_nose_1 = get_vec2(frame_joints["faces"], 'left_nose_1')
#     right_nose_2 = get_vec2(frame_joints["faces"], 'right_nose_2')

#     left_mouth_1 = get_vec2(frame_joints["faces"], 'left_mouth_1')
#     left_mouth_2 = get_vec2(frame_joints["faces"], 'left_mouth_2')
#     left_mouth_3 = get_vec2(frame_joints["faces"], 'left_mouth_3')
#     mouth_top = get_vec2(frame_joints["faces"], 'mouth_top')
#     right_mouth_3 = get_vec2(frame_joints["faces"], 'right_mouth_3')
#     right_mouth_2 = get_vec2(frame_joints["faces"], 'right_mouth_2')
#     right_mouth_1 = get_vec2(frame_joints["faces"], 'right_mouth_1')
#     right_mouth_5 = get_vec2(frame_joints["faces"], 'right_mouth_5')
#     right_mouth_4 = get_vec2(frame_joints["faces"], 'right_mouth_4')
#     mouth_bottom = get_vec2(frame_joints["faces"], 'mouth_bottom')
#     left_mouth_4 = get_vec2(frame_joints["faces"], 'left_mouth_4')
#     left_mouth_5 = get_vec2(frame_joints["faces"], 'left_mouth_5')
#     left_lip_1 = get_vec2(frame_joints["faces"], 'left_lip_1')
#     left_lip_2 = get_vec2(frame_joints["faces"], 'left_lip_2')
#     lip_top = get_vec2(frame_joints["faces"], 'lip_top')
#     right_lip_2 = get_vec2(frame_joints["faces"], 'right_lip_2')
#     right_lip_1 = get_vec2(frame_joints["faces"], 'right_lip_1')
#     right_lip_3 = get_vec2(frame_joints["faces"], 'right_lip_3')
#     lip_bottom = get_vec2(frame_joints["faces"], 'lip_bottom')
#     left_lip_3 = get_vec2(frame_joints["faces"], 'left_lip_3')

#     # 鼻の幅
#     nose_width = abs(left_nose_1.x() - right_nose_2.x())

#     # 口角の平均値
#     corner_center = (left_mouth_1 + right_mouth_1) / 2
#     # 口角の幅
#     mouse_width = abs(left_mouth_1.x() - right_mouth_1.x())

#     # 鼻基準の口の横幅比率
#     mouse_width_ratio = mouse_width / nose_width

#     # 上唇の平均値
#     top_mouth_center = (right_mouth_3 + mouth_top + left_mouth_3) / 3
#     top_lip_center = (right_lip_2 + lip_top + left_lip_2) / 3

#     # 下唇の平均値
#     bottom_mouth_center = (right_mouth_4 + mouth_bottom + left_mouth_4) / 3
#     bottom_lip_center = (right_lip_3 + lip_bottom + left_lip_3) / 3

#     # 唇の外側の開き具合に対する内側の開き具合
#     open_ratio = (bottom_lip_center.y() - top_lip_center.y()) / (bottom_mouth_center.y() - top_mouth_center.y())

#     # 笑いの比率
#     smile_ratio = (bottom_mouth_center.y() - corner_center.y()) / (bottom_mouth_center.y() - top_mouth_center.y())

#     if smile_ratio >= 0:
#         mf = VmdMorphFrame(fno)
#         mf.set_name("∧")
#         mf.ratio = 0
#         motion.regist_mf(mf, mf.name, mf.fno)

#         mf = VmdMorphFrame(fno)
#         mf.set_name("にやり")
#         mf.ratio = max(0, min(1, abs(smile_ratio)))
#         motion.regist_mf(mf, mf.name, mf.fno)
#     else:
#         mf = VmdMorphFrame(fno)
#         mf.set_name("∧")
#         mf.ratio = max(0, min(1, abs(smile_ratio)))
#         motion.regist_mf(mf, mf.name, mf.fno)

#         mf = VmdMorphFrame(fno)
#         mf.set_name("にやり")
#         mf.ratio = 0
#         motion.regist_mf(mf, mf.name, mf.fno)

#     if mouse_width_ratio > 1.3:
#         # 横幅広がってる場合は「い」
#         mf = VmdMorphFrame(fno)
#         mf.set_name("い")
#         mf.ratio = max(0, min(1, 1.5 / mouse_width_ratio) * open_ratio)
#         motion.regist_mf(mf, mf.name, mf.fno)

#         mf = VmdMorphFrame(fno)
#         mf.set_name("う")
#         mf.ratio = 0
#         motion.regist_mf(mf, mf.name, mf.fno)

#         mf = VmdMorphFrame(fno)
#         mf.set_name("あ")
#         mf.ratio = max(0, min(1 - min(0.7, smile_ratio), open_ratio))
#         motion.regist_mf(mf, mf.name, mf.fno)

#         mf = VmdMorphFrame(fno)
#         mf.set_name("お")
#         mf.ratio = 0
#         motion.regist_mf(mf, mf.name, mf.fno)
#     else:
#         # 狭まっている場合は「う」
#         mf = VmdMorphFrame(fno)
#         mf.set_name("い")
#         mf.ratio = 0
#         motion.regist_mf(mf, mf.name, mf.fno)

#         mf = VmdMorphFrame(fno)
#         mf.set_name("う")
#         mf.ratio = max(0, min(1, (1.2 / mouse_width_ratio) * open_ratio))
#         motion.regist_mf(mf, mf.name, mf.fno)

#         mf = VmdMorphFrame(fno)
#         mf.set_name("あ")
#         mf.ratio = 0
#         motion.regist_mf(mf, mf.name, mf.fno)        

#         mf = VmdMorphFrame(fno)
#         mf.set_name("お")
#         mf.ratio = max(0, min(1 - min(0.7, smile_ratio), open_ratio))
#         motion.regist_mf(mf, mf.name, mf.fno)

# def blend_eye(fno: int, motion: VmdMotion, left_eye_euler: MVector3D, right_eye_euler: MVector3D):
#     min_blink = min(motion.morphs["ウィンク右"][fno].ratio, motion.morphs["ウィンク"][fno].ratio)
#     min_smile = min(motion.morphs["ｳｨﾝｸ２右"][fno].ratio, motion.morphs["ウィンク２"][fno].ratio)

#     # 両方の同じ値はさっぴく
#     motion.morphs["ウィンク右"][fno].ratio -= min_smile
#     motion.morphs["ウィンク"][fno].ratio -= min_smile

#     motion.morphs["ｳｨﾝｸ２右"][fno].ratio -= min_blink
#     motion.morphs["ウィンク２"][fno].ratio -= min_blink

#     for morph_name in ["ウィンク右", "ウィンク", "ｳｨﾝｸ２右", "ウィンク２"]:
#         motion.morphs[morph_name][fno].ratio = max(0, min(1, motion.morphs[morph_name][fno].ratio))

#     mf = VmdMorphFrame(fno)
#     mf.set_name("笑い")
#     mf.ratio = max(0, min(1, min_smile))
#     motion.regist_mf(mf, mf.name, mf.fno)

#     mf = VmdMorphFrame(fno)
#     mf.set_name("まばたき")
#     mf.ratio = max(0, min(1, min_blink))
#     motion.regist_mf(mf, mf.name, mf.fno)

#     # 両目の平均とする
#     mean_eye_euler = (right_eye_euler + left_eye_euler) / 2
#     eye_bf = motion.calc_bf("両目", fno)
#     eye_bf.rotation = MQuaternion.fromEulerAngles(mean_eye_euler.x(), mean_eye_euler.y(), 0)
#     motion.regist_bf(eye_bf, "両目", fno)
    

# def calc_left_eye(fno: int, motion: VmdMotion, frame_joints: dict):
#     left_eye1 = get_vec2(frame_joints["faces"], "left_eye1")
#     left_eye2 = get_vec2(frame_joints["faces"], "left_eye2")
#     left_eye3 = get_vec2(frame_joints["faces"], "left_eye3")
#     left_eye4 = get_vec2(frame_joints["faces"], "left_eye4")
#     left_eye5 = get_vec2(frame_joints["faces"], "left_eye5")
#     left_eye6 = get_vec2(frame_joints["faces"], "left_eye6")

#     if "eyes" in frame_joints:
#         left_pupil = MVector2D(frame_joints["eyes"]["left"]["x"], frame_joints["eyes"]["left"]["y"])
#     else:
#         left_pupil = MVector2D()

#     # 左目のEAR(eyes aspect ratio)
#     left_blink, left_smile, left_eye_euler = get_blink_ratio(fno, left_eye1, left_eye2, left_eye3, left_eye4, left_eye5, left_eye6, left_pupil)

#     mf = VmdMorphFrame(fno)
#     mf.set_name("ウィンク右")
#     mf.ratio = max(0, min(1, left_smile))

#     motion.regist_mf(mf, mf.name, mf.fno)

#     mf = VmdMorphFrame(fno)
#     mf.set_name("ｳｨﾝｸ２右")
#     mf.ratio = max(0, min(1, left_blink))

#     motion.regist_mf(mf, mf.name, mf.fno)

#     return left_eye_euler

# def calc_right_eye(fno: int, motion: VmdMotion, frame_joints: dict):
#     right_eye1 = get_vec2(frame_joints["faces"], "right_eye1")
#     right_eye2 = get_vec2(frame_joints["faces"], "right_eye2")
#     right_eye3 = get_vec2(frame_joints["faces"], "right_eye3")
#     right_eye4 = get_vec2(frame_joints["faces"], "right_eye4")
#     right_eye5 = get_vec2(frame_joints["faces"], "right_eye5")
#     right_eye6 = get_vec2(frame_joints["faces"], "right_eye6")

#     if "eyes" in frame_joints:
#         right_pupil = MVector2D(frame_joints["eyes"]["right"]["x"], frame_joints["eyes"]["right"]["y"])
#     else:
#         right_pupil = MVector2D()

#     # 右目のEAR(eyes aspect ratio)
#     right_blink, right_smile, right_eye_euler = get_blink_ratio(fno, right_eye1, right_eye2, right_eye3, right_eye4, right_eye5, right_eye6, right_pupil, is_right=True)

#     mf = VmdMorphFrame(fno)
#     mf.set_name("ウィンク")
#     mf.ratio = max(0, min(1, right_smile))
#     motion.regist_mf(mf, mf.name, mf.fno)

#     mf = VmdMorphFrame(fno)
#     mf.set_name("ウィンク２")
#     mf.ratio = max(0, min(1, right_blink))
#     motion.regist_mf(mf, mf.name, mf.fno)

#     return right_eye_euler

# def euclidean_distance(point1: MVector2D, point2: MVector2D):
#     return math.sqrt((point1.x() - point2.x())**2 + (point1.y() - point2.y())**2)

# def get_blink_ratio(fno: int, eye1: MVector2D, eye2: MVector2D, eye3: MVector2D, eye4: MVector2D, eye5: MVector2D, eye6: MVector2D, pupil: MVector2D, is_right=False):
#     #loading all the required points
#     corner_left  = eye4
#     corner_right = eye1
#     corner_center = (eye1 + eye4) / 2
    
#     center_top = (eye2 + eye3) / 2
#     center_bottom = (eye5 + eye6) / 2

#     #calculating distance
#     horizontal_length = euclidean_distance(corner_left, corner_right)
#     vertical_length = euclidean_distance(center_top, center_bottom)

#     ratio = horizontal_length / vertical_length
#     new_ratio = min(1, math.sin(calc_ratio(ratio, 0, 12, 0, 1)))

#     # 笑いの比率(日本人用に比率半分)
#     smile_ratio = (center_bottom.y() - corner_center.y()) / (center_bottom.y() - center_top.y())

#     if smile_ratio > 1:
#         # １より大きい場合、目頭よりも上瞼が下にあるという事なので、通常瞬きと見なす
#         return 1, 0, MVector3D()
    
#     # 目は四角の中にあるはず
#     pupil_x = 0
#     pupil_y = 0
#     if pupil != MVector2D():
#         eye_right = ((eye3 + eye5) / 2)
#         eye_left = ((eye2 + eye6) / 2)
#         pupil_horizonal_ratio = (pupil.x() - min(eye1.x(), eye4.x())) / (max(eye1.x(), eye4.x()) - min(eye1.x(), eye4.x()))
#         pupil_vertical_ratio = (pupil.y() - center_top.y()) / (center_bottom.y() - center_top.y())

#         if abs(pupil_horizonal_ratio) <= 1 and abs(pupil_vertical_ratio) <= 1:
#             # 比率を超えている場合、計測失敗してるので初期値            
#             pupil_x = calc_ratio(pupil_vertical_ratio, 0, 1, -15, 15)
#             pupil_y = calc_ratio(pupil_horizonal_ratio, 0, 1, -30, 20)

#     return new_ratio * (1 - smile_ratio), new_ratio * smile_ratio, MVector3D(pupil_x * -1, pupil_y * -1, 0)

# def calc_ratio(ratio: float, oldmin: float, oldmax: float, newmin: float, newmax: float):
#     # https://qastack.jp/programming/929103/convert-a-number-range-to-another-range-maintaining-ratio
#     # NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
#     return (((ratio - oldmin) * (newmax - newmin)) / (oldmax - oldmin)) + newmin

# def get_vec2(joints: dict, jname: str):
#     if jname in MORPH_CONNECTIONS:
#         joint = joints[MORPH_CONNECTIONS[jname]]
#         return MVector2D(joint["x"], joint["y"])
    
#     return MVector2D()


# MORPH_CONNECTIONS = {
#     # Left Eye brow
#     'left_eye_brow1': "23",
#     'left_eye_brow2': "24",
#     'left_eye_brow3': "25",
#     'left_eye_brow4': "26",
#     'left_eye_brow5': "27",

#     # Right Eye brow
#     'right_eye_brow1': "22",
#     'right_eye_brow2': "21",
#     'right_eye_brow3': "20",
#     'right_eye_brow4': "19",
#     'right_eye_brow5': "18",

#     # Left Eye
#     'left_eye1': "46",
#     'left_eye2': "45",
#     'left_eye3': "44",
#     'left_eye4': "43",
#     'left_eye5': "48",
#     'left_eye6': "47",

#     # Right Eye
#     'right_eye1': "40",
#     'right_eye2': "39",
#     'right_eye3': "38",
#     'right_eye4': "37",
#     'right_eye5': "42",
#     'right_eye6': "41",

#     # Nose Vertical
#     'nose1': "28",
#     'nose2': "29",
#     'nose3': "30",
#     'nose4': "31",

#     # Nose Horizontal
#     'left_nose_1': "36",
#     'left_nose_2': "35",
#     'nose_middle': "34",
#     'right_nose_1': "33",
#     'right_nose_2': "32",

#     # Mouth
#     'left_mouth_1': "55",
#     'left_mouth_2': "54",
#     'left_mouth_3': "53",
#     'mouth_top': "52",
#     'right_mouth_3': "51",
#     'right_mouth_2': "50",
#     'right_mouth_1': "49",
#     'right_mouth_5': "60",
#     'right_mouth_4': "59",
#     'mouth_bottom': "58",
#     'left_mouth_4': "57",
#     'left_mouth_5': "56",

#     # Lips
#     'left_lip_1': "65",
#     'left_lip_2': "64",
#     'lip_top': "63",
#     'right_lip_2': "62",
#     'right_lip_1': "61",
#     'right_lip_3': "68",
#     'lip_bottom': "67",
#     'left_lip_3': "66",
# }

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


PMX_CONNECTIONS = {
    "pelvis": {"mmd": "下半身", "parent": "グルーブ", "tail": "pelvis2", "display": "体幹", "axis": None},
    "pelvis2": {"mmd": "下半身先", "parent": "pelvis", "tail": "", "display": "体幹", "axis": None},
    "left_hip": {"mmd": "左足", "parent": "pelvis", "tail": "left_knee", "display": "左足", "axis": MVector3D(1, 0, 0)},
    "right_hip": {"mmd": "右足", "parent": "pelvis", "tail": "right_knee", "display": "右足", "axis": MVector3D(-1, 0, 0)},
    "left_knee": {"mmd": "左ひざ", "parent": "left_hip", "tail": "left_ankle", "display": "左足", "axis": MVector3D(1, 0, 0)},
    "right_knee": {"mmd": "右ひざ", "parent": "right_hip", "tail": "right_ankle", "display": "右足", "axis": MVector3D(-1, 0, 0)},
    "left_ankle": {"mmd": "左足首", "parent": "left_knee", "tail": "left_foot", "display": "左足", "axis": MVector3D(1, 0, 0)},
    "right_ankle": {"mmd": "右足首", "parent": "right_knee", "tail": "right_foot", "display": "右足", "axis": MVector3D(-1, 0, 0)},
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
    "left_wrist_tail": {"mmd": "左手首先", "parent": "left_wrist", "tail": "", "display": "左手", "axis": MVector3D(1, 0, 0)},
    "right_wrist_tail": {"mmd": "右手首先", "parent": "right_wrist", "tail": "", "display": "右手", "axis": MVector3D(-1, 0, 0)},
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

    "left_index1": {"mmd": "左人指１", "parent": "left_wrist", "tail": "left_index2", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_index2": {"mmd": "左人指２", "parent": "left_index1", "tail": "left_index3", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_index3": {"mmd": "左人指３", "parent": "left_index2", "tail": "left_index", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_middle1": {"mmd": "左中指１", "parent": "left_wrist", "tail": "left_middle2", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_middle2": {"mmd": "左中指２", "parent": "left_middle1", "tail": "left_middle3", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_middle3": {"mmd": "左中指３", "parent": "left_middle2", "tail": "left_middle", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_pinky1": {"mmd": "左小指１", "parent": "left_wrist", "tail": "left_pinky2", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_pinky2": {"mmd": "左小指２", "parent": "left_pinky1", "tail": "left_pinky3", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_pinky3": {"mmd": "左小指３", "parent": "left_pinky2", "tail": "left_pinky", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_ring1": {"mmd": "左薬指１", "parent": "left_wrist", "tail": "left_ring2", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_ring2": {"mmd": "左薬指２", "parent": "left_ring1", "tail": "left_ring3", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_ring3": {"mmd": "左薬指３", "parent": "left_ring2", "tail": "left_ring", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_thumb1": {"mmd": "左親指０", "parent": "left_wrist", "tail": "left_thumb2", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_thumb2": {"mmd": "左親指１", "parent": "left_thumb1", "tail": "left_thumb3", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_thumb3": {"mmd": "左親指２", "parent": "left_thumb2", "tail": "left_thumb", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "right_index1": {"mmd": "右人指１", "parent": "right_wrist", "tail": "right_index2", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_index2": {"mmd": "右人指２", "parent": "right_index1", "tail": "right_index3", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_index3": {"mmd": "右人指３", "parent": "right_index2", "tail": "right_index", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_middle1": {"mmd": "右中指１", "parent": "right_wrist", "tail": "right_middle2", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_middle2": {"mmd": "右中指２", "parent": "right_middle1", "tail": "right_middle3", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_middle3": {"mmd": "右中指３", "parent": "right_middle2", "tail": "right_middle", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_pinky1": {"mmd": "右小指１", "parent": "right_wrist", "tail": "right_pinky2", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_pinky2": {"mmd": "右小指２", "parent": "right_pinky1", "tail": "right_pinky3", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_pinky3": {"mmd": "右小指３", "parent": "right_pinky2", "tail": "right_pinky", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_ring1": {"mmd": "右薬指１", "parent": "right_wrist", "tail": "right_ring2", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_ring2": {"mmd": "右薬指２", "parent": "right_ring1", "tail": "right_ring3", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_ring3": {"mmd": "右薬指３", "parent": "right_ring2", "tail": "right_ring", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_thumb1": {"mmd": "右親指０", "parent": "right_wrist", "tail": "right_thumb2", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_thumb2": {"mmd": "右親指１", "parent": "right_thumb1", "tail": "right_thumb3", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_thumb3": {"mmd": "右親指２", "parent": "right_thumb2", "tail": "right_thumb", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "left_thumb": {"mmd": "左親指先", "parent": "left_thumb3", "tail": "", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_index": {"mmd": "左人差指先", "parent": "left_index3", "tail": "", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_middle": {"mmd": "左中指先", "parent": "left_middle3", "tail": "", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_ring": {"mmd": "左薬指先", "parent": "left_ring3", "tail": "", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_pinky": {"mmd": "左小指先", "parent": "left_pinky3", "tail": "", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "right_thumb": {"mmd": "右親指先", "parent": "right_thumb3", "tail": "", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_index": {"mmd": "右人差指先", "parent": "right_index3", "tail": "", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_middle": {"mmd": "右中指先", "parent": "right_middle3", "tail": "", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_ring": {"mmd": "右薬指先", "parent": "right_ring3", "tail": "", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_pinky": {"mmd": "右小指先", "parent": "right_pinky3", "tail": "", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    
    "right_eye_brow1": {"mmd": "right_eye_brow1", "parent": "head", "tail": "", "display": "眉", "axis": None},
    "right_eye_brow2": {"mmd": "right_eye_brow2", "parent": "right_eye_brow1", "tail": "", "display": "眉", "axis": None},
    "right_eye_brow3": {"mmd": "right_eye_brow3", "parent": "right_eye_brow2", "tail": "", "display": "眉", "axis": None},
    "right_eye_brow4": {"mmd": "right_eye_brow4", "parent": "right_eye_brow3", "tail": "", "display": "眉", "axis": None},
    "right_eye_brow5": {"mmd": "right_eye_brow5", "parent": "right_eye_brow4", "tail": "", "display": "眉", "axis": None},
    "left_eye_brow1": {"mmd": "left_eye_brow1", "parent": "head", "tail": "", "display": "眉", "axis": None},
    "left_eye_brow2": {"mmd": "left_eye_brow2", "parent": "left_eye_brow1", "tail": "", "display": "眉", "axis": None},
    "left_eye_brow3": {"mmd": "left_eye_brow3", "parent": "left_eye_brow2", "tail": "", "display": "眉", "axis": None},
    "left_eye_brow4": {"mmd": "left_eye_brow4", "parent": "left_eye_brow3", "tail": "", "display": "眉", "axis": None},
    "left_eye_brow5": {"mmd": "left_eye_brow5", "parent": "left_eye_brow4", "tail": "", "display": "眉", "axis": None},
    "nose1": {"mmd": "nose1", "parent": "nose", "tail": "", "display": "鼻", "axis": None},
    "nose2": {"mmd": "nose2", "parent": "nose1", "tail": "", "display": "鼻", "axis": None},
    "nose3": {"mmd": "nose3", "parent": "nose2", "tail": "", "display": "鼻", "axis": None},
    "nose4": {"mmd": "nose4", "parent": "nose3", "tail": "", "display": "鼻", "axis": None},
    "right_nose_2": {"mmd": "right_nose_2", "parent": "nose", "tail": "", "display": "鼻", "axis": None},
    "right_nose_1": {"mmd": "right_nose_1", "parent": "nose", "tail": "", "display": "鼻", "axis": None},
    "nose_middle": {"mmd": "nose_middle", "parent": "nose", "tail": "", "display": "鼻", "axis": None},
    "left_nose_1": {"mmd": "left_nose_1", "parent": "nose", "tail": "", "display": "鼻", "axis": None},
    "left_nose_2": {"mmd": "left_nose_2", "parent": "nose", "tail": "", "display": "鼻", "axis": None},
    "right_eye1": {"mmd": "right_eye1", "parent": "right_eye", "tail": "", "display": "目", "axis": None},
    "right_eye2": {"mmd": "right_eye2", "parent": "right_eye1", "tail": "", "display": "目", "axis": None},
    "right_eye3": {"mmd": "right_eye3", "parent": "right_eye2", "tail": "", "display": "目", "axis": None},
    "right_eye4": {"mmd": "right_eye4", "parent": "right_eye3", "tail": "", "display": "目", "axis": None},
    "right_eye5": {"mmd": "right_eye5", "parent": "right_eye4", "tail": "", "display": "目", "axis": None},
    "right_eye6": {"mmd": "right_eye6", "parent": "right_eye5", "tail": "", "display": "目", "axis": None},
    "left_eye1": {"mmd": "left_eye1", "parent": "left_eye", "tail": "", "display": "目", "axis": None},
    "left_eye2": {"mmd": "left_eye2", "parent": "left_eye1", "tail": "", "display": "目", "axis": None},
    "left_eye3": {"mmd": "left_eye3", "parent": "left_eye2", "tail": "", "display": "目", "axis": None},
    "left_eye4": {"mmd": "left_eye4", "parent": "left_eye3", "tail": "", "display": "目", "axis": None},
    "left_eye5": {"mmd": "left_eye5", "parent": "left_eye4", "tail": "", "display": "目", "axis": None},
    "left_eye6": {"mmd": "left_eye6", "parent": "left_eye5", "tail": "", "display": "目", "axis": None},
    "right_mouth_1": {"mmd": "right_mouth_1", "parent": "head", "tail": "", "display": "口", "axis": None},
    "right_mouth_2": {"mmd": "right_mouth_2", "parent": "right_mouth_1", "tail": "", "display": "口", "axis": None},
    "right_mouth_3": {"mmd": "right_mouth_3", "parent": "right_mouth_2", "tail": "", "display": "口", "axis": None},
    "right_mouth_4": {"mmd": "right_mouth_4", "parent": "right_mouth_3", "tail": "", "display": "口", "axis": None},
    "right_mouth_5": {"mmd": "right_mouth_5", "parent": "right_mouth_4", "tail": "", "display": "口", "axis": None},
    "mouth_top": {"mmd": "mouth_top", "parent": "head", "tail": "", "display": "口", "axis": None},
    "left_mouth_1": {"mmd": "left_mouth_1", "parent": "head", "tail": "", "display": "口", "axis": None},
    "left_mouth_2": {"mmd": "left_mouth_2", "parent": "left_mouth_1", "tail": "", "display": "口", "axis": None},
    "left_mouth_3": {"mmd": "left_mouth_3", "parent": "left_mouth_2", "tail": "", "display": "口", "axis": None},
    "left_mouth_4": {"mmd": "left_mouth_4", "parent": "left_mouth_3", "tail": "", "display": "口", "axis": None},
    "left_mouth_5": {"mmd": "left_mouth_5", "parent": "left_mouth_4", "tail": "", "display": "口", "axis": None},
    "mouth_bottom": {"mmd": "mouth_bottom", "parent": "head", "tail": "", "display": "口", "axis": None},
    "right_lip_1": {"mmd": "right_lip_1", "parent": "head", "tail": "", "display": "口", "axis": None},
    "right_lip_2": {"mmd": "right_lip_2", "parent": "right_lip_1", "tail": "", "display": "口", "axis": None},
    "right_lip_3": {"mmd": "right_lip_3", "parent": "right_lip_2", "tail": "", "display": "口", "axis": None},
    "lip_top": {"mmd": "lip_top", "parent": "head", "tail": "", "display": "口", "axis": None},
    "left_lip_1": {"mmd": "left_lip_1", "parent": "head", "tail": "", "display": "口", "axis": None},
    "left_lip_2": {"mmd": "left_lip_2", "parent": "left_lip_1", "tail": "", "display": "口", "axis": None},
    "left_lip_3": {"mmd": "left_lip_3", "parent": "left_lip_2", "tail": "", "display": "口", "axis": None},
    "lip_bottom": {"mmd": "lip_bottom", "parent": "head", "tail": "", "display": "口", "axis": None},
    "right_contour_1": {"mmd": "right_contour_1", "parent": "head", "tail": "", "display": "輪郭", "axis": None},
    "right_contour_2": {"mmd": "right_contour_2", "parent": "right_contour_1", "tail": "", "display": "輪郭", "axis": None},
    "right_contour_3": {"mmd": "right_contour_3", "parent": "right_contour_2", "tail": "", "display": "輪郭", "axis": None},
    "right_contour_4": {"mmd": "right_contour_4", "parent": "right_contour_3", "tail": "", "display": "輪郭", "axis": None},
    "right_contour_5": {"mmd": "right_contour_5", "parent": "right_contour_4", "tail": "", "display": "輪郭", "axis": None},
    "right_contour_6": {"mmd": "right_contour_6", "parent": "right_contour_5", "tail": "", "display": "輪郭", "axis": None},
    "right_contour_7": {"mmd": "right_contour_7", "parent": "right_contour_6", "tail": "", "display": "輪郭", "axis": None},
    "right_contour_8": {"mmd": "right_contour_8", "parent": "right_contour_7", "tail": "", "display": "輪郭", "axis": None},
    "contour_middle": {"mmd": "contour_middle", "parent": "head", "tail": "", "display": "輪郭", "axis": None},
    "left_contour_1": {"mmd": "left_contour_1", "parent": "head", "tail": "", "display": "輪郭", "axis": None},
    "left_contour_2": {"mmd": "left_contour_2", "parent": "left_contour_1", "tail": "", "display": "輪郭", "axis": None},
    "left_contour_3": {"mmd": "left_contour_3", "parent": "left_contour_2", "tail": "", "display": "輪郭", "axis": None},
    "left_contour_4": {"mmd": "left_contour_4", "parent": "left_contour_3", "tail": "", "display": "輪郭", "axis": None},
    "left_contour_5": {"mmd": "left_contour_5", "parent": "left_contour_4", "tail": "", "display": "輪郭", "axis": None},
    "left_contour_6": {"mmd": "left_contour_6", "parent": "left_contour_5", "tail": "", "display": "輪郭", "axis": None},
    "left_contour_7": {"mmd": "left_contour_7", "parent": "left_contour_6", "tail": "", "display": "輪郭", "axis": None},
    "left_contour_8": {"mmd": "left_contour_8", "parent": "left_contour_7", "tail": "", "display": "輪郭", "axis": None},
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
    # 'left_wrist': {'direction': ('left_wrist', 'left_wrist_tail'), 'up': ('left_elbow', 'left_wrist'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow']},
    # 'left_thumb1': {'direction': ('left_thumb1', 'left_thumb2'), 'up': ('left_index1', 'left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist']},
    # 'left_thumb2': {'direction': ('left_thumb2', 'left_thumb3'), 'up': ('left_index1', 'left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist', 'left_thumb1']},
    # 'left_thumb3': {'direction': ('left_thumb3', 'left_thumb'), 'up': ('left_index1', 'left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist', 'left_thumb1', 'left_thumb2']},
    # 'left_index1': {'direction': ('left_index1', 'left_index2'), 'up': ('left_index1', 'left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist']},
    # 'left_index2': {'direction': ('left_index2', 'left_index3'), 'up': ('left_index1', 'left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist', 'left_index1']},
    # 'left_index3': {'direction': ('left_index3', 'left_index'), 'up': ('left_index1', 'left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist', 'left_index1', 'left_index2']},
    # 'left_middle1': {'direction': ('left_middle1', 'left_middle2'), 'up': ('left_index1', 'left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist']},
    # 'left_middle2': {'direction': ('left_middle2', 'left_middle3'), 'up': ('left_index1', 'left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist', 'left_middle1']},
    # 'left_middle3': {'direction': ('left_middle3', 'left_middle'), 'up': ('left_index1', 'left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist', 'left_middle1', 'left_middle2']},
    # 'left_ring1': {'direction': ('left_ring1', 'left_ring2'), 'up': ('left_index1', 'left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist']},
    # 'left_ring2': {'direction': ('left_ring2', 'left_ring3'), 'up': ('left_index1', 'left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist', 'left_ring1']},
    # 'left_ring3': {'direction': ('left_ring3', 'left_ring'), 'up': ('left_index1', 'left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist', 'left_ring1', 'left_ring2']},
    # 'left_pinky1': {'direction': ('left_pinky1', 'left_pinky2'), 'up': ('left_index1', 'left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist']},
    # 'left_pinky2': {'direction': ('left_pinky2', 'left_pinky3'), 'up': ('left_index1', 'left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist', 'left_pinky1']},
    # 'left_pinky3': {'direction': ('left_pinky3', 'left_pinky'), 'up': ('left_index1', 'left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist', 'left_pinky1', 'left_pinky2']},
    'left_hip': {'direction': ('left_hip', 'left_knee'), 'up': ('left_hip', 'right_hip'), 'cancel': ['pelvis']},
    'left_knee': {'direction': ('left_knee', 'left_ankle'), 'up': ('left_hip', 'left_knee'), 'cancel': ['pelvis', 'left_hip']},
    'left_ankle': {'direction': ('left_ankle', 'left_foot'), 'up': ('left_knee', 'left_ankle'), 'cancel': ['pelvis', 'left_hip', 'left_knee']},

    'right_collar': {'direction': ('right_collar', 'right_shoulder'), 'up': ('spine2', 'neck'), 'cross': ('right_shoulder', 'left_shoulder'), 'cancel': ['spine1', 'spine2']},
    'right_shoulder': {'direction': ('right_shoulder', 'right_elbow'), 'up': ('right_collar', 'right_shoulder'), 'cancel': ['spine1', 'spine2', 'right_collar']},
    'right_elbow': {'direction': ('right_elbow', 'right_wrist'), 'up': ('right_shoulder', 'right_elbow'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder']},
    # 'right_wrist': {'direction': ('right_wrist', 'right_wrist_tail'), 'up': ('right_elbow', 'right_wrist'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow']},
    # 'right_thumb1': {'direction': ('right_thumb1', 'right_thumb2'), 'up': ('right_index1', 'right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist']},
    # 'right_thumb2': {'direction': ('right_thumb2', 'right_thumb3'), 'up': ('right_index1', 'right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist', 'right_thumb1']},
    # 'right_thumb3': {'direction': ('right_thumb3', 'right_thumb'), 'up': ('right_index1', 'right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist', 'right_thumb1', 'right_thumb2']},
    # 'right_index1': {'direction': ('right_index1', 'right_index2'), 'up': ('right_index1', 'right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist']},
    # 'right_index2': {'direction': ('right_index2', 'right_index3'), 'up': ('right_index1', 'right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist', 'right_index1']},
    # 'right_index3': {'direction': ('right_index3', 'right_index'), 'up': ('right_index1', 'right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist', 'right_index1', 'right_index2']},
    # 'right_middle1': {'direction': ('right_middle1', 'right_middle2'), 'up': ('right_index1', 'right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist']},
    # 'right_middle2': {'direction': ('right_middle2', 'right_middle3'), 'up': ('right_index1', 'right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist', 'right_middle1']},
    # 'right_middle3': {'direction': ('right_middle3', 'right_middle'), 'up': ('right_index1', 'right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist', 'right_middle1', 'right_middle2']},
    # 'right_ring1': {'direction': ('right_ring1', 'right_ring2'), 'up': ('right_index1', 'right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist']},
    # 'right_ring2': {'direction': ('right_ring2', 'right_ring3'), 'up': ('right_index1', 'right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist', 'right_ring1']},
    # 'right_ring3': {'direction': ('right_ring3', 'right_ring'), 'up': ('right_index1', 'right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist', 'right_ring1', 'right_ring2']},
    # 'right_pinky1': {'direction': ('right_pinky1', 'right_pinky2'), 'up': ('right_index1', 'right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist']},
    # 'right_pinky2': {'direction': ('right_pinky2', 'right_pinky3'), 'up': ('right_index1', 'right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist', 'right_pinky1']},
    # 'right_pinky3': {'direction': ('right_pinky3', 'right_pinky'), 'up': ('right_index1', 'right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist', 'right_pinky1', 'right_pinky2']},
    'right_hip': {'direction': ('right_hip', 'right_knee'), 'up': ('right_hip', 'left_hip'), 'cancel': ['pelvis']},
    'right_knee': {'direction': ('right_knee', 'right_ankle'), 'up': ('right_hip', 'right_knee'), 'cancel': ['pelvis', 'right_hip']},
    'right_ankle': {'direction': ('right_ankle', 'right_foot'), 'up': ('right_knee', 'right_ankle'), 'cancel': ['pelvis', 'right_hip', 'right_knee']},
}
