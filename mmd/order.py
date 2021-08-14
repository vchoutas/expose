# -*- coding: utf-8 -*-
import os
import argparse
import glob
import re
import json
import csv
import cv2
import shutil
import sys
import pathlib

# import vision essentials
import numpy as np
from tqdm import tqdm

from mmd.utils.MLogger import MLogger
from mmd.utils.MServiceUtils import sort_by_numeric
from mmd.mmd.VmdData import OneEuroFilter
from lighttrack.visualizer.detection_visualizer import draw_bbox


logger = MLogger(__name__, level=MLogger.DEBUG)

def execute(args):
    try:
        logger.info('人物再追跡処理開始: {0}', args.img_dir, decoration=MLogger.DECORATION_BOX)

        if not os.path.exists(args.img_dir):
            logger.error("指定された処理用ディレクトリが存在しません。: {0}", args.img_dir, decoration=MLogger.DECORATION_BOX)
            return False

        if not os.path.exists(args.order_file):
            logger.error("指定された順番指定用ファイルが存在しません。: {0}", args.order_file, decoration=MLogger.DECORATION_BOX)
            return False
        
        # 行：人物、列：INDEX指定
        order_list = []
        try:
            with open(args.order_file, "r", encoding='utf-8') as of:
                reader = csv.reader(of)
                order_list = [row for row in reader if len(row) > 0]
        except Exception as e:
            logger.error("指定された順番指定用ファイルのCSV読み取り処理に失敗しました", e, decoration=MLogger.DECORATION_BOX)
            return False

        logger.info("人物追跡指定開始", decoration=MLogger.DECORATION_LINE)

        # 全人物分の順番別フォルダ(順番INDEX分生成しておく)

        # 既存は削除
        if os.path.exists(os.path.join(args.img_dir, "ordered")):
            shutil.rmtree(os.path.join(args.img_dir, "ordered"))

        ordered_dir_pathes = []
        for oidx, _ in enumerate(order_list):
            ordered_dir_path = os.path.join(args.img_dir, "ordered", f"{oidx:03}")
            os.makedirs(ordered_dir_path, exist_ok=True)
            ordered_dir_pathes.append(ordered_dir_path)

        process_img_pathes = sorted(glob.glob(os.path.join(args.img_dir, "frames", "**", "frame_*.png")), key=sort_by_numeric)

        # 順番指定後はDLを早くするため、mp4のままとする
        ordered_bbox_path = os.path.join(args.img_dir, "ordered_bbox.mp4")
        # fourcc_name = "IYUV" if os.name == "nt" else "I420"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        avi_out = None
        avi_width = 0
        avi_height = 0
        if len(process_img_pathes) > 0:
            process_img_path = process_img_pathes[0]
            img = cv2.imread(process_img_path)
            # scale = min(1, 2000 / len(process_img_pathes))
            scale = 1
            avi_width = int(img.shape[1] * scale)
            avi_height = int(img.shape[0] * scale)
            avi_out = cv2.VideoWriter(ordered_bbox_path, fourcc, 30.0, (avi_width, avi_height))
    
        # 順番指定正規表現
        order_pattern = re.compile(r'(\d+)\:(\d*)\-(\d*)')

        for iidx, process_img_path in enumerate(tqdm(process_img_pathes)):
            # 元のbboxは削除
            bbox_path = os.path.join(str(pathlib.Path(process_img_path).parent), os.path.basename(process_img_path).replace("frame", "bbox"))
            if os.path.exists(bbox_path):
                os.remove(bbox_path)

            # 入力画像パス
            out_frame = cv2.imread(process_img_path)
            # 人数分読み込む
            joint_json_pathes = sorted(glob.glob(os.path.join(args.img_dir, "frames", f"{iidx:012}", "frame_*.json")), key=sort_by_numeric)

            for joint_json_path in joint_json_pathes:
                with open(joint_json_path, 'r') as f:
                    bbox_frame = json.load(f)
                    if 'track_id' not in bbox_frame:
                        continue
                    
                    track_id = bbox_frame['track_id']
                    for oidx, order_idxs in enumerate(order_list):
                        if str(track_id) in order_idxs:
                            # 追跡IDXが順番指定ファイルにある場合、採用してファイルコピー
                            ordered_dir_path = os.path.join(args.img_dir, "ordered", f"{oidx:03}")
                            shutil.copy(joint_json_path, ordered_dir_path)

                            bbox = [bbox_frame['bbox']['x'], bbox_frame['bbox']['y'], bbox_frame['bbox']['width'], bbox_frame['bbox']['height']]

                            # bbox描画
                            out_frame = draw_bbox(out_frame, bbox, 1, None, track_id=oidx)
                        else:
                            # 追跡IDXそのものがない場合、フレーム途中までの指定がないかチェック
                            for order_idx_str in order_idxs:
                                m = order_pattern.match(order_idx_str)
                                if m:
                                    # 正規表現グループを分解
                                    order_idx, order_startf_str, order_endf_str = m.groups()
                                    if str(track_id) == order_idx:
                                        # INDEXが見つかった場合、フレームの開始と終了を確認する
                                        order_startf = 0 if not order_startf_str else int(order_startf_str)
                                        order_endf = sys.maxsize if not order_endf_str else int(order_endf_str)
                                        if order_startf <= iidx <= order_endf:
                                            # フレーム範囲内である場合、採用
                                            ordered_dir_path = os.path.join(args.img_dir, "ordered", f"{oidx:03}")
                                            shutil.copy(joint_json_path, ordered_dir_path)

                                            bbox = [bbox_frame['bbox']['x'], bbox_frame['bbox']['y'], bbox_frame['bbox']['width'], bbox_frame['bbox']['height']]

                                            # bbox描画
                                            out_frame = draw_bbox(out_frame, bbox, 1, None, track_id=oidx)

            # フレーム番号追記
            cv2.putText(out_frame, f'{iidx:05}F', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(182, 0, 182), thickness = 2, lineType = cv2.LINE_AA)
            
            # 画像出力
            cv2.imwrite(os.path.join(args.img_dir, "frames", f"{iidx:012}", f"order_{iidx:012}.png"), out_frame)

            # 縮小
            avi_frame = cv2.resize(out_frame, (avi_width, avi_height))

            # トラッキングavi合成
            avi_out.write(avi_frame)

        avi_out.release()
        cv2.destroyAllWindows()

        logger.info('人物再追跡処理終了: {0}', args.img_dir, decoration=MLogger.DECORATION_BOX)

        return True
    except Exception as e:
        logger.critical("人物再追跡で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        return False
