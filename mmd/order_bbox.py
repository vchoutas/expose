# -*- coding: utf-8 -*-
import os
import argparse
import glob
import sys
import json
import csv
import cv2
import pathlib

# import vision essentials
import numpy as np
from tqdm import tqdm

from mmd.utils.MLogger import MLogger
from mmd.utils.MServiceUtils import sort_by_numeric
from lighttrack.visualizer.detection_visualizer import draw_bbox
import mmd.config as mconfig


logger = MLogger(__name__, level=MLogger.DEBUG)

def execute(args):
    try:
        logger.info('人物再追跡処理開始: %s', args.img_dir, decoration=MLogger.DECORATION_BOX)

        if not os.path.exists(args.img_dir):
            logger.error("指定された処理用ディレクトリが存在しません。: %s", args.img_dir, decoration=MLogger.DECORATION_BOX)
            return False

        if not os.path.exists(args.order_file):
            logger.error("指定された順番指定用ファイルが存在しません。: %s", args.order_file, decoration=MLogger.DECORATION_BOX)
            return False
        
        # 行：人物、列：INDEX指定
        order_list = []
        with open(args.order_file, "r") as of:
            reader = csv.reader(of)
            order_list = [row for row in reader]

        bbox_json_pathes = sorted(glob.glob(os.path.join(args.img_dir, "**", "bbox_*.json")), key=sort_by_numeric)

        for bidx, bbox_json_path in enumerate(tqdm(bbox_json_pathes)):
            # bboxをフレームごとに読み込む

            p_file = pathlib.Path(bbox_json_path)
            bbox_img_path = str(p_file.with_name(f"frame_{bidx:012}.png"))
            out_frame = cv2.imread(bbox_img_path)

            bbox_dict = {}
            with open(bbox_json_path, 'r') as f:
                bbox_dict = json.load(f)
                for oidx, ol in enumerate(order_list):
                    candidate = bbox_dict['candidates'][oidx]
                    if candidate['track_id'] in ol:
                        # track id が、人物単位のリストに存在している場合、その人物INDEXを採用する
                        candidate['person_id'] = oidx

                        # bboxを再出力
                        bbox = np.array(candidate["det_bbox"]).astype(int)
                        score = candidate["det_score"]
                        out_frame = draw_bbox(out_frame, bbox, score, None, track_id = oidx)

            # 画像出力
            cv2.imwrite(bbox_img_path, out_frame)

            # json再保存
            with open(bbox_json_path, 'w') as f:
                json.dump(bbox_dict, f, indent=4)

        logger.info('人物再追跡処理終了: %s', args.img_dir, decoration=MLogger.DECORATION_BOX)

        return True

        #     # フレーム
        #     frame = cv2.imread(file_path)
        #     out.write(frame)

        # order_dict = {}
        # with open(args.order_file, "r") as f:
        #     reader = csv.reader(f)
        #     for ridx, rows in enumerate(reader):
        #         if len(rows) <= 0:
        #             continue

        #         now_track_id = rows[0]
        #         for fidx in range(last_frames + 1):
        #             if fidx in rows:
        #                 # フレーム番号がIDにある場合、トラッキングID切り替え
        #                 now_track_id
        #             order_dict[ridx] = now_track_id
        

                
        # for json_path in tqdm(sorted(glob.glob(os.path.join(args.img_dir, "**", "*_joints.json")), key=sort_by_numeric)):
        #     bbox_dict = {}
        #     with open(json_path, 'r') as f:
        #         bbox_dict = json.load(f)
    except Exception as e:
        logger.critical("人物再追跡で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--process-dir', type=str, dest='img_dir', help='process dir path')
    parser.add_argument('--order-file', type=str, dest='order_file', help='specifying tracking')
    parser.add_argument('--verbose', type=int, dest='verbose', default=20, help='log level')

    args = parser.parse_args()

    MLogger.initialize(level=args.verbose, is_file=True)

    execute(args)

