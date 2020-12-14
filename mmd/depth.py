# -*- coding: utf-8 -*-
import os
import glob
import json
import csv
import argparse

from PIL import Image

# import vision essentials
import numpy as np
from tqdm import tqdm

from mmd.utils.MLogger import MLogger
from mmd.utils.MServiceUtils import sort_by_numeric

from mmd.tracking import xywh_to_x1y1x2y2_from_dict, enlarge_bbox, x1y1x2y2_to_xywh
from monoloco.monoloco.network.process import factory_for_gt
from monoloco.monoloco.network import MonoLoco

logger = MLogger(__name__, level=MLogger.DEBUG)

def execute(args):
    try:
        logger.info('人物深度処理開始: {0}', args.img_dir, decoration=MLogger.DECORATION_BOX)

        if not os.path.exists(args.img_dir):
            logger.error("指定された処理用ディレクトリが存在しません。: {0}", args.img_dir, decoration=MLogger.DECORATION_BOX)
            return False

        parser = get_parser()
        argv = parser.parse_args(args=[])

        monoloco = MonoLoco(model=argv.model, device=argv.device, n_dropout=argv.n_dropout, p_dropout=argv.dropout)

        logger.info("人物深度推定開始", decoration=MLogger.DECORATION_LINE)

        process_img_pathes = sorted(glob.glob(os.path.join(args.img_dir, "frames", "**", "frame_*.png")), key=sort_by_numeric)

        os.makedirs(os.path.join(args.img_dir, "depths"), exist_ok=True)

        for iidx, process_img_path in enumerate(tqdm(process_img_pathes)):
            # 人数分読み込む
            bbox_frames = {}
            joint_json_pathes = sorted(glob.glob(os.path.join(args.img_dir, "frames", f"{iidx:012}", "frame_*.json")), key=sort_by_numeric)

            if len(joint_json_pathes) == 0:
                # 人物が一件も見つからなかった場合
                continue
            
            width = 0
            height = 0
            boxes = []
            keypoints = []
            # 一人以上人物が見つかった場合
            for joint_json_path in joint_json_pathes:
                with open(joint_json_path, 'r') as f:
                    bbox_frames[joint_json_path] = json.load(f)
                    width = bbox_frames[joint_json_path]['image']['width']
                    height = bbox_frames[joint_json_path]['image']['height']

                # enlarge bbox by 20% with same center position
                bbox_x1y1x2y2 = xywh_to_x1y1x2y2_from_dict(bbox_frames[joint_json_path]['bbox'])
                bbox_in_xywh = enlarge_bbox(bbox_x1y1x2y2, argv.enlarge_scale, width, height)
                bbox_det = x1y1x2y2_to_xywh(bbox_in_xywh)
                boxes.append(bbox_det)

                #bbox
                bbox_pos = np.array([bbox_frames[joint_json_path]["bbox"]["x"], bbox_frames[joint_json_path]["bbox"]["y"], bbox_frames[joint_json_path]["bbox"]["x"]])
                bbox_size = np.array([bbox_frames[joint_json_path]["bbox"]["width"], bbox_frames[joint_json_path]["bbox"]["height"], bbox_frames[joint_json_path]["bbox"]["width"]])

                # 関節は使えるのだけピックアップ
                xs = []
                ys = []
                zs = []
                for joint_name in ["head", "left_eye", "right_ear", "left_ear", "right_ear", "left_shoulder", "right_shoulder", \
                                   "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", \
                                   "left_knee", "right_knee", "left_ankle", "right_ankle"]:

                    # カメラの中心からの相対位置(Yはセンターからみて上が＋、下が－なので、反転させておく)
                    relative_pos = np.array([bbox_frames[joint_json_path]["joints"][joint_name]["x"] * -1 + 0.5, \
                                             bbox_frames[joint_json_path]["joints"][joint_name]["y"] * -1 + 1, \
                                             bbox_frames[joint_json_path]["joints"][joint_name]["z"]])

                    # グローバル位置
                    global_pos = (bbox_size * relative_pos) + bbox_pos

                    xs.append(float(global_pos[0]))
                    ys.append(float(global_pos[1]))
                    zs.append(float(global_pos[2]))

                keypoints.append([xs, ys, zs])

            im_size = (width, height)  # Width, Height (original)
            im_name = os.path.basename(process_img_path)

            kk, dic_gt = factory_for_gt(im_size, name=im_name, path_gt=argv.path_gt)

            outputs, varss = monoloco.forward(keypoints, kk)
            dic_out = monoloco.post_process(outputs, varss, boxes, keypoints, kk, dic_gt)

            # 深度のみのJSON出力
            depth_json_path = os.path.join(args.img_dir, "depths", im_name.replace("png", "json"))
            with open(depth_json_path, "w") as f:
                json.dump(dic_out, f, indent=4)

            # フレーム別情報に深度追加
            for oidx, out_bbox in enumerate(dic_out["boxes"]):
                for bidx, bbox in enumerate(boxes):
                    if bbox[0] == out_bbox[0] and bbox[1] == out_bbox[1] and bbox[2] == out_bbox[2] and bbox[3] == out_bbox[3]:
                        # bboxが合っている要素のトコに出力する
                        # depth自身は既にあるので、追記
                        joint_json_path = joint_json_pathes[bidx]
                        bbox_frames[joint_json_path]["depth"]["x"] = dic_out["xyz_pred"][oidx][0]
                        bbox_frames[joint_json_path]["depth"]["y"] = dic_out["xyz_pred"][oidx][1]
                        bbox_frames[joint_json_path]["depth"]["z"] = dic_out["xyz_pred"][oidx][2]

                        # JSON出力
                        with open(joint_json_path, "w") as f:
                            json.dump(bbox_frames[joint_json_path], f, indent=4)

        logger.info('人物深度処理終了: {0}', args.img_dir, decoration=MLogger.DECORATION_BOX)

        return True
    except Exception as e:
        logger.critical("人物深度で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        return False


def get_parser():
    parser = argparse.ArgumentParser()

    # Monoloco
    parser.add_argument('--model', help='path of MonoLoco model to load', default="monoloco/data/models/monoloco-190719-0923.pkl")
    parser.add_argument('--hidden_size', type=int, help='Number of hidden units in the model', default=512)
    parser.add_argument('--path_gt', help='path of json file with gt 3d localization', default='monoloco/data/arrays/names-kitti-190716-1618.json')
    parser.add_argument('--transform', help='transformation for the pose', default='None')
    parser.add_argument('--draw_box', help='to draw box in the images', action='store_true')
    parser.add_argument('--predict', help='whether to make prediction', action='store_true')
    parser.add_argument('--z_max', type=int, help='maximum meters distance for predictions', default=22)
    parser.add_argument('--n_dropout', type=int, help='Epistemic uncertainty evaluation', default=0)
    parser.add_argument('--device', type=int, help='device', default=0)
    parser.add_argument('--dropout', type=float, help='dropout parameter', default=0.2)
    parser.add_argument('--webcam', help='monoloco streaming', action='store_true')
    parser.add_argument('--enlarge_scale', help='monoloco streaming', default=0.15)

    return parser
