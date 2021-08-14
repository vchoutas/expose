# -*- coding: utf-8 -*-
import os
import argparse
import glob
import sys
import json
import pathlib
import _pickle as cPickle

# import vision essentials
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# import GCN utils
from lighttrack.graph.visualize_pose_matching import graph_pair_to_data, keypoints_to_graph

# 姿勢推定用
from lighttrack.network_mobile_deconv import Network
from lighttrack.HPE.config import cfg
from lighttrack.lib.tfflat.base import Tester

from lighttrack.visualizer.detection_visualizer import draw_bbox
from lighttrack.graph.visualize_pose_matching import Pose_Matcher
from mmd.utils.MLogger import MLogger
from mmd.utils.MServiceUtils import sort_by_numeric

flag_flip = True
os.environ["CUDA_VISIBLE_DEVICES"]="0"

logger = MLogger(__name__)

def execute(args):
    try:
        logger.info('人物追跡処理開始: {0}', args.img_dir, decoration=MLogger.DECORATION_BOX)

        if not os.path.exists(args.img_dir):
            logger.error("指定された処理用ディレクトリが存在しません。: {0}", args.img_dir, decoration=MLogger.DECORATION_BOX)
            return False

        if not os.path.exists(args.tracking_config):
            logger.error("指定された人物追跡設定ファイルが存在しません。: {0}", args.tracking_config, decoration=MLogger.DECORATION_BOX)
            return False

        if not os.path.exists(f"{args.tracking_model}.meta") or not os.path.exists(f"{args.tracking_model}.index") or not os.path.exists(f"{args.tracking_model}.data-00000-of-00001"):
            model_dir_path = os.path.abspath(str(pathlib.Path(args.tracking_model).parent))
            logger.error("追跡に必要な学習モデルが見つかりません。\nモデルディレクトリパス: {0}\n上記ディレクトリの中に、{1}から始まる3ファイルがある事を確認してください", 
                        model_dir_path, "snapshot_296.ckpt", decoration=MLogger.DECORATION_BOX)
            return False

        try:
            pose_matcher = Pose_Matcher(args)
        except Exception as e:
            logger.error("人物追跡設定ファイル読み込み失敗", e, decoration=MLogger.DECORATION_BOX)
            return False

        # -----------------------

        logger.info('追跡準備開始', decoration=MLogger.DECORATION_LINE)
        joint_json_pathes = sorted(glob.glob(os.path.join(args.img_dir, "frames", "**", "frame_*.json")), key=sort_by_numeric)

        # bboxのサイズの中央値を求める
        all_w = []
        all_h = []
        for iidx, joint_json_path in enumerate(tqdm(joint_json_pathes)):
            with open(joint_json_path, 'r') as f:
                json_data = json.load(f)
                width = json_data['image']['width']
                height = json_data['image']['height']

            all_w.append(json_data['bbox']['width'])
            all_h.append(json_data['bbox']['height'])
        median_w = np.median(all_w)
        median_h = np.median(all_h)

        # -----------------------

        pose_estimator = Tester(Network(), cfg)
        pose_estimator.load_weights(args.tracking_model)

        args.bbox_thresh = 0.4

        args.nms_method = 'nms'
        args.nms_thresh = 1.
        args.min_scores = 1e-10
        args.min_box_size = 0.
        args.draw_threshold = 0.2

        args.enlarge_scale = 0.2 # how much to enlarge the bbox before pose estimation

        process_bbox_path = os.path.join(args.img_dir, "bbox.mp4")
        process_img_pathes = sorted(glob.glob(os.path.join(args.img_dir, "frames", "**", "frame_*.png")), key=sort_by_numeric)

        logger.info("人物追跡開始", decoration=MLogger.DECORATION_LINE)

        # process the frames sequentially
        all_bbox_frames = []
        prev_bbox_frames = []
        width = 0
        height = 0

        # 人物ID
        track_id = -1
        # 次の人物ID
        next_id = 0
        # 動画のサイズ
        width = 0
        height = 0
        # 出現回数
        track_cnt_dict = {}

        for iidx, process_img_path in enumerate(tqdm(process_img_pathes)):
            # 人数分読み込む
            bbox_frames = {}
            now_bbox_frames = []
            joint_json_pathes = sorted(glob.glob(os.path.join(args.img_dir, "frames", f"{iidx:012}", "frame_*.json")), key=sort_by_numeric)

            if len(joint_json_pathes) == 0:
                # 人物が一件も見つからなかった場合
                now_bbox_frames = []
                all_bbox_frames.append([])

                continue
            
            # 一人以上人物が見つかった場合
            for joint_json_path in joint_json_pathes:
                with open(joint_json_path, 'r') as f:
                    bbox_frames[joint_json_path] = json.load(f)
                    # とりあえず初期値は追跡不能
                    bbox_frames[joint_json_path]["track_id"] = -1
                    width = bbox_frames[joint_json_path]['image']['width']
                    height = bbox_frames[joint_json_path]['image']['height']

                if median_w * 0.5 <= bbox_frames[joint_json_path]['bbox']['width'] and median_h * 0.5 <= bbox_frames[joint_json_path]['bbox']['height']:
                    # 中央値の半分以上の大きさである場合のみ、追跡対象とする

                    # enlarge bbox by 20% with same center position
                    bbox_x1y1x2y2 = xywh_to_x1y1x2y2_from_dict(bbox_frames[joint_json_path]['bbox'])
                    bbox_in_xywh = enlarge_bbox(bbox_x1y1x2y2, args.enlarge_scale, width, height)
                    bbox_det = x1y1x2y2_to_xywh(bbox_in_xywh)

                    # 関節は使えるのだけピックアップ
                    keypoints = []
                    for joint_name in ["pelvis", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "neck", "head", \
                                        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]:
                        keypoints.append((bbox_frames[joint_json_path]['proj_joints'][joint_name]['x'], bbox_frames[joint_json_path]['proj_joints'][joint_name]['y']))

                    if iidx == 0 or len(prev_bbox_frames) == 0:   # First frame, all ids are assigned automatically
                        track_id = next_id
                        next_id += 1
                    else:
                        # 姿勢もbboxも類似してるのを優先して追跡
                        track_id, match_index = get_track_id_SGCN(args, bbox_det, keypoints, prev_bbox_frames, pose_matcher)

                        if track_id > -1:  # if candidate from prev frame matched, prevent it from matching another
                            del prev_bbox_frames[match_index]
                            bbox_frames[joint_json_path]["track_id"] = track_id

                    if track_id > -1:
                        bbox_frames[joint_json_path]['track_id'] = track_id
                        if track_id not in track_cnt_dict:
                            # まだ出現なかったtrack_idの場合、場所用意
                            track_cnt_dict[track_id] = 0
                        # 出現回数カウント
                        track_cnt_dict[track_id] += 1

                    # JSON出力
                    with open(joint_json_path, "w") as f:
                        json.dump(bbox_frames[joint_json_path], f, indent=4)

                    # 今回分として保持
                    now_bbox_frames.append({'track_id': track_id, 'bbox': bbox_det, 'keypoints': keypoints, 'width': width, 'height': height, 'json_path': joint_json_path})

                else:
                    # tracking対象外でJSON出力
                    bbox_frames[joint_json_path]["track_id"] = -1
                    with open(joint_json_path, "w") as f:
                        json.dump(bbox_frames[joint_json_path], f, indent=4)

            for bidx, now_bbox_frame in enumerate(now_bbox_frames):
                joint_json_path = now_bbox_frame['json_path']

                if now_bbox_frame['track_id'] == -1:
                    # bboxベースの追跡再検討
                    track_id, match_index = get_track_id_SpatialConsistency(now_bbox_frame['bbox'], prev_bbox_frames)

                    if track_id > -1:
                        # 追跡できた場合、上書き
                        now_bbox_frame['track_id'] = track_id

                        if track_id not in track_cnt_dict:
                            # まだ出現なかったtrack_idの場合、場所用意
                            track_cnt_dict[track_id] = 0
                        # 出現回数カウント
                        track_cnt_dict[track_id] += 1

                        # JSON用データにも書き込み
                        bbox_frames[joint_json_path]['track_id'] = track_id

                    # if still can not find a match from previous frame, then assign a new id
                    if track_id == -1 and not bbox_invalid(now_bbox_frame['bbox'], width, height):
                        bbox_frames[joint_json_path]["track_id"] = next_id
                        now_bbox_frame["track_id"] = next_id
                        next_id += 1
                    else:
                        pass

                    # JSON出力
                    with open(joint_json_path, "w") as f:
                        json.dump(bbox_frames[joint_json_path], f, indent=4)

            # 前回分として保持しなおし
            prev_bbox_frames = now_bbox_frames
            # 全データを保持（前回データはヒット分を削除したりするのでコピー保持）
            all_bbox_frames.append(cPickle.loads(cPickle.dumps(now_bbox_frames, -1)))

        logger.info('追跡結果生成開始', decoration=MLogger.DECORATION_LINE)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(process_bbox_path, fourcc, 30.0, (width, height))

        for iidx, process_img_path in enumerate(tqdm(process_img_pathes)):
            out_frame = cv2.imread(process_img_path)

            if len(all_bbox_frames) >= iidx:
                for bbox_frame in all_bbox_frames[iidx]:
                    # track_idが管理出現回数とbboxサイズをクリアしている場合のみ画像出力
                    track_id = bbox_frame['track_id']
                    x1, y1, w, h = bbox_frame['bbox']
                    if track_id > -1 and track_id in track_cnt_dict \
                        and (track_cnt_dict[track_id] > 3 or (len(process_img_pathes) < 3 and track_cnt_dict[track_id] > 1)) \
                        and median_w * 0.5 <= w and median_h * 0.5 <= h:

                        # bbox描画
                        out_frame = draw_bbox(out_frame, bbox_frame['bbox'], 1, None, track_id=track_id)
            
            # フレーム番号追記
            cv2.putText(out_frame, f'{iidx:05}F', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.1, color=(182, 0, 182), thickness = 2, lineType = cv2.LINE_AA)

            # 画像出力
            cv2.imwrite(os.path.join(args.img_dir, "frames", f"{iidx:012}", f"bbox_{iidx:012}.png"), out_frame)

            # トラッキングmp4合成
            out.write(out_frame)

        out.release()
        cv2.destroyAllWindows()

        logger.info('人物追跡処理終了: {0}', process_bbox_path, decoration=MLogger.DECORATION_BOX)

        return True
    except Exception as e:
        logger.critical("人物追跡で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        return False


def x1y1x2y2_to_xywh(det):
    x1, y1, x2, y2 = det
    w, h = int(x2) - int(x1), int(y2) - int(y1)
    return [x1, y1, w, h]


def xywh_to_x1y1x2y2_from_dict(det):
    x1 = det['x']
    y1 = det['y']
    w = det['width']
    h = det['height']
    x2, y2 = x1 + w, y1 + h
    return [x1, y1, x2, y2]


def xywh_to_x1y1x2y2(det):
    x1, y1, w, h = det
    x2, y2 = x1 + w, y1 + h
    return [x1, y1, x2, y2]


def enlarge_bbox(bbox, scale, img_width, img_height):
    assert(scale > 0)
    min_x, min_y, max_x, max_y = bbox
    margin_x = int(0.5 * scale * (max_x - min_x))
    margin_y = int(0.5 * scale * (max_y - min_y))
    if margin_x < 0: margin_x = 2
    if margin_y < 0: margin_y = 2

    min_x -= margin_x
    max_x += margin_x
    min_y -= margin_y
    max_y += margin_y

    width = max_x - min_x
    height = max_y - min_y
    if max_y < 0 or max_x < 0 or width <= 0 or height <= 0:
        min_x=0
        max_x=2
        min_y=0
        max_y=2

    bbox_enlarged = [min_x, min_y, max_x, max_y]
    return bbox_enlarged


def bbox_invalid(bbox, img_width, img_height):
    if bbox == [0, 0, 2, 2]:
        return True
    # 範囲外でもbboxの可能性はあるのでエラーはなし
    # if bbox[2] <= 0 or bbox[3] <= 0 or bbox[2] > img_width or bbox[3] > img_height:
    #     return True
    return False


def get_track_id_SGCN(args, bbox_cur_frame, keypoints_cur_frame, prev_bbox_frames, pose_matcher):
    min_index = None
    min_matching_score = sys.maxsize
    # if track_id is still not assigned, the person is really missing or track is really lost
    track_id = -1
    pose_matching_threshold = 0.4

    # bboxベースの類似追跡ID
    similar_bbox_idxs = get_similar_bbox_track_id(bbox_cur_frame, prev_bbox_frames)

    # 類似bbox内だけチェック
    for det_index in similar_bbox_idxs:
        prev_bbox_frame = prev_bbox_frames[det_index]
        bbox_prev_frame = prev_bbox_frame["bbox"]

        # check the pose matching score
        keypoints_prev_frame = prev_bbox_frame["keypoints"]

        img_width = prev_bbox_frame["width"]
        img_height = prev_bbox_frame["height"]        

        pose_matching_score = get_pose_matching_score(keypoints_cur_frame, keypoints_prev_frame, bbox_cur_frame, bbox_prev_frame, img_width, img_height, pose_matcher)

        if pose_matching_score <= pose_matching_threshold and pose_matching_score <= min_matching_score:
            # match the target based on the pose matching score
            min_matching_score = pose_matching_score
            min_index = det_index

    if min_index is None:
        return -1, None
    else:
        track_id = prev_bbox_frames[min_index]["track_id"]
        return track_id, min_index


def get_pose_matching_score(keypoints_A, keypoints_B, bbox_A, bbox_B, img_width, img_height, pose_matcher):
    if keypoints_A == [] or keypoints_B == []:
        logger.debug("graph not correctly generated!")
        return sys.maxsize

    if bbox_invalid(bbox_A, img_width, img_height) or bbox_invalid(bbox_B, img_width, img_height):
        logger.debug("graph not correctly generated!")
        return sys.maxsize

    sample_graph_pair = (keypoints_A, keypoints_B)
    data_A, data_B = graph_pair_to_data(sample_graph_pair)

    flag_match, dist = pose_matcher.inference(data_A, data_B)

    # 合致してる場合は結果を返す
    return dist if flag_match else sys.maxsize


# bbox単位で似ている場所にある前回フレームのbboxを抽出
def get_similar_bbox_track_id(bbox_cur_frame, prev_bbox_frames):
    thresh = 0.4
    similar_bbox_idxs = []

    for bbox_index, bbox_det_dict in enumerate(prev_bbox_frames):
        bbox_prev_frame = bbox_det_dict["bbox"]

        boxA = xywh_to_x1y1x2y2(bbox_cur_frame)
        boxB = xywh_to_x1y1x2y2(bbox_prev_frame)
        iou_score = iou(boxA, boxB)
        if iou_score > thresh:
            similar_bbox_idxs.append(bbox_index)

    return similar_bbox_idxs

# 最もbboxが近いtrack_idを抽出
def get_track_id_SpatialConsistency(bbox_cur_frame, bbox_list_prev_frame):
    thresh = 0.6
    max_iou_score = 0
    max_index = -1

    for bbox_index, bbox_det_dict in enumerate(bbox_list_prev_frame):
        bbox_prev_frame = bbox_det_dict["bbox"]

        boxA = xywh_to_x1y1x2y2(bbox_cur_frame)
        boxB = xywh_to_x1y1x2y2(bbox_prev_frame)
        iou_score = iou(boxA, boxB)
        if iou_score > max_iou_score:
            max_iou_score = iou_score
            max_index = bbox_index

    if max_iou_score > thresh:
        track_id = bbox_list_prev_frame[max_index]["track_id"]
        return track_id, max_index
    else:
        return -1, None


def iou(boxA, boxB):
    # box: (x1, y1, x2, y2)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou
