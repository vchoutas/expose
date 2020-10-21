# -*- coding: utf-8 -*-
import os
import argparse
import glob
import sys
import json
import pathlib

# import vision essentials
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# detector utils
from lighttrack.detector.detector_yolov3 import inference_yolov3_from_img

# import GCN utils
from lighttrack.graph.visualize_pose_matching import graph_pair_to_data, pose_matching, keypoints_to_graph

# 姿勢推定用
from lighttrack.network_mobile_deconv import Network
from lighttrack.HPE.dataset import Preprocessing
from lighttrack.HPE.config import cfg
from lighttrack.lib.tfflat.base import Tester

from lighttrack.visualizer.detection_visualizer import draw_bbox

from miu.utils.MLogger import MLogger
from miu.utils.MServiceUtils import sort_by_numeric
import miu.config as mconfig

flag_flip = True
os.environ["CUDA_VISIBLE_DEVICES"]="0"

logger = MLogger(__name__, level=MLogger.DEBUG)

def execute(args):
    logger.info(f'人物追跡処理開始: {args.process_dir}', decoration=MLogger.DECORATION_BOX)

    if not os.path.exists(args.process_dir):
        logger.error("指定された処理用ディレクトリパスが存在しません。: {0}".format(args.process_dir))
        return

    args.bbox_thresh = 0.4
    args.test_model = "lighttrack/weights/mobile-deconv/snapshot_296.ckpt"

    pose_estimator = Tester(Network(), cfg)
    pose_estimator.load_weights(args.test_model)

    args.nms_method = 'nms'
    args.nms_thresh = 1.
    args.min_scores = 1e-10
    args.min_box_size = 0.
    args.draw_threshold = 0.2

    args.keyframe_interval = 40 # choice examples: [2, 3, 5, 8, 10, 20, 40, 100, ....]
    args.enlarge_scale = 0.2 # how much to enlarge the bbox before pose estimation
    args.pose_matching_threshold = 0.5

    process_bbox_path = os.path.join(args.process_dir, "bbox.mp4")
    process_img_path = os.path.join(args.process_dir, "**", "frame_*.png")
    process_img_paths = glob.glob(process_img_path)
    img_nums = len(process_img_paths)

    # process the frames sequentially
    bbox_dets_list = []
    frame_prev = -1
    frame_cur = 0
    img_id = -1
    next_id = 0
    bbox_dets_list_list = []
    bbox_list_prev_frame = None
    num_dets = 0
    width = 0
    height = 0
    
    for img_id in tqdm(range(img_nums)):
        img_path = process_img_paths[img_id]
        out_frame = cv2.imread(img_path)
        width = out_frame.shape[1]
        height = out_frame.shape[0]

        bbox_json_path = os.path.join(str(pathlib.Path(img_path).parent), f"bbox_{img_id:012}.json")
        bbox_img_path = os.path.join(str(pathlib.Path(img_path).parent), f"bbox_{img_id:012}.png")

        bbox_dets_list = []  # keyframe: start from empty

        # bbox抽出
        human_candidates = inference_yolov3_from_img(out_frame)
        num_dets = len(human_candidates)

        # bboxが見つからなかった場合、一旦INDEXクリア
        if num_dets <= 0:
            # add empty result
            bbox_det_dict = {"img_id":img_id,
                                "det_id":  0,
                                "track_id": None,
                                "imgpath": img_path,
                                "bbox": [0, 0, 2, 2]}
            bbox_dets_list.append(bbox_det_dict)
            bbox_dets_list_list.append(bbox_dets_list)

            flag_mandatory_keyframe = True
            continue

        if img_id > 0:   # First frame does not have previous frame
            bbox_list_prev_frame = bbox_dets_list_list[img_id - 1].copy()

        # For each candidate, perform pose estimation and data association based on Spatial Consistency (SC)
        for det_id in range(num_dets):
            # obtain bbox position and track id
            bbox_det = human_candidates[det_id]

            # enlarge bbox by 20% with same center position
            bbox_x1y1x2y2 = xywh_to_x1y1x2y2(bbox_det)
            bbox_in_xywh = enlarge_bbox(bbox_x1y1x2y2, args.enlarge_scale)
            bbox_det = x1y1x2y2_to_xywh(bbox_in_xywh)

            # Keyframe: use provided bbox
            if bbox_invalid(bbox_det):
                track_id = None # this id means null
                keypoints = []
                bbox_det = [0, 0, 2 ,2]
                # update current frame bbox
                bbox_det_dict = {"img_id":img_id,
                                    "det_id":det_id,
                                    "track_id": track_id,
                                    "imgpath": img_path,
                                    "bbox":bbox_det}
                bbox_dets_list.append(bbox_det_dict)
                continue

            if img_id == 0:   # First frame, all ids are assigned automatically
                track_id = next_id
                next_id += 1
            else:
                track_id, match_index = get_track_id_SpatialConsistency(bbox_det, bbox_list_prev_frame)

                if track_id != -1:  # if candidate from prev frame matched, prevent it from matching another
                    del bbox_list_prev_frame[match_index]

            # update current frame bbox
            bbox_det_dict = {"img_id":img_id,
                                "det_id":det_id,
                                "track_id":track_id,
                                "imgpath": img_path,
                                "bbox":bbox_det}
            bbox_dets_list.append(bbox_det_dict)

        keypoints_list_prev_frame = []

        if bbox_list_prev_frame and len(bbox_list_prev_frame) > 0:
            # まだ残っている場合、前回分のポーズ解析を行う
            for bbox_det_dict in bbox_list_prev_frame:
                keypoints = inference_keypoints(pose_estimator, bbox_det_dict)[0]["keypoints"]

                # update current frame keypoints
                keypoints_dict = {"img_id":img_id,
                                  "det_id":bbox_det_dict["det_id"],
                                  "track_id":bbox_det_dict["track_id"],
                                  "imgpath": img_path,
                                  "keypoints":keypoints}
                keypoints_list_prev_frame.append(keypoints_dict)

        # For candidate that is not assopciated yet, perform data association based on Pose Similarity (SGCN)
        for det_id in range(num_dets):
            bbox_det_dict = bbox_dets_list[det_id]
            assert(det_id == bbox_det_dict["det_id"])

            if bbox_det_dict["track_id"] == -1:    # this id means matching not found yet
                # この時だけ、ポーズ解析
                keypoints = inference_keypoints(pose_estimator, bbox_det_dict)[0]["keypoints"]

                track_id, match_index = get_track_id_SGCN(args, bbox_det_dict["bbox"], bbox_list_prev_frame,
                                                          keypoints, keypoints_list_prev_frame)

                if track_id != -1:  # if candidate from prev frame matched, prevent it from matching another
                    del bbox_list_prev_frame[match_index]
                    bbox_det_dict["track_id"] = track_id

                # if still can not find a match from previous frame, then assign a new id
                if track_id == -1 and not bbox_invalid(bbox_det_dict["bbox"]):
                    bbox_det_dict["track_id"] = next_id
                    next_id += 1
        
        # if bbox_list_prev_frame and len(bbox_list_prev_frame) > 0:
        #     # bboxが抽出できなかった場合、前回人物が残るので、とりあえず引き継ぐ
        #     for bbox_det_dict in bbox_list_prev_frame:
        #         bbox_dets_list.append(bbox_det_dict)

        # JSONデータ整形
        json_data = pose_to_standard_mot(img_id, bbox_dets_list)

        # JSON出力
        with open(bbox_json_path, "w") as f:
            json.dump(json_data, f, indent=4)

        # bbox描画
        for det_id in range(num_dets):
            bbox_det_dict = bbox_dets_list[det_id]

            candidates = json_data["candidates"]
            for candidate in candidates:
                bbox = np.array(candidate["det_bbox"]).astype(int)
                score = candidate["det_score"]
                if score < args.draw_threshold: continue

                track_id = candidate["track_id"]
                out_frame = draw_bbox(out_frame, bbox, score, None, track_id = track_id)

        # 画像出力
        cv2.imwrite(bbox_img_path, out_frame)

        # update frame
        bbox_dets_list_list.append(bbox_dets_list)
        frame_prev = frame_cur

    logger.info(f'トラッキング結果生成:', decoration=MLogger.DECORATION_LINE)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(process_bbox_path, fourcc, 30.0, (width, height))

    # トラッキングmp4合成
    for file_path in tqdm(sorted(glob.glob(os.path.join(args.process_dir, "**", "bbox_*.png")), key=sort_by_numeric)):
        # フレーム
        frame = cv2.imread(file_path)
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()

    logger.info(f'人物追跡処理終了: {process_bbox_path}', decoration=MLogger.DECORATION_BOX)


def inference_keypoints(pose_estimator, test_data):
    cls_dets = test_data["bbox"]
    test_data = [test_data]

    # crop and detect pose
    pose_heatmaps, details, cls_skeleton, crops, start_id, end_id = get_pose_from_bbox(pose_estimator, test_data, cfg)
    # get keypoint positions from pose
    keypoints = get_keypoints_from_pose(pose_heatmaps, details, cls_skeleton, crops, start_id, end_id)
    # dump results
    results = prepare_results(test_data[0], keypoints, cls_dets)
    return results


def get_pose_from_bbox(pose_estimator, test_data, cfg):
    cls_skeleton = np.zeros((len(test_data), cfg.nr_skeleton, 3))
    crops = np.zeros((len(test_data), 4))

    batch_size = 1
    start_id = 0
    end_id = min(len(test_data), batch_size)

    test_imgs = []
    details = []
    for i in range(start_id, end_id):
        test_img, detail = Preprocessing(test_data[i], stage='test')
        test_imgs.append(test_img)
        details.append(detail)

    details = np.asarray(details)
    feed = test_imgs
    for i in range(end_id - start_id):
        ori_img = test_imgs[i][0].transpose(1, 2, 0)
        if flag_flip == True:
            flip_img = cv2.flip(ori_img, 1)
            feed.append(flip_img.transpose(2, 0, 1)[np.newaxis, ...])
    feed = np.vstack(feed)

    res = pose_estimator.predict_one([feed.transpose(0, 2, 3, 1).astype(np.float32)])[0]
    res = res.transpose(0, 3, 1, 2)

    if flag_flip == True:
        for i in range(end_id - start_id):
            fmp = res[end_id - start_id + i].transpose((1, 2, 0))
            fmp = cv2.flip(fmp, 1)
            fmp = list(fmp.transpose((2, 0, 1)))
            for (q, w) in cfg.symmetry:
                fmp[q], fmp[w] = fmp[w], fmp[q]
            fmp = np.array(fmp)
            res[i] += fmp
            res[i] /= 2

    pose_heatmaps = res
    return pose_heatmaps, details, cls_skeleton, crops, start_id, end_id


def get_keypoints_from_pose(pose_heatmaps, details, cls_skeleton, crops, start_id, end_id):
    res = pose_heatmaps
    for test_image_id in range(start_id, end_id):
        r0 = res[test_image_id - start_id].copy()
        r0 /= 255.
        r0 += 0.5

        for w in range(cfg.nr_skeleton):
            res[test_image_id - start_id, w] /= np.amax(res[test_image_id - start_id, w])

        border = 10
        dr = np.zeros((cfg.nr_skeleton, cfg.output_shape[0] + 2 * border, cfg.output_shape[1] + 2 * border))
        dr[:, border:-border, border:-border] = res[test_image_id - start_id][:cfg.nr_skeleton].copy()

        for w in range(cfg.nr_skeleton):
            dr[w] = cv2.GaussianBlur(dr[w], (21, 21), 0)

        for w in range(cfg.nr_skeleton):
            lb = dr[w].argmax()
            y, x = np.unravel_index(lb, dr[w].shape)
            dr[w, y, x] = 0
            lb = dr[w].argmax()
            py, px = np.unravel_index(lb, dr[w].shape)
            y -= border
            x -= border
            py -= border + y
            px -= border + x
            ln = (px ** 2 + py ** 2) ** 0.5
            delta = 0.25
            if ln > 1e-3:
                x += delta * px / ln
                y += delta * py / ln
            x = max(0, min(x, cfg.output_shape[1] - 1))
            y = max(0, min(y, cfg.output_shape[0] - 1))
            cls_skeleton[test_image_id, w, :2] = (x * 4 + 2, y * 4 + 2)
            cls_skeleton[test_image_id, w, 2] = r0[w, int(round(y) + 1e-10), int(round(x) + 1e-10)]

        # map back to original images
        crops[test_image_id, :] = details[test_image_id - start_id, :]
        for w in range(cfg.nr_skeleton):
            cls_skeleton[test_image_id, w, 0] = cls_skeleton[test_image_id, w, 0] / cfg.data_shape[1] * (crops[test_image_id][2] - crops[test_image_id][0]) + crops[test_image_id][0]
            cls_skeleton[test_image_id, w, 1] = cls_skeleton[test_image_id, w, 1] / cfg.data_shape[0] * (crops[test_image_id][3] - crops[test_image_id][1]) + crops[test_image_id][1]
    return cls_skeleton


def prepare_results(test_data, cls_skeleton, cls_dets):
    cls_partsco = cls_skeleton[:, :, 2].copy().reshape(-1, cfg.nr_skeleton)

    cls_scores = 1
    dump_results = []
    cls_skeleton = np.concatenate(
        [cls_skeleton.reshape(-1, cfg.nr_skeleton * 3), (cls_scores * cls_partsco.mean(axis=1))[:, np.newaxis]],
        axis=1)
    for i in range(len(cls_skeleton)):
        result = dict(image_id=test_data['img_id'],
                      category_id=1,
                      score=float(round(cls_skeleton[i][-1], 4)),
                      keypoints=cls_skeleton[i][:-1].round(3).tolist())
        dump_results.append(result)
    return dump_results


def pose_to_standard_mot(img_id, bbox_dets_list):
    num_dets = len(bbox_dets_list)
    json_data = {}
    json_data["id"] = img_id

    candidate_list = []

    for j in range(num_dets):
        bbox_det_dict = bbox_dets_list[j]
        dets_dict = bbox_dets_list[j]

        img_id = bbox_det_dict["img_id"]
        det_id = bbox_det_dict["det_id"]
        track_id = bbox_det_dict["track_id"]

        bbox_dets_data = bbox_dets_list[det_id]
        det = dets_dict["bbox"]
        if  det == [0, 0, 2, 2]:
            # do not provide keypoints
            candidate = {"det_bbox": [0, 0, 2, 2],
                            "det_score": 0}
        else:
            bbox_in_xywh = det[0:4]
            track_score = 0

            candidate = {"det_bbox": bbox_in_xywh,
                         "det_score": 1,
                         "track_id": track_id,
                         "track_score": track_score}
        candidate_list.append(candidate)

    json_data["candidates"] = candidate_list

    return json_data


def get_pose_matching_score(keypoints_A, keypoints_B, bbox_A, bbox_B):
    if keypoints_A == [] or keypoints_B == []:
        print("graph not correctly generated!")
        return sys.maxsize

    if bbox_invalid(bbox_A) or bbox_invalid(bbox_B):
        print("graph not correctly generated!")
        return sys.maxsize

    graph_A, flag_pass_check = keypoints_to_graph(keypoints_A, bbox_A)
    if flag_pass_check is False:
        print("graph not correctly generated!")
        return sys.maxsize

    graph_B, flag_pass_check = keypoints_to_graph(keypoints_B, bbox_B)
    if flag_pass_check is False:
        print("graph not correctly generated!")
        return sys.maxsize

    sample_graph_pair = (graph_A, graph_B)
    data_A, data_B = graph_pair_to_data(sample_graph_pair)

    flag_match, dist = pose_matching(data_A, data_B)
    return dist


def get_track_id_SGCN(args, bbox_cur_frame, bbox_list_prev_frame, keypoints_cur_frame, keypoints_list_prev_frame):
    assert(len(bbox_list_prev_frame) == len(keypoints_list_prev_frame))

    min_index = None
    min_matching_score = sys.maxsize
    # if track_id is still not assigned, the person is really missing or track is really lost
    track_id = -1

    for det_index, bbox_det_dict in enumerate(bbox_list_prev_frame):
        bbox_prev_frame = bbox_det_dict["bbox"]

        # check the pose matching score
        keypoints_dict = keypoints_list_prev_frame[det_index]
        keypoints_prev_frame = keypoints_dict["keypoints"]
        pose_matching_score = get_pose_matching_score(keypoints_cur_frame, keypoints_prev_frame, bbox_cur_frame, bbox_prev_frame)

        if pose_matching_score <= args.pose_matching_threshold and pose_matching_score <= min_matching_score:
            # match the target based on the pose matching score
            min_matching_score = pose_matching_score
            min_index = det_index

    if min_index is None:
        return -1, None
    else:
        track_id = bbox_list_prev_frame[min_index]["track_id"]
        return track_id, min_index


def get_track_id_SpatialConsistency(bbox_cur_frame, bbox_list_prev_frame):
    thresh = 0.3
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


def x1y1x2y2_to_xywh(det):
    x1, y1, x2, y2 = det
    w, h = int(x2) - int(x1), int(y2) - int(y1)
    return [x1, y1, w, h]


def xywh_to_x1y1x2y2(det):
    x1, y1, w, h = det
    x2, y2 = x1 + w, y1 + h
    return [x1, y1, x2, y2]


def enlarge_bbox(bbox, scale):
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
    if max_y < 0 or max_x < 0 or width <= 0 or height <= 0 or width > 2000 or height > 2000:
        min_x=0
        max_x=2
        min_y=0
        max_y=2

    bbox_enlarged = [min_x, min_y, max_x, max_y]
    return bbox_enlarged

def bbox_invalid(bbox):
    if bbox == [0, 0, 2, 2]:
        return True
    if bbox[2] <= 0 or bbox[3] <= 0 or bbox[2] > 2000 or bbox[3] > 2000:
        return True
    return False



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--process-dir', type=str, dest='process_dir', help='process dir path')
    parser.add_argument('--verbose', type=int, dest='verbose', default=20, help='log level')

    args = parser.parse_args()

    MLogger.initialize(level=args.verbose, is_file=True)

    execute(args)

