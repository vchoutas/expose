# -*- coding: utf-8 -*-
import os
import glob
import re
import traceback
from tqdm import tqdm
import json
import cv2
import dlib
import numpy as np
from imutils import face_utils

from mmd.utils.MLogger import MLogger
from mmd.utils.MServiceUtils import sort_by_numeric

logger = MLogger(__name__)

def execute(args):
    try:
        logger.info('表情推定処理開始: {0}', args.img_dir, decoration=MLogger.DECORATION_BOX)

        if not os.path.exists(args.img_dir):
            logger.error("指定された処理用ディレクトリが存在しません。: {0}", args.img_dir, decoration=MLogger.DECORATION_BOX)
            return False

        if not os.path.exists(os.path.join(args.img_dir, "ordered")):
            logger.error("指定された順番ディレクトリが存在しません。\n順番指定が完了していない可能性があります。: {0}", \
                         os.path.join(args.img_dir, "ordered"), decoration=MLogger.DECORATION_BOX)
            return False

        # 全人物分の順番別フォルダ
        ordered_person_dir_pathes = sorted(glob.glob(os.path.join(args.img_dir, "ordered", "*")), key=sort_by_numeric)

        frame_pattern = re.compile(r'^(frame_(\d+)\.png)')

        # 表情推定detector
        detector = dlib.get_frontal_face_detector()

        for oidx, ordered_person_dir_path in enumerate(ordered_person_dir_pathes):    
            logger.info("【No.{0}】表情推定開始", f"{oidx:03}", decoration=MLogger.DECORATION_LINE)

            frame_json_pathes = sorted(glob.glob(os.path.join(ordered_person_dir_path, "frame_*.json")), key=sort_by_numeric)

            for frame_json_path in tqdm(frame_json_pathes, desc=f"No.{oidx:03} ... "):                
                m = frame_pattern.match(os.path.basename(frame_json_path))
                if m:
                    frame_image_name = str(m.groups()[0])
                    fno_name = str(m.groups()[1])
                    
                    # 該当フレームの画像パス
                    frame_image_path = os.path.join(args.img_dir, "frames", fno_name, frame_image_name)

                    if os.path.exists(frame_image_path):

                        frame_joints = {}
                        with open(frame_json_path, 'r') as f:
                            frame_joints = json.load(f)
                        
                        bbox_x = int(frame_joints["bbox"]["x"])
                        bbox_y = int(frame_joints["bbox"]["y"])
                        bbox_w = int(frame_joints["bbox"]["width"])
                        bbox_h = int(frame_joints["bbox"]["height"])

                        image = cv2.imread(frame_image_path)
                        # bboxの範囲でトリミング
                        image_trim = image[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w]

                        # 顔抽出
                        faces, _, _ = detector.run(image=image_trim, upsample_num_times=0, adjust_threshold=0.0)

                        if len(faces) > 0:
                            face = faces[0]

                            predictor = dlib.shape_predictor(args.face_model)

                            frame_joints["faces"] = {}

                            try:                                
                                landmarks = predictor(image_trim, face)
                                shape = face_utils.shape_to_np(landmarks)
                                j = 0    
                                for (x, y) in shape:
                                    j += 1
                                    frame_joints["faces"][j] = {"x": float(bbox_x+x), "y": float(bbox_y+y)}

                                # 目の重心を求める
                                left_cx, left_cy, left_eye_image = get_eye_point(image_trim, shape, True)
                                right_cx, right_cy, right_eye_image = get_eye_point(image_trim, shape, False)
                                frame_joints["eyes"] = {
                                    "left": {"x": bbox_x+left_cx, "y": bbox_y+left_cy}, 
                                    "right": {"x": bbox_x+right_cx, "y": bbox_y+right_cy}
                                }
                            except Exception as e:
                                logger.debug("表情推定失敗: fno: {0}\n\n{1}", fno_name, traceback.extract_stack(), decoration=MLogger.DECORATION_BOX)
                                
                            # cv2.imwrite(os.path.join(args.img_dir, "frames", fno_name, "pupil.png"), right_eye_image)

                            with open(frame_json_path, 'w') as f:
                                json.dump(frame_joints, f, indent=4)

        logger.info('表情推定処理終了: {0}', os.path.join(args.img_dir, "ordered"), decoration=MLogger.DECORATION_BOX)

        return True
    except Exception as e:
        logger.critical("表情推定で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        return False

# 瞳の重心を求める
# https://cppx.hatenablog.com/entry/2017/12/25/231121#%E7%9E%B3%E5%BA%A7%E6%A8%99%E3%82%92%E5%8F%96%E5%BE%97
def get_eye_point(img, parts, left=True):
    if not left:
        eyes = [
                parts[36],
                min(parts[37], parts[38], key=lambda x: x[1]),
                max(parts[40], parts[41], key=lambda x: x[1]),
                parts[39],
                ]
    else:
        eyes = [
                parts[42],
                min(parts[43], parts[44], key=lambda x: x[1]),
                max(parts[46], parts[47], key=lambda x: x[1]),
                parts[45],
                ]
    org_x = eyes[0][0]
    org_y = eyes[1][1]

    resultFrame = img

    windowClose = np.ones((5,5),np.uint8)
    windowOpen = np.ones((2,2),np.uint8)
    windowErode = np.ones((2,2),np.uint8)

    eye = img[org_y:eyes[2][1], org_x:eyes[-1][0]]
    cv2.rectangle(resultFrame, (int(org_x),int(org_y)), (int(eyes[-1][0]),int(eyes[2][1])), (0,0,255),1) 

    ret, pupilFrame = cv2.threshold(eye, 55,255,cv2.THRESH_BINARY)        #50 ..nothin 70 is better
    pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_CLOSE, windowClose)
    pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_ERODE, windowErode)
    pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_OPEN, windowOpen)

    _, eye = cv2.threshold(cv2.cvtColor(pupilFrame, cv2.COLOR_RGB2GRAY), 30, 255, cv2.THRESH_BINARY_INV)

    moments = cv2.moments(eye, False)
    try:
        cx, cy = moments['m10'] / moments['m00'], moments['m01'] / moments['m00']
        cv2.circle(resultFrame,(int(cx + org_x), int(cy + org_y)),5,(255,0,0),-1)

        return cx + org_x, cy + org_y, resultFrame
    except:
        pass

    return 0, 0, resultFrame
