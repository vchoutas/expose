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
import mediapipe as mp

from mmd.utils.MLogger import MLogger
from mmd.utils.MServiceUtils import sort_by_numeric

logger = MLogger(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def execute(args):
    try:
        logger.info('mediapipe推定処理開始: {0}', args.img_dir, decoration=MLogger.DECORATION_BOX)

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


        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:

            for oidx, ordered_person_dir_path in enumerate(ordered_person_dir_pathes):    
                logger.info("【No.{0}】mediapipe推定開始", f"{oidx:03}", decoration=MLogger.DECORATION_LINE)

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

                            image_trim2 = cv2.cvtColor(cv2.flip(image_trim, 1), cv2.COLOR_BGR2RGB)
                            
                            # 一旦書き込み不可
                            image_trim2.flags.writeable = False
                            results = holistic.process(image_trim2)

                            if results.pose_landmarks and results.pose_landmarks.landmark and results.pose_world_landmarks and results.pose_world_landmarks.landmark:
                                frame_joints["mp_body_joints"] = {}
                                for landmark, world_landmark, output_name in zip(results.pose_landmarks.landmark, results.pose_world_landmarks.landmark, POSE_LANDMARKS):

                                    frame_joints["mp_body_joints"][f'mp_{output_name}'] = {'x': -float(landmark.x), 'y': -float(landmark.y), 'z': float(landmark.z), 
                                                                        'wx': -float(world_landmark.x), 'wy': -float(world_landmark.y), 'wz': float(world_landmark.z), 
                                                                        'visibility': float(landmark.visibility)}
                                
                            if results.right_hand_landmarks:
                                frame_joints["mp_left_hand_joints"] = {}
                                for landmark, output_name in zip(results.right_hand_landmarks.landmark, HAND_LANDMARKS):
                                    frame_joints["mp_left_hand_joints"][f'mp_left_{output_name}'] = {'x': -float(landmark.x), 'y': -float(landmark.y), 'z': float(landmark.z)}

                            if results.left_hand_landmarks:
                                frame_joints["mp_right_hand_joints"] = {}
                                for landmark, output_name in zip(results.left_hand_landmarks.landmark, HAND_LANDMARKS):
                                    frame_joints["mp_right_hand_joints"][f'mp_right_{output_name}'] = {'x': -float(landmark.x), 'y': -float(landmark.y), 'z': float(landmark.z)}

                            if results.face_landmarks:
                                frame_joints["mp_face_joints"] = {}
                                for lidx, landmark in enumerate(results.face_landmarks.landmark):
                                    frame_joints["mp_face_joints"][lidx] = {'x': -float(landmark.x), 'y': -float(landmark.y), 'z': float(landmark.z)}

                            with open(frame_json_path, 'w') as f:
                                json.dump(frame_joints, f, indent=4)

        logger.info('mediapipe推定処理終了: {0}', os.path.join(args.img_dir, "ordered"), decoration=MLogger.DECORATION_BOX)

        return True
    except Exception as e:
        logger.critical("mediapipe推定で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
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

# 左右逆
POSE_LANDMARKS = [
    'nose',
    'right_eye_inner',
    'right_eye',
    'right_eye_outer',
    'left_eye_inner',
    'left_eye',
    'left_eye_outer',
    'right_ear',
    'left_ear',
    'mouth_right',
    'mouth_left',
    'right_shoulder',
    'left_shoulder',
    'right_elbow',
    'left_elbow',
    'body_right_wrist',
    'body_left_wrist',
    'body_right_pinky',
    'body_left_pinky',
    'body_right_index',
    'body_left_index',
    'body_right_thumb',
    'body_left_thumb',
    'right_hip',
    'left_hip',
    'right_knee',
    'left_knee',
    'right_ankle',
    'left_ankle',
    'right_heel',
    'left_heel',
    'right_foot_index',
    'left_foot_index',
]

HAND_LANDMARKS = [
    "wrist", 
    'thumb1', 
    'thumb2', 
    'thumb3', 
    'thumb', 
    'index1', 
    'index2', 
    'index3', 
    'index', 
    'middle1', 
    'middle2', 
    'middle3', 
    'middle', 
    'ring1', 
    'ring2', 
    'ring3', 
    'ring', 
    'pinky1', 
    'pinky2',
    'pinky3',  
    'pinky', 
]
