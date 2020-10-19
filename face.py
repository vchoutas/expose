# -*- coding: utf-8 -*-
from os import name
import numpy as np
import argparse
import os
import cv2
import math
import PIL.Image as pil_img
import matplotlib.pyplot as plt
from miu.mmd.VmdWriter import VmdWriter
import os.path as osp
import json
import glob
import dlib
from imutils import face_utils
from expose.data.targets.keypoints import ALL_CONNECTIONS, KEYPOINT_NAMES, FACE_CONNECTIONS

from miu.module.MMath import MMatrix4x4, MQuaternion, MVector3D
from miu.mmd.VmdData import VmdBoneFrame, VmdMotion, VmdShowIkFrame, VmdInfoIk
from miu.mmd.PmxData import PmxModel, Bone, Vertex
from miu.utils.MServiceUtils import get_file_encoding, calc_global_pos, separate_local_qq
from miu.utils.MLogger import MLogger

logger = MLogger(__name__, level=MLogger.DEBUG)

def execute(cmd_args):
    detector = dlib.get_frontal_face_detector()

    #these landmarks are based on the image above 
    left_eye_landmarks  = [36, 37, 38, 39, 40, 41]
    right_eye_landmarks = [42, 43, 44, 45, 46, 47]

    # 画像フォルダ作成
    image_folder = osp.join(osp.dirname(osp.abspath(cmd_args.video_path)), osp.basename(cmd_args.video_path).replace(".", "_"), "face")
    os.makedirs(image_folder, exist_ok=True)
    
    # 動画を静画に変えて出力
    idx = 0
    cap = cv2.VideoCapture(cmd_args.video_path)
    while (cap.isOpened()):
        logger.info(f"■ idx: {idx} -----")

        # 動画から1枚キャプチャして読み込む
        flag, frame = cap.read()  # Capture frame-by-frame
        
        # 終わったフレームより後は飛ばす
        # 明示的に終わりが指定されている場合、その時も終了する
        if flag == False:
            break
            
        faces, _, _ = detector.run(image = frame, upsample_num_times = 0, adjust_threshold = 0.0)
        
        predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")

        # json出力
        joint_dict = {}

        for fidx, face in enumerate(faces):
            joint_dict[fidx] = {}
            joint_dict[fidx]["faces"] = {}

            landmarks = predictor(frame, face)
            shape = face_utils.shape_to_np(landmarks)
            (x, y, w, h) = face_utils.rect_to_bb(face)
            j = 0    
            for (x, y) in shape:
                j = j + 1
                cv2.putText(frame, str(j), (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                joint_dict[fidx]["faces"][j] = {"x": float(x), "y": float(y)}

            #-----Step 5: Calculating blink ratio for one eye-----
            left_eye_ratio  = get_blink_ratio(left_eye_landmarks, landmarks)
            right_eye_ratio = get_blink_ratio(right_eye_landmarks, landmarks)
            blink_ratio     = (left_eye_ratio + right_eye_ratio) / 2

            cv2.putText(frame,str(blink_ratio),(10,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),2,cv2.LINE_AA)

        cv2.imwrite(osp.join(image_folder, f"capture_{idx:012d}.png"), frame)

        with open(osp.join(image_folder, f"faces_{idx:012d}.json"), 'w') as f:
            json.dump(joint_dict, f, indent=4)

        idx += 1
    
    cap.release()

#-----Step 5: Getting to know blink ratio

def midpoint(point1 ,point2):
    return (point1.x + point2.x)/2,(point1.y + point2.y)/2

def euclidean_distance(point1 , point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def get_blink_ratio(eye_points, facial_landmarks):
    
    #loading all the required points
    corner_left  = (facial_landmarks.part(eye_points[0]).x, 
                    facial_landmarks.part(eye_points[0]).y)
    corner_right = (facial_landmarks.part(eye_points[3]).x, 
                    facial_landmarks.part(eye_points[3]).y)
    
    center_top    = midpoint(facial_landmarks.part(eye_points[1]), 
                             facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), 
                             facial_landmarks.part(eye_points[4]))

    #calculating distance
    horizontal_length = euclidean_distance(corner_left,corner_right)
    vertical_length = euclidean_distance(center_top,center_bottom)

    ratio = horizontal_length / vertical_length

    return ratio


if __name__ == '__main__':
    arg_formatter = argparse.ArgumentDefaultsHelpFormatter
    description = 'VMD Output'

    parser = argparse.ArgumentParser(formatter_class=arg_formatter,
                                     description=description)

    parser.add_argument('--video-path', type=str, dest='video_path', help='The folder with joint json')
    parser.add_argument('--verbose', type=int, dest='verbose', default=20, help='The csv file pmx born')

    cmd_args = parser.parse_args()

    MLogger.initialize(level=cmd_args.verbose, is_file=True)

    execute(cmd_args)
