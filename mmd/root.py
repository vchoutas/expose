# -*- coding: utf-8 -*-
import os
import glob
import json
import argparse
import math
import re

# import vision essentials
import numpy as np
from tqdm import tqdm
import cv2

import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

from mmd.utils.MLogger import MLogger
from mmd.utils.MServiceUtils import sort_by_numeric
from mmd.tracking import xywh_to_x1y1x2y2_from_dict, enlarge_bbox, x1y1x2y2_to_xywh

from root.model import get_pose_net
from root.utils.pose_utils import process_bbox
from root.data.dataset import generate_patch_image

logger = MLogger(__name__, level=MLogger.DEBUG)

def execute(args):
    try:
        logger.info('人物深度処理開始: {0}', args.img_dir, decoration=MLogger.DECORATION_BOX)

        if not os.path.exists(args.img_dir):
            logger.error("指定された処理用ディレクトリが存在しません。: {0}", args.img_dir, decoration=MLogger.DECORATION_BOX)
            return False

        parser = get_parser()
        argv = parser.parse_args(args=[])

        if not os.path.exists(argv.model_path):
            logger.error("指定された学習モデルが存在しません。: {0}", argv.model_path, decoration=MLogger.DECORATION_BOX)
            return False

        cudnn.benchmark = True

        # snapshot load
        model = get_pose_net(argv, False)
        model = DataParallel(model).to('cuda')
        ckpt = torch.load(argv.model_path)
        model.load_state_dict(ckpt['network'])
        model.eval()
        focal = [1500, 1500] # x-axis, y-axis

        # prepare input image
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=argv.pixel_mean, std=argv.pixel_std)])

        # 全人物分の順番別フォルダ
        ordered_person_dir_pathes = sorted(glob.glob(os.path.join(args.img_dir, "ordered", "*")), key=sort_by_numeric)

        frame_pattern = re.compile(r'^(frame_(\d+)\.png)')

        for oidx, ordered_person_dir_path in enumerate(ordered_person_dir_pathes):    
            logger.info("【No.{0}】人物深度推定開始", f"{oidx:03}", decoration=MLogger.DECORATION_LINE)

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
                        
                        width = int(frame_joints['image']['width'])
                        height = int(frame_joints['image']['height'])

                        original_img = cv2.imread(frame_image_path)

                        bx = float(frame_joints["bbox"]["x"])
                        by = float(frame_joints["bbox"]["y"])
                        bw = float(frame_joints["bbox"]["width"])
                        bh = float(frame_joints["bbox"]["height"])

                        # ROOT_NETで深度推定
                        bbox = process_bbox([bx, by, bw, bh], width, height, argv)
                        img, img2bb_trans = generate_patch_image(original_img, bbox, False, 0.0, argv)
                        img = transform(img).to('cuda')[None,:,:,:]
                        k_value = np.array([math.sqrt(argv.bbox_real[0] * argv.bbox_real[1] * focal[0] * focal[1] / (bbox[2] * bbox[3]))]).astype(np.float32)
                        k_value = torch.FloatTensor([k_value]).to('cuda')[None,:]

                        with torch.no_grad():
                            root_3d = model(img, k_value) # x,y: pixel, z: root-relative depth (mm)

                        img = img[0].to('cpu').numpy()
                        root_3d = root_3d[0].to('cpu').numpy()
                        root_3d[0] = root_3d[0] / argv.output_shape[0] * bbox[2] + bbox[0]
                        root_3d[1] = root_3d[1] / argv.output_shape[1] * bbox[3] + bbox[1]

                        frame_joints["root"] = {"x": float(root_3d[0]), "y": float(root_3d[1]), "z": float(root_3d[2]), \
                                                "input": {"x": argv.input_shape[0], "y": argv.input_shape[1]}, "output": {"x": argv.output_shape[0], "y": argv.output_shape[1]}, \
                                                "focal": {"x": focal[0], "y": focal[1]}}

                        with open(frame_json_path, 'w') as f:
                            json.dump(frame_joints, f, indent=4)

        logger.info('人物深度処理終了: {0}', args.img_dir, decoration=MLogger.DECORATION_BOX)

        return True
    except Exception as e:
        logger.critical("人物深度で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        return False


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', default="data/snapshot_18.pth.tar")
    parser.add_argument('--resnet_type', type=int, default=50, help="50, 101, 152")
    parser.add_argument('--input_shape', type=tuple, default=(256, 256))
    parser.add_argument('--output_shape', type=tuple, default=(256 // 4, 256 // 4))
    parser.add_argument('--pixel_mean', type=tuple, default=(0.485, 0.456, 0.406))
    parser.add_argument('--pixel_std', type=tuple, default=(0.229, 0.224, 0.225))
    parser.add_argument('--bbox_real', type=tuple, default=(2000, 2000), help="Human36M, MuCo, MuPoTS: (2000, 2000), PW3D: (2, 2)")
    parser.add_argument('--lr_dec_epoch', type=list, default=[17])
    parser.add_argument('--end_epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_dec_factor', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--use_gt_bbox', type=bool, default=True)
    parser.add_argument('--num_thread', type=int, default=8)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--continue_train', type=bool, default=False)
    parser.add_argument('--enlarge_scale', default=0.15)

    return parser

