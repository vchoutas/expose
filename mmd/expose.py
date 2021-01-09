# -*- coding: utf-8 -*-
from ast import parse
import os
import argparse
import pathlib

import sys
import os
import os.path as osp
from typing import List, Optional
import functools
import glob
import datetime

import numpy as np
from collections import OrderedDict, defaultdict
import cv2
import argparse
import time
from tqdm import tqdm
from threadpoolctl import threadpool_limits
import PIL.Image as pil_img
# import matplotlib.pyplot as plt
import json

import torch
import torch.utils.data as dutils
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.transforms import Compose, Normalize, ToTensor

from expose.data.datasets import ImageFolder, ImageFolderWithBoxes

from expose.data.targets.image_list import to_image_list
from expose.utils.checkpointer import Checkpointer
import expose.data.utils.bbox as bboxutils

from expose.data.build import collate_batch
from expose.data.transforms import build_transforms

from expose.models.smplx_net import SMPLXNet
from expose.config import cfg
from expose.config.cmd_parser import set_face_contour
# from expose.utils.plot_utils import HDRenderer

from expose.data.targets import Keypoints2D
from expose.data.targets.keypoints import body_model_to_dset, ALL_CONNECTIONS, KEYPOINT_NAMES, BODY_CONNECTIONS

# import vision essentials
import numpy as np
from tqdm import tqdm

from mmd.utils.MLogger import MLogger

# 指数表記なし、有効小数点桁数6、30を超えると省略あり、一行の文字数200
np.set_printoptions(suppress=True, precision=6, threshold=30, linewidth=200)

if os.name == 'posix':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

logger = MLogger(__name__, level=MLogger.DEBUG)


def execute(args):
    try:
        logger.info('人物姿勢推定開始: {0}', args.img_dir, decoration=MLogger.DECORATION_BOX)

        if not os.path.exists(args.img_dir):
            logger.error("指定された処理用ディレクトリが存在しません。: {0}", args.img_dir, decoration=MLogger.DECORATION_BOX)
            return False

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        parser = get_parser()
        argv = parser.parse_args(args=[])

        show = argv.show
        pause = argv.pause
        focal_length = argv.focal_length
        save_vis = argv.save_vis
        save_params = argv.save_params
        save_mesh = argv.save_mesh
        degrees = argv.degrees
        expose_batch = argv.expose_batch
        rcnn_batch = argv.rcnn_batch

        cfg.merge_from_file(argv.exp_cfg)
        cfg.merge_from_list(argv.exp_opts)

        cfg.datasets.body.batch_size = expose_batch

        cfg.is_training = False
        cfg.datasets.body.splits.test = argv.datasets
        use_face_contour = cfg.datasets.use_face_contour
        set_face_contour(cfg, use_face_contour=use_face_contour)

        output_folder = os.path.join(args.img_dir, "pose")

        result = False
        with threadpool_limits(limits=1):
            result = main(
                args, 
                cfg,
                show=show,
                output_folder=output_folder,
                pause=pause,
                focal_length=focal_length,
                save_vis=save_vis,
                save_mesh=save_mesh,
                save_params=save_params,
                degrees=degrees,
                rcnn_batch=rcnn_batch,
            )

        logger.info('人物姿勢推定終了: {0}', args.img_dir, decoration=MLogger.DECORATION_BOX)
        
        return result
    except Exception as e:
        logger.critical("姿勢推定で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        return False


@torch.no_grad()
def main(
    args,
    exp_cfg,
    show: bool = False,
    output_folder: str = 'pose',
    pause: float = -1,
    focal_length: float = 5000,
    rcnn_batch: int = 1,
    sensor_width: float = 36,
    save_vis: bool = False,
    save_params: bool = False,
    save_mesh: bool = False,
    degrees: Optional[List[float]] = [],
) -> bool:

    device = torch.device('cuda')
    if not torch.cuda.is_available():
        logger.error('CUDAが無効になっています')
        return False

    process_img_pathes = os.path.join(args.img_dir, "frames", "**", "frame_*.png")
    
    # 準備
    expose_dloader = preprocess_images(process_img_pathes, exp_cfg, batch_size=rcnn_batch, device=device)

    model = None
    try:
        model = SMPLXNet(exp_cfg)
        model = model.to(device=device)
    except RuntimeError:
        logger.error('学習モデルが解析出来ませんでした')
        return False

    output_folder = exp_cfg.output_folder
    checkpoint_folder = osp.join(output_folder, exp_cfg.checkpoint_folder)
    checkpointer = Checkpointer(model, save_dir=checkpoint_folder, pretrained=exp_cfg.pretrained)

    arguments = {'iteration': 0, 'epoch_number': 0}
    extra_checkpoint_data = checkpointer.load_checkpoint()
    for key in arguments:
        if key in extra_checkpoint_data:
            arguments[key] = extra_checkpoint_data[key]

    model = model.eval()

    means = np.array(exp_cfg.datasets.body.transforms.mean)
    std = np.array(exp_cfg.datasets.body.transforms.std)

    render = save_vis or show
    body_crop_size = exp_cfg.get('datasets', {}).get('body', {}).get('transforms').get('crop_size', 256)
    if render:
        hd_renderer = HDRenderer(img_size=body_crop_size)

    logger.info("姿勢推定開始（人物×フレーム数分）", decoration=MLogger.DECORATION_LINE)

    cnt = 0
    for bidx, batch in enumerate(tqdm(expose_dloader, dynamic_ncols=True)):

        full_imgs_list, body_imgs, body_targets = batch
        if full_imgs_list is None:
            continue

        full_imgs = to_image_list(full_imgs_list)
        body_imgs = body_imgs.to(device=device)
        body_targets = [target.to(device) for target in body_targets]
        full_imgs = full_imgs.to(device=device)
        camera_parameters = None
        camera_scale = None
        camera_transl = None

        model_output = model(body_imgs, body_targets, full_imgs=full_imgs,
                             device=device)
        cnt += 1

        body_imgs = body_imgs.detach().cpu().numpy()
        body_output = model_output.get('body')

        _, _, H, W = full_imgs.shape

        body_output = model_output.get('body', {})
        num_stages = body_output.get('num_stages', 3)
        stage_n_out = body_output.get(f'stage_{num_stages - 1:02d}', {})
        model_vertices = stage_n_out.get('vertices', None)

        if stage_n_out is not None:
            model_vertices = stage_n_out.get('vertices', None)

        # faces = stage_n_out['faces']
        if model_vertices is not None:
            # model_vertices = model_vertices.detach().cpu().numpy()
            camera_parameters = body_output.get('camera_parameters', {})
            camera_scale = camera_parameters['scale'].detach()
            camera_transl = camera_parameters['translation'].detach()

        out_img = OrderedDict()

        final_model_vertices = None
        stage_n_out = model_output.get('body', {}).get('final', {})
        if stage_n_out is not None:
            final_model_vertices = stage_n_out.get('vertices', None)

        if final_model_vertices is not None:
            # final_model_vertices = final_model_vertices.detach().cpu().numpy()
            camera_parameters = model_output.get('body', {}).get(
                'camera_parameters', {})
            camera_scale = camera_parameters['scale'].detach()
            camera_transl = camera_parameters['translation'].detach()

        hd_params = weak_persp_to_blender(
            body_targets,
            camera_scale=camera_scale,
            camera_transl=camera_transl,
            H=H, W=W,
            sensor_width=sensor_width,
            focal_length=focal_length,
        )

        # hd_imgs = full_imgs.images.detach().cpu().numpy().squeeze()
        # if render:
        #     hd_imgs = np.transpose(undo_img_normalization(hd_imgs, means, std),
        #                            [0, 2, 3, 1])
        #     hd_imgs = np.clip(hd_imgs, 0, 1.0)
        #     right_hand_crops = body_output.get('right_hand_crops')
        #     left_hand_crops = torch.flip(
        #         body_output.get('left_hand_crops'), dims=[-1])
        #     head_crops = body_output.get('head_crops')
        #     bg_imgs = undo_img_normalization(body_imgs, means, std)

        #     right_hand_crops = undo_img_normalization(
        #         right_hand_crops, means, std)
        #     left_hand_crops = undo_img_normalization(
        #         left_hand_crops, means, std)
        #     head_crops = undo_img_normalization(head_crops, means, std)

        # if save_vis:
        #     bg_hd_imgs = np.transpose(hd_imgs, [0, 3, 1, 2])
        #     out_img['hd_imgs'] = bg_hd_imgs
        # if render:
        #     # Render the initial predictions on the original image resolution
        #     hd_orig_overlays = hd_renderer(
        #         model_vertices, faces,
        #         focal_length=hd_params['focal_length_in_px'],
        #         camera_translation=hd_params['transl'],
        #         camera_center=hd_params['center'],
        #         bg_imgs=bg_hd_imgs,
        #         return_with_alpha=True,
        #     )            
        #     out_img['hd_orig_overlay'] = hd_orig_overlays

        # # Render the overlays of the final prediction
        # if render:
        #     hd_overlays = hd_renderer(
        #         final_model_vertices,
        #         faces,
        #         focal_length=hd_params['focal_length_in_px'],
        #         camera_translation=hd_params['transl'],
        #         camera_center=hd_params['center'],
        #         bg_imgs=bg_hd_imgs,
        #         return_with_alpha=True,
        #         body_color=[0.4, 0.4, 0.7]
        #     )
        #     out_img['hd_overlay'] = hd_overlays

        # if save_vis:
        #     for key in out_img.keys():
        #         out_img[key] = np.clip(
        #             np.transpose(
        #                 out_img[key], [0, 2, 3, 1]) * 255, 0, 255).astype(
        #                     np.uint8)

        camera_scale_np = camera_scale.cpu().numpy()
        camera_tansl_np = camera_transl.cpu().numpy()

        # bbox保持
        cbbox = body_targets[0].bbox.detach().cpu().numpy()
        bbox_size = np.array(body_targets[0].size)
        dset_center = np.array(body_targets[0].extra_fields['center'])
        dset_size = np.array(body_targets[0].extra_fields['bbox_size'])
        # 画面サイズに合わせる(描画のため、int)
        bbox = np.tile(dset_center, 2) + ((cbbox / np.tile(bbox_size, 2) - np.tile(0.5, 4)) * np.tile(dset_size, 4))
        img_bbox = bbox.astype(np.int)

        hd_params['img_bbox'] = bbox
        
        proj_joints = stage_n_out['proj_joints'][0].detach().cpu().numpy()
        hd_params['proj_joints'] = proj_joints

        for idx in range(len(body_targets)):
            fname = body_targets[idx].get_field('fname')
            idx_dir = body_targets[idx].get_field('idx_dir')

            params_json_path = osp.join(args.img_dir, "frames", idx_dir, f'{fname}_joints.json')

            out_params = dict(fname=fname)
            for key, val in stage_n_out.items():
                if torch.is_tensor(val):
                    val = val.detach().cpu().numpy()[idx]
                out_params[key] = val

            if save_vis:
                for name, curr_img in out_img.items():
                    pil_img.fromarray(curr_img[idx]).save(
                        osp.join(args.img_dir, "frames", idx_dir, f'{name}.png'))

            # json出力
            joint_dict = {}
            joint_dict["image"] = {"width": W, "height": H}
            joint_dict["depth"] = {"depth": float(hd_params["depth"][0][0])}
            joint_dict["camera"] = {"scale": float(camera_scale_np[0][0]), "transl": {"x": float(camera_tansl_np[0, 0]), "y": float(camera_tansl_np[0, 1])}}
            joint_dict["bbox"] = {"x": float(hd_params["img_bbox"][0]), "y": float(hd_params["img_bbox"][1]), \
                                  "width": float(hd_params["img_bbox"][2]) - float(hd_params["img_bbox"][0]), "height": float(hd_params["img_bbox"][3]) - float(hd_params["img_bbox"][1])}
            joint_dict["others"] = {'shift_x': float(hd_params["shift_x"][0]), 'shift_y': float(hd_params["shift_y"][0]), \
                                    'focal_length_in_mm': float(hd_params["focal_length_in_mm"][0]), 'focal_length_in_px': float(hd_params["focal_length_in_px"][0]), \
                                    'sensor_width': float(hd_params["sensor_width"][0]), 'center': {"x": float(hd_params['center'][0, 0]), "y": float(hd_params['center'][0, 1])}}
            joint_dict["joints"] = {}
            joint_dict["proj_joints"] = {}

            proj_joints = hd_params["proj_joints"]
            joints = out_params["joints"]
            min_joints = np.min(proj_joints, axis=0)
            max_joints = np.max(proj_joints, axis=0)
            diff_joints = max_joints - min_joints
            diff_bbox = np.array([hd_params['img_bbox'][2] - hd_params['img_bbox'][0], hd_params['img_bbox'][3] - hd_params['img_bbox'][1]])
            jscale = diff_joints / diff_bbox
            jscale = np.mean([jscale[0], jscale[1]])
            for jidx, jname in enumerate(KEYPOINT_NAMES):
                j2d = proj_joints[jidx] / jscale
                joint_dict["proj_joints"][jname] = {'x': float(hd_params['center'][0, 0] + j2d[0]), 'y': float(hd_params['center'][0, 1] + j2d[1])}
                joint_dict["joints"][jname] = {'x': float(joints[jidx][0]), 'y': float(-joints[jidx][1]), 'z': float(joints[jidx][2])}

            # for pose_name in ["global_orient", "body_pose", "left_hand_pose", "right_hand_pose", "jaw_pose"]:
            #     joint_dict[pose_name] = {}
            #     for pidx, pvalues in enumerate(out_params[pose_name]):
            #         joint_dict[pose_name][pidx] = {
            #             'xAxis': {'x': float(pvalues[0,0]), 'y': float(pvalues[0,1]), 'z': float(pvalues[0,2])},
            #             'yAxis': {'x': float(pvalues[1,0]), 'y': float(pvalues[1,1]), 'z': float(pvalues[1,2])},
            #             'zAxis': {'x': float(pvalues[2,0]), 'y': float(pvalues[2,1]), 'z': float(pvalues[2,2])}
            #         }

            with open(params_json_path, 'w') as f:
                json.dump(joint_dict, f, indent=4)

    return True


def collate_fn(batch):
    output_dict = dict()

    for d in batch:
        for key, val in d.items():
            if key not in output_dict:
                output_dict[key] = []
            output_dict[key].append(val)
    return output_dict


def preprocess_images(
    image_folder: str,
    exp_cfg,
    num_workers: int = 8, batch_size: int = 1,
    min_score: float = 0.5,
    scale_factor: float = 1.2,
    device: Optional[torch.device] = None
) -> dutils.DataLoader:

    if device is None:
        device = torch.device('cuda')
        if not torch.cuda.is_available():
            logger.error('CUDA is not available!')
            sys.exit(3)

    rcnn_model = keypointrcnn_resnet50_fpn(pretrained=True)
    rcnn_model.eval()
    rcnn_model = rcnn_model.to(device=device)

    transform = Compose(
        [ToTensor(), ]
    )

    # Load the images
    dataset = ImageFolder(image_folder, transforms=transform)
    rcnn_dloader = dutils.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        collate_fn=collate_fn
    )

    logger.info("姿勢推定準備", decoration=MLogger.DECORATION_LINE)

    img_paths = []
    bboxes = []
    for bidx, batch in enumerate(tqdm(rcnn_dloader)):
        batch['images'] = [x.to(device=device) for x in batch['images']]

        output = rcnn_model(batch['images'])
        for ii, x in enumerate(output):
            img = np.transpose(batch['images'][ii].detach().cpu().numpy(), [1, 2, 0])
            img = (img * 255).astype(np.uint8)

            img_path = batch['paths'][ii]
            _, fname = osp.split(img_path)
            fname, _ = osp.splitext(fname)

            #  out_path = osp.join(out_dir, f'{fname}_{ii:03d}.jpg')
            for n, bbox in enumerate(output[ii]['boxes']):
                bbox = bbox.detach().cpu().numpy()
                if output[ii]['scores'][n].item() < min_score:
                    continue
                img_paths.append(img_path)
                bboxes.append(bbox)

                #  cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[2:]),
                #  (255, 0, 0))
            #  cv2.imwrite(out_path, img[:, :, ::-1])

    dataset_cfg = exp_cfg.get('datasets', {})
    body_dsets_cfg = dataset_cfg.get('body', {})

    body_transfs_cfg = body_dsets_cfg.get('transforms', {})
    transforms = build_transforms(body_transfs_cfg, is_train=False)
    batch_size = body_dsets_cfg.get('batch_size', 64)

    expose_dset = ImageFolderWithBoxes(
        img_paths, bboxes, scale_factor=scale_factor, transforms=transforms)

    expose_collate = functools.partial(
        collate_batch, use_shared_memory=num_workers > 0,
        return_full_imgs=True)
    expose_dloader = dutils.DataLoader(
        expose_dset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=expose_collate,
        drop_last=False,
        pin_memory=True,
    )
    return expose_dloader


def weak_persp_to_blender(
        targets,
        camera_scale,
        camera_transl,
        H, W,
        sensor_width=36,
        focal_length=5000):
    ''' Converts weak-perspective camera to a perspective camera
    '''
    if torch.is_tensor(camera_scale):
        camera_scale = camera_scale.detach().cpu().numpy()
    if torch.is_tensor(camera_transl):
        camera_transl = camera_transl.detach().cpu().numpy()

    output = defaultdict(lambda: [])
    for ii, target in enumerate(targets):
        orig_bbox_size = target.get_field('orig_bbox_size')
        bbox_center = target.get_field('orig_center')
        z = 2 * focal_length / (camera_scale[ii] * orig_bbox_size)

        transl = [
            camera_transl[ii, 0].item(), camera_transl[ii, 1].item(),
            z.item()]
        shift_x = - (bbox_center[0] / W - 0.5)
        shift_y = (bbox_center[1] - 0.5 * H) / W
        focal_length_in_mm = focal_length / W * sensor_width
        output['shift_x'].append(shift_x)
        output['shift_y'].append(shift_y)
        output['transl'].append(transl)
        output['focal_length_in_mm'].append(focal_length_in_mm)
        output['focal_length_in_px'].append(focal_length)
        output['center'].append(bbox_center)
        output['depth'].append(z)
        output['sensor_width'].append(sensor_width)
    for key in output:
        output[key] = np.stack(output[key], axis=0)
    return output


def undo_img_normalization(image, mean, std, add_alpha=True):
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy().squeeze()

    out_img = (image * std[np.newaxis, :, np.newaxis, np.newaxis] +
               mean[np.newaxis, :, np.newaxis, np.newaxis])
    if add_alpha:
        out_img = np.pad(
            out_img, [[0, 0], [0, 1], [0, 0], [0, 0]],
            mode='constant', constant_values=1.0)
    return out_img


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp-cfg', type=str, dest='exp_cfg', default='config/expose-config.yaml', help='The configuration of the experiment')
    parser.add_argument('--exp-opts', default=[], dest='exp_opts', nargs='*', help='Extra command line arguments')
    parser.add_argument('--datasets', nargs='+', default=['openpose'], type=str, help='Datasets to process')
    parser.add_argument('--show', default=False, type=lambda arg: arg.lower() in ['true'], help='Display the results')
    parser.add_argument('--expose-batch', dest='expose_batch', default=1, type=int, help='ExPose batch size')
    parser.add_argument('--rcnn-batch', dest='rcnn_batch', default=1, type=int, help='R-CNN batch size')
    parser.add_argument('--pause', default=-1, type=float, help='How much to pause the display')
    parser.add_argument('--focal-length', dest='focal_length', type=float, default=5000, help='Focal length')
    parser.add_argument('--degrees', type=float, nargs='*', default=[], help='Degrees of rotation around the vertical axis')
    parser.add_argument('--save-vis', dest='save_vis', default=False, type=lambda x: x.lower() in ['true'], help='Whether to save visualizations')
    parser.add_argument('--save-mesh', dest='save_mesh', default=False, type=lambda x: x.lower() in ['true'], help='Whether to save meshes')
    parser.add_argument('--save-params', dest='save_params', default=False, type=lambda x: x.lower() in ['true'], help='Whether to save parameters')
    parser.add_argument('--verbose', type=int, dest='verbose', default=20, help='log level')

    return parser

