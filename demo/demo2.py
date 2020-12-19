# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de



import sys
import os
import os.path as osp
from typing import List, Optional
import functools
import glob
import datetime
from matplotlib.pyplot import ylabel
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
from collections import OrderedDict, defaultdict
from loguru import logger
import cv2
import argparse
import time
import open3d as o3d
from tqdm import tqdm
from threadpoolctl import threadpool_limits
import PIL.Image as pil_img
import matplotlib.pyplot as plt
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
from expose.utils.plot_utils import HDRenderer

from expose.data.targets import Keypoints2D
from expose.data.targets.keypoints import body_model_to_dset, ALL_CONNECTIONS, KEYPOINT_NAMES, BODY_CONNECTIONS

# 指数表記なし、有効小数点桁数6、30を超えると省略あり、一行の文字数200
np.set_printoptions(suppress=True, precision=6, threshold=30, linewidth=200)

if os.name == 'posix':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))


Vec3d = o3d.utility.Vector3dVector
Vec3i = o3d.utility.Vector3iVector


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

    img_paths = []
    bboxes = []
    for bidx, batch in enumerate(
            tqdm(rcnn_dloader, desc='Processing with R-CNN')):
        batch['images'] = [x.to(device=device) for x in batch['images']]

        output = rcnn_model(batch['images'])
        for ii, x in enumerate(output):
            img = np.transpose(
                batch['images'][ii].detach().cpu().numpy(), [1, 2, 0])
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

@torch.no_grad()
def main(
    video_path: str,
    exp_cfg,
    show: bool = False,
    demo_output_folder: str = 'demo_output',
    pause: float = -1,
    focal_length: float = 5000,
    rcnn_batch: int = 1,
    sensor_width: float = 36,
    save_vis: bool = True,
    save_params: bool = False,
    save_mesh: bool = False,
    degrees: Optional[List[float]] = [],
) -> None:

    device = torch.device('cuda')
    if not torch.cuda.is_available():
        logger.error('CUDA is not available!')
        sys.exit(3)

    logger.remove()
    logger.add(lambda x: tqdm.write(x, end=''),
               level=exp_cfg.logger_level.upper(),
               colorize=True)
    
    # 画像フォルダ作成
    image_folder = osp.join(osp.dirname(osp.abspath(video_path)), osp.basename(video_path).replace(".", "_"))
    os.makedirs(image_folder, exist_ok=True)
    
    # 動画を静画に変えて出力
    idx = 0
    cap = cv2.VideoCapture(video_path)
    # 幅と高さを取得
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while(cap.isOpened()):
        # 動画から1枚キャプチャして読み込む
        flag, frame = cap.read()  # Capture frame-by-frame
            
        # 終わったフレームより後は飛ばす
        # 明示的に終わりが指定されている場合、その時も終了する
        if flag == False:
            break
            
        cv2.imwrite(osp.join(image_folder, "capture_{0:012d}.png".format(idx)), frame)
        idx += 1
    
    cap.release()

    expose_dloader = preprocess_images(
        image_folder + "/*.png", exp_cfg, batch_size=rcnn_batch, device=device)

    demo_output_folder = osp.join(osp.expanduser(osp.expandvars(demo_output_folder)), osp.basename(video_path).replace(".", "_"), datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logger.info(f'Saving results to: {demo_output_folder}')
    os.makedirs(demo_output_folder, exist_ok=True)

    #関節位置情報ファイル
    posf = open(osp.join(demo_output_folder, 'pos.txt'), 'w')

    model = SMPLXNet(exp_cfg)
    try:
        model = model.to(device=device)
    except RuntimeError:
        # Re-submit in case of a device error
        sys.exit(3)

    output_folder = exp_cfg.output_folder
    checkpoint_folder = osp.join(output_folder, exp_cfg.checkpoint_folder)
    checkpointer = Checkpointer(
        model, save_dir=checkpoint_folder, pretrained=exp_cfg.pretrained)

    arguments = {'iteration': 0, 'epoch_number': 0}
    extra_checkpoint_data = checkpointer.load_checkpoint()
    for key in arguments:
        if key in extra_checkpoint_data:
            arguments[key] = extra_checkpoint_data[key]

    model = model.eval()

    means = np.array(exp_cfg.datasets.body.transforms.mean)
    std = np.array(exp_cfg.datasets.body.transforms.std)

    render = save_vis or show
    body_crop_size = exp_cfg.get('datasets', {}).get('body', {}).get(
        'transforms').get('crop_size', 256)
    if render:
        hd_renderer = HDRenderer(img_size=body_crop_size)

    total_time = 0
    cnt = 0
    for bidx, batch in enumerate(tqdm(expose_dloader, dynamic_ncols=True)):

        full_imgs_list, body_imgs, body_targets = batch
        if full_imgs_list is None:
            continue

        full_imgs = to_image_list(full_imgs_list)
        body_imgs = body_imgs.to(device=device)
        body_targets = [target.to(device) for target in body_targets]
        full_imgs = full_imgs.to(device=device)

        torch.cuda.synchronize()
        start = time.perf_counter()
        model_output = model(body_imgs, body_targets, full_imgs=full_imgs,
                             device=device)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        cnt += 1
        total_time += elapsed

        hd_imgs = full_imgs.images.detach().cpu().numpy().squeeze()
        body_imgs = body_imgs.detach().cpu().numpy()
        body_output = model_output.get('body')

        _, _, H, W = full_imgs.shape
        #  logger.info(f'{H}, {W}')
        #  H, W, _ = hd_imgs.shape
        if render:
            hd_imgs = np.transpose(undo_img_normalization(hd_imgs, means, std),
                                   [0, 2, 3, 1])
            hd_imgs = np.clip(hd_imgs, 0, 1.0)
            right_hand_crops = body_output.get('right_hand_crops')
            left_hand_crops = torch.flip(
                body_output.get('left_hand_crops'), dims=[-1])
            head_crops = body_output.get('head_crops')
            bg_imgs = undo_img_normalization(body_imgs, means, std)

            right_hand_crops = undo_img_normalization(
                right_hand_crops, means, std)
            left_hand_crops = undo_img_normalization(
                left_hand_crops, means, std)
            head_crops = undo_img_normalization(head_crops, means, std)

        body_output = model_output.get('body', {})
        num_stages = body_output.get('num_stages', 3)
        stage_n_out = body_output.get(f'stage_{num_stages - 1:02d}', {})
        model_vertices = stage_n_out.get('vertices', None)

        if stage_n_out is not None:
            model_vertices = stage_n_out.get('vertices', None)

        faces = stage_n_out['faces']
        if model_vertices is not None:
            model_vertices = model_vertices.detach().cpu().numpy()
            camera_parameters = body_output.get('camera_parameters', {})
            camera_scale = camera_parameters['scale'].detach()
            camera_transl = camera_parameters['translation'].detach()

        out_img = OrderedDict()

        final_model_vertices = None
        stage_n_out = model_output.get('body', {}).get('final', {})
        if stage_n_out is not None:
            final_model_vertices = stage_n_out.get('vertices', None)

        if final_model_vertices is not None:
            final_model_vertices = final_model_vertices.detach().cpu().numpy()
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

        if save_vis:
            bg_hd_imgs = np.transpose(hd_imgs, [0, 3, 1, 2])
            out_img['hd_imgs'] = bg_hd_imgs
        if render:
            # Render the initial predictions on the original image resolution
            hd_orig_overlays = hd_renderer(
                model_vertices, faces,
                focal_length=hd_params['focal_length_in_px'],
                camera_translation=hd_params['transl'],
                camera_center=hd_params['center'],
                bg_imgs=bg_hd_imgs,
                return_with_alpha=True,
            )
            out_img['hd_orig_overlay'] = hd_orig_overlays

        # Render the overlays of the final prediction
        if render:
            # bbox保持
            cbbox = body_targets[0].bbox.detach().cpu().numpy()
            bbox_size = np.array(body_targets[0].size)
            dset_center = np.array(body_targets[0].extra_fields['center'])
            dset_size = np.array(body_targets[0].extra_fields['bbox_size'])
            # 画面サイズに合わせる(描画のため、int)
            bbox = np.tile(dset_center, 2) + ((cbbox / np.tile(bbox_size, 2) - np.tile(0.5, 4)) * np.tile(dset_size, 4))
            img_bbox = bbox.astype(np.int)

            hd_params['img_bbox'] = bbox

            hd_overlays = hd_renderer(
                final_model_vertices,
                faces,
                focal_length=hd_params['focal_length_in_px'],
                camera_translation=hd_params['transl'],
                camera_center=hd_params['center'],
                bg_imgs=bg_hd_imgs,
                return_with_alpha=True,
                body_color=[0.4, 0.4, 0.7]
            )
            
            proj_joints = stage_n_out['proj_joints'][0].detach().cpu().numpy()
            hd_params['proj_joints'] = proj_joints

            try:
                # 横線
                for x in range(img_bbox[0], img_bbox[2] + 1):
                    for y in [img_bbox[1], img_bbox[3] + 1]:
                        if hd_overlays.shape[2] > x and hd_overlays[3] > y:
                            hd_overlays[:, :, y, x] = np.array([1, 0, 0, 1])
                
                # 縦線
                for x in [img_bbox[0], img_bbox[2] + 1]:
                    for y in range(img_bbox[1], img_bbox[3] + 1):
                        if hd_overlays.shape[2] > x and hd_overlays[3] > y:
                            hd_overlays[:, :, y, x] = np.array([1, 0, 0, 1])
                
                # カメラ中央
                for x in range(int(hd_params['center'][0, 0] - 1), int(hd_params['center'][0, 0] + 2)):
                    for y in range(int(hd_params['center'][0, 1] - 1), int(hd_params['center'][0, 1] + 2)):
                        if hd_overlays.shape[2] > x and hd_overlays[3] > y:
                            hd_overlays[:, :, y, x] = np.array([0, 1, 0, 1])
            
                min_joints = np.min(proj_joints, axis=0)
                max_joints = np.max(proj_joints, axis=0)
                diff_joints = max_joints - min_joints
                diff_bbox = np.array([hd_params['img_bbox'][2] - hd_params['img_bbox'][0], hd_params['img_bbox'][3] - hd_params['img_bbox'][1]])
                jscale = diff_joints / diff_bbox
                jscale = np.mean([jscale[0], jscale[1]])
                for jidx, jname in enumerate(KEYPOINT_NAMES):
                    j2d = proj_joints[jidx] / jscale

                    # ジョイント
                    for x in range(int(hd_params['center'][0, 0] + j2d[0] - 1), int(hd_params['center'][0, 0] + j2d[0] + 2)):
                        for y in range(int(hd_params['center'][0, 1] + j2d[1] - 1), int(hd_params['center'][0, 1] + j2d[1] + 2)):
                            if hd_overlays.shape[2] > x and hd_overlays[3] > y:
                                hd_overlays[:, :, y, x] = np.array([0, 0, 1, 1])
            
            except Exception as e:
                print('hd_overlays error: %s' % e)
                pass
            
            out_img['hd_overlay'] = hd_overlays

        for deg in degrees:
            hd_overlays = hd_renderer(
                final_model_vertices, faces,
                focal_length=hd_params['focal_length_in_px'],
                camera_translation=hd_params['transl'],
                camera_center=hd_params['center'],
                bg_imgs=bg_hd_imgs,
                return_with_alpha=True,
                render_bg=False,
                body_color=[0.4, 0.4, 0.7],
                deg=deg,
            )
            out_img[f'hd_rendering_{deg:03.0f}'] = hd_overlays

        if save_vis:
            for key in out_img.keys():
                out_img[key] = np.clip(
                    np.transpose(
                        out_img[key], [0, 2, 3, 1]) * 255, 0, 255).astype(
                            np.uint8)

        for idx in tqdm(range(len(body_targets)), 'Saving ...'):

            # TODO 複数人対応
            if idx > 0:
                break

            fname = body_targets[idx].get_field('fname')
            curr_out_path = osp.join(demo_output_folder, fname)
            os.makedirs(curr_out_path, exist_ok=True)

            if save_vis:
                for name, curr_img in out_img.items():
                    pil_img.fromarray(curr_img[idx]).save(
                        osp.join(curr_out_path, f'{name}.png'))

            if save_mesh:
                # Store the mesh predicted by the body-crop network
                naive_mesh = o3d.geometry.TriangleMesh()
                naive_mesh.vertices = Vec3d(
                    model_vertices[idx] + hd_params['transl'][idx])
                naive_mesh.triangles = Vec3i(faces)
                mesh_fname = osp.join(curr_out_path, f'body_{fname}.ply')
                o3d.io.write_triangle_mesh(mesh_fname, naive_mesh)

                # Store the final mesh
                expose_mesh = o3d.geometry.TriangleMesh()
                expose_mesh.vertices = Vec3d(
                    final_model_vertices[idx] + hd_params['transl'][idx])
                expose_mesh.triangles = Vec3i(faces)
                mesh_fname = osp.join(curr_out_path, f'{fname}.ply')
                o3d.io.write_triangle_mesh(mesh_fname, expose_mesh)

            if save_params:
                params_fname = osp.join(curr_out_path, f'{fname}_params.npz')
                out_params = dict(fname=fname)
                for key, val in stage_n_out.items():
                    if torch.is_tensor(val):
                        val = val.detach().cpu().numpy()[idx]
                    out_params[key] = val
                for key, val in hd_params.items():
                    if torch.is_tensor(val):
                        val = val.detach().cpu().numpy()
                    if np.isscalar(val[idx]):
                        out_params[key] = val[idx].item()
                    else:
                        out_params[key] = val[idx]

                try:
                    for param_name in ['center']:
                        params_txt_fname = osp.join(curr_out_path, f'{fname}_params_{param_name}.txt')
                        np.savetxt(params_txt_fname, out_params[param_name])

                    for param_name in ['img_bbox']:
                        params_txt_fname = osp.join(curr_out_path, f'{fname}_params_{param_name}.txt')
                        np.savetxt(params_txt_fname, hd_params[param_name])

                    for param_name in ['joints']:
                        params_txt_fname = osp.join(curr_out_path, f'{fname}_params_{param_name}.json')
                        
                        # json出力
                        joint_dict = {}
                        joint_dict["image"] = {"width": W, "height": H}
                        joint_dict["depth"] = {"depth": float(hd_params["depth"][0][0])}
                        joint_dict["center"] = {"x": float(hd_params['center'][0, 0]), "y": float(hd_params['center'][0, 1])}
                        joint_dict["bbox"] = {"x": float(hd_params["img_bbox"][0]), "y": float(hd_params["img_bbox"][1]), "width": float(hd_params["img_bbox"][2]), "height": float(hd_params["img_bbox"][3])}
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

                        with open(params_txt_fname, 'w') as f:
                            json.dump(joint_dict, f, indent=4)

                        # 描画設定
                        fig = plt.figure(figsize=(15,15),dpi=100)
                        # 3DAxesを追加
                        ax = fig.add_subplot(111, projection='3d')

                        # ジョイント出力                    
                        ax.set_xlim3d(int(-(original_width / 2)), int(original_width / 2))
                        ax.set_ylim3d(0, int(original_height / 2))
                        ax.set_zlim3d(0, int(original_height))
                        ax.set(xlabel='x', ylabel='y', zlabel='z')

                        xs = []
                        ys = []
                        zs = []

                        for j3d_from_idx, j3d_to_idx in ALL_CONNECTIONS:
                            jfname = KEYPOINT_NAMES[j3d_from_idx]
                            jtname = KEYPOINT_NAMES[j3d_to_idx]

                            xs = [joint_dict[jfname]['x'], joint_dict[jtname]['x']]
                            ys = [joint_dict[jfname]['y'], joint_dict[jtname]['y']]
                            zs = [joint_dict[jfname]['z'], joint_dict[jtname]['z']]

                            ax.plot3D(xs, ys, zs, marker="o", ms=2, c="#0000FF")
                        
                        plt.savefig(os.path.join(curr_out_path, f'{fname}_{param_name}.png'))
                        plt.close()

                        # posの出力
                        joint_names = [(0, 'pelvis'), (1, 'right_hip'), (2, 'right_knee'), (3, 'right_ankle'), \
                                    (6, 'left_hip'), (7, 'left_knee'), (8, 'left_ankle'), \
                                    (12, 'spine1'), (13, 'spine2'), (14, 'neck'), (15, 'head'), \
                                    (17, 'left_shoulder'), (18, 'left_elbow'), (19, 'left_wrist'), \
                                    (25, 'right_shoulder'), (26, 'right_elbow'), (27, 'right_wrist')
                                    ]

                        N = []
                        I = []

                        for (jnidx, iname) in joint_names:
                            for jidx, smplx_jn in enumerate(KEYPOINT_NAMES):
                                if smplx_jn == iname:
                                    N.append(jnidx)
                                    I.append([joint_dict[iname]['x'], joint_dict[iname]['y'], joint_dict[iname]['z']])

                        for i in np.arange( len(I) ):
                            # 0: index, 1: x軸, 2:Y軸, 3:Z軸
                            posf.write(str(N[i]) + " "+ str(I[i][0]) +" "+ str(I[i][2]) +" "+ str(I[i][1]) + ", ")

                        #終わったら改行
                        posf.write("\n")

                except Exception as e:
                    print('savetxt error: %s' % e)
                    pass

                np.savez_compressed(params_fname, **out_params)

            if show:
                nrows = 1
                ncols = 4 + len(degrees)
                fig, axes = plt.subplots(
                    ncols=ncols, nrows=nrows, num=0,
                    gridspec_kw={'wspace': 0, 'hspace': 0})
                axes = axes.reshape(nrows, ncols)
                for ax in axes.flatten():
                    ax.clear()
                    ax.set_axis_off()

                axes[0, 0].imshow(hd_imgs[idx])
                axes[0, 1].imshow(out_img['rgb'][idx])
                axes[0, 2].imshow(out_img['hd_orig_overlay'][idx])
                axes[0, 3].imshow(out_img['hd_overlay'][idx])
                start = 4
                for deg in degrees:
                    axes[0, start].imshow(
                        out_img[f'hd_rendering_{deg:03.0f}'][idx])
                    start += 1

                plt.draw()
                if pause > 0:
                    plt.pause(pause)
                else:
                    plt.show()

    fmt = cv2.VideoWriter_fourcc(*'mp4v')
    
    # メッシュの結合
    writer = cv2.VideoWriter(osp.join(demo_output_folder, 'hd_overlay.mp4'), fmt, 30, (original_width, original_height))
    for img_path in glob.glob(osp.join(demo_output_folder, "**/hd_overlay.png")):
        writer.write(cv2.imread(img_path))
    writer.release()

    # JOINTの結合
    writer = cv2.VideoWriter(osp.join(demo_output_folder, 'joints.mp4'), fmt, 30, (1500, 1500))
    for img_path in glob.glob(osp.join(demo_output_folder, "**/*_joints.png")):
        writer.write(cv2.imread(img_path))
    writer.release()

    cv2.destroyAllWindows()
    posf.close()

    logger.info(f'Average inference time: {total_time / cnt}')


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    arg_formatter = argparse.ArgumentDefaultsHelpFormatter
    description = 'PyTorch SMPL-X Regressor Demo'
    parser = argparse.ArgumentParser(formatter_class=arg_formatter,
                                     description=description)

    parser.add_argument('--video-path', type=str, dest='video_path',
                        help='The folder with images that will be processed')
    parser.add_argument('--exp-cfg', type=str, dest='exp_cfg',
                        help='The configuration of the experiment')
    parser.add_argument('--output-folder', dest='output_folder',
                        default='demo_output', type=str,
                        help='The folder where the demo renderings will be' +
                        ' saved')
    parser.add_argument('--exp-opts', default=[], dest='exp_opts',
                        nargs='*', help='Extra command line arguments')
    parser.add_argument('--datasets', nargs='+',
                        default=['openpose'], type=str,
                        help='Datasets to process')
    parser.add_argument('--show', default=False,
                        type=lambda arg: arg.lower() in ['true'],
                        help='Display the results')
    parser.add_argument('--expose-batch',
                        dest='expose_batch',
                        default=1, type=int,
                        help='ExPose batch size')
    parser.add_argument('--rcnn-batch',
                        dest='rcnn_batch',
                        default=1, type=int,
                        help='R-CNN batch size')
    parser.add_argument('--pause', default=-1, type=float,
                        help='How much to pause the display')
    parser.add_argument('--focal-length', dest='focal_length', type=float,
                        default=5000,
                        help='Focal length')
    parser.add_argument('--degrees', type=float, nargs='*', default=[],
                        help='Degrees of rotation around the vertical axis')
    parser.add_argument('--save-vis', dest='save_vis', default=False,
                        type=lambda x: x.lower() in ['true'],
                        help='Whether to save visualizations')
    parser.add_argument('--save-mesh', dest='save_mesh', default=False,
                        type=lambda x: x.lower() in ['true'],
                        help='Whether to save meshes')
    parser.add_argument('--save-params', dest='save_params', default=False,
                        type=lambda x: x.lower() in ['true'],
                        help='Whether to save parameters')

    cmd_args = parser.parse_args()

    video_path = cmd_args.video_path
    show = cmd_args.show
    output_folder = cmd_args.output_folder
    pause = cmd_args.pause
    focal_length = cmd_args.focal_length
    save_vis = cmd_args.save_vis
    save_params = cmd_args.save_params
    save_mesh = cmd_args.save_mesh
    degrees = cmd_args.degrees
    expose_batch = cmd_args.expose_batch
    rcnn_batch = cmd_args.rcnn_batch

    cfg.merge_from_file(cmd_args.exp_cfg)
    cfg.merge_from_list(cmd_args.exp_opts)

    cfg.datasets.body.batch_size = expose_batch

    cfg.is_training = False
    cfg.datasets.body.splits.test = cmd_args.datasets
    use_face_contour = cfg.datasets.use_face_contour
    set_face_contour(cfg, use_face_contour=use_face_contour)

    with threadpool_limits(limits=1):
        main(
            video_path,
            cfg,
            show=show,
            demo_output_folder=output_folder,
            pause=pause,
            focal_length=focal_length,
            save_vis=save_vis,
            save_mesh=save_mesh,
            save_params=save_params,
            degrees=degrees,
            rcnn_batch=rcnn_batch,
        )
