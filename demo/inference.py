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
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import matplotlib.pyplot as plt
import PIL.Image as pil_img
from threadpoolctl import threadpool_limits
from tqdm import tqdm

import open3d as o3d

import time
import argparse
from collections import defaultdict
from loguru import logger
from collections import OrderedDict
import numpy as np

import torch

import resource

from expose.utils.plot_utils import HDRenderer
from expose.config.cmd_parser import set_face_contour
from expose.config import cfg
from expose.models.smplx_net import SMPLXNet
from expose.data import make_all_data_loaders
from expose.utils.checkpointer import Checkpointer
from expose.data.targets.image_list import to_image_list


rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))


Vec3d = o3d.utility.Vector3dVector
Vec3i = o3d.utility.Vector3iVector


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
    exp_cfg,
    show=False,
    demo_output_folder='demo_output',
    pause=-1,
    focal_length=5000, sensor_width=36,
    save_vis=True,
    save_params=False,
    save_mesh=False,
    degrees=[],
):

    device = torch.device('cuda')
    if not torch.cuda.is_available():
        logger.error('CUDA is not available!')
        sys.exit(3)

    logger.remove()
    logger.add(lambda x: tqdm.write(x, end=''),
               level=exp_cfg.logger_level.upper(),
               colorize=True)

    demo_output_folder = osp.expanduser(osp.expandvars(demo_output_folder))
    logger.info(f'Saving results to: {demo_output_folder}')
    os.makedirs(demo_output_folder, exist_ok=True)

    model = SMPLXNet(exp_cfg)
    try:
        model = model.to(device=device)
    except RuntimeError:
        # Re-submit in case of a device error
        sys.exit(3)

    checkpoint_folder = osp.join(
        exp_cfg.output_folder, exp_cfg.checkpoint_folder)
    checkpointer = Checkpointer(model, save_dir=checkpoint_folder,
                                pretrained=exp_cfg.pretrained)

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

    dataloaders = make_all_data_loaders(exp_cfg, split='test')

    body_dloader = dataloaders['body'][0]

    total_time = 0
    cnt = 0
    for bidx, batch in enumerate(tqdm(body_dloader, dynamic_ncols=True)):

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

    logger.info(f'Average inference time: {total_time / cnt}')


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    arg_formatter = argparse.ArgumentDefaultsHelpFormatter
    description = 'PyTorch SMPL-X Regressor Demo'
    parser = argparse.ArgumentParser(formatter_class=arg_formatter,
                                     description=description)

    parser.add_argument('--exp-cfg', type=str, dest='exp_cfg',
                        help='The configuration of the experiment')
    parser.add_argument('--output-folder', dest='output_folder',
                        default='demo_output', type=str,
                        help='The folder where the demo renderings will be' +
                        ' saved')
    parser.add_argument('--datasets', nargs='+',
                        default=['openpose'], type=str,
                        help='Datasets to process')
    parser.add_argument('--show', default=False,
                        type=lambda arg: arg.lower() in ['true'],
                        help='Display the results')
    parser.add_argument('--pause', default=-1, type=float,
                        help='How much to pause the display')
    parser.add_argument('--exp-opts', default=[], dest='exp_opts',
                        nargs='*', help='Extra command line arguments')
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

    show = cmd_args.show
    output_folder = cmd_args.output_folder
    pause = cmd_args.pause
    focal_length = cmd_args.focal_length
    save_vis = cmd_args.save_vis
    save_params = cmd_args.save_params
    save_mesh = cmd_args.save_mesh
    degrees = cmd_args.degrees

    cfg.merge_from_file(cmd_args.exp_cfg)
    cfg.merge_from_list(cmd_args.exp_opts)

    cfg.is_training = False
    cfg.datasets.body.splits.test = cmd_args.datasets
    use_face_contour = cfg.datasets.use_face_contour
    set_face_contour(cfg, use_face_contour=use_face_contour)

    with threadpool_limits(limits=1):
        main(cfg, show=show, demo_output_folder=output_folder, pause=pause,
             focal_length=focal_length,
             save_vis=save_vis,
             save_mesh=save_mesh,
             save_params=save_params,
             degrees=degrees,
             )
