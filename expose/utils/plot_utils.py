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

from typing import Union, NewType, List

import sys
import numpy as np
import torch

import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors
import trimesh
import pyrender

from loguru import logger
import cv2


Tensor = NewType('Tensor', torch.Tensor)
Array = NewType('Array', np.ndarray)


COLORS = {
    'N': [1.0, 1.0, 0.9],
    'GT': [146 / 255.0, 189 / 255.0, 163 / 255.0]
}
HAND_COLORS = np.array([[0.4, 0.4, 0.4],
                        [0.4, 0.4, 0.],
                        [0.6, 0.6, 0.],
                        [0.8, 0.8, 0.],
                        [0., 0.4, 0.2],
                        [0., 0.6, 0.3],
                        [0., 0.8, 0.4],
                        [0., 0.2, 0.4],
                        [0., 0.3, 0.6],
                        [0., 0.4, 0.8],
                        [0.4, 0., 0.4],
                        [0.6, 0., 0.6],
                        [0.7, 0., 0.8],
                        [0.4, 0., 0.],
                        [0.6, 0., 0.],
                        [0.8, 0., 0.],
                        [1., 0., 0.],
                        [1., 1., 0.],
                        [0., 1., 0.5],
                        [1., 0., 1.],
                        [0., 0.5, 1.]])
HAND_COLORS = HAND_COLORS[:, ::-1]
FINGER_NAMES = [
    f'{side}_{finger_name}'
    for side in ['left', 'right']
    for finger_name in ['thumb', 'index', 'middle', 'ring', 'pinky']]

RIGHT_FINGER = [
    'right_wrist',
    'right_index1',
    'right_index2',
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3',
    'right_thumb',
    'right_index',
    'right_middle',
    'right_ring',
    'right_pinky']
LEFT_FINGER = [name.replace('right', 'left') for name in RIGHT_FINGER]


def blend_images(img1, img2, alpha=0.7):
    return img1 * alpha + (1 - alpha) * img2


def target_to_part_mask_img(target, num_parts=14, cmap_name='tab20'):
    cmap = mpl_cm.get_cmap(name='tab20')
    norm = mpl_colors.Normalize(0, num_parts + 1)

    full_mask = np.full(tuple(target.size), num_parts + 1,
                        dtype=np.float32)

    for part_idx in range(num_parts):
        if not target.has_field(f'part_mask{part_idx}'):
            continue

        masks = target.get_field(f'part_mask{part_idx}')
        masks = masks.get_mask_tensor()
        masks = masks.detach().cpu().numpy().astype(np.float32)

        full_mask[masks > 0] = part_idx
        #  color = np.asarray(cmap(norm(part_idx)))[:3].reshape(1, 1, 3)
        #  if colored_mask is None:
        #  colored_mask = np.zeros(masks.shape + (3,), dtype=masks.dtype)
        #  colored_mask += masks[:, :, np.newaxis] * color
    colored_mask = cmap(norm(full_mask))[:, :, :3]
    colored_mask = np.clip(colored_mask, 0.0, 1.0)

    return colored_mask


def create_skel_img(img, keypoints, connections, valid=None,
                    names=None,
                    color_left=[0.9, 0.0, 0.0],
                    color_right=[0.0, 0.0, 0.9],
                    color_else=[1.0, 1.0, 1.0],
                    marker_size=2, linewidth=2, draw_skel=True,
                    draw_text=True,
                    ):
    kp_mask = np.copy(img)
    if valid is None:
        valid = np.ones([keypoints.shape[0]])

    for idx, pair in enumerate(connections):
        if pair[0] > len(valid) or pair[1] > len(valid):
            continue
        if not valid[pair[0]] or not valid[pair[1]]:
            continue

        curr_line_width = linewidth
        if pair[1] >= 22:
            curr_marker_size = int(0.1 * marker_size)
            #  curr_line_width = 1
        else:
            curr_marker_size = marker_size

        if names is not None:
            curr_name = names[pair[1]]

            if any([finger_name in curr_name for finger_name in FINGER_NAMES]):
                if 'left' in curr_name:
                    color = HAND_COLORS[LEFT_FINGER.index(curr_name)]
                else:
                    color = HAND_COLORS[RIGHT_FINGER.index(curr_name)]
            elif 'left' in curr_name:
                color = color_left
            elif 'right' in curr_name:
                color = color_right
            else:
                color = color_else
        else:
            color = color_else

        if pair[1] >= keypoints.shape[0] or pair[0] >= keypoints.shape[0]:
            continue
        center = tuple(keypoints[pair[1], :].astype(np.int32).tolist())

        cv2.circle(kp_mask, center, curr_marker_size, color)

        if draw_skel:
            if not valid[pair[0]] and not valid[pair[1]]:
                continue
            start_pt = tuple(keypoints[pair[0], :2].astype(np.int32).tolist())
            end_pt = tuple(keypoints[pair[1], :2].astype(np.int32).tolist())
            cv2.line(kp_mask, start_pt, end_pt,
                     color, thickness=curr_line_width,
                     lineType=cv2.LINE_AA)

        if pair[1] <= 22 and draw_text:
            cv2.putText(kp_mask, f'{pair[1]}',
                        center, cv2.FONT_HERSHEY_PLAIN, fontScale=1.0,
                        color=[0.0, 0.0, 0.0], thickness=4)
            cv2.putText(kp_mask, f'{pair[1]}',
                        center, cv2.FONT_HERSHEY_PLAIN, fontScale=1.0,
                        color=color, thickness=2)

    return kp_mask


def create_bbox_img(img, bounding_box, color=(0.0, 0.0, 0.0),
                    linewidth=2):
    bbox_img = img.copy()
    xmin, ymin, xmax, ymax = bounding_box.reshape(4)

    cv2.rectangle(bbox_img, (xmin, ymin), (xmax, ymax),
                  color, thickness=linewidth)
    return bbox_img


def create_dp_img(img, dp_points, cmap='viridis', marker_size=4):
    ''' Creates a Dense Pose visualization
    '''
    dp_img = np.copy(img)

    cm = mpl_cm.get_cmap(name=cmap)

    num_points = dp_points.shape[0]
    colors = cm(np.linspace(0, 1, num_points))[:, :3]
    for idx in range(num_points):
        center = tuple(dp_points[idx, :].astype(np.int32).tolist())
        cv2.circle(dp_img, center, marker_size,
                   colors[idx], -1)

    return dp_img


class OpenCVCamera(pyrender.Camera):
    PIXEL_CENTER_OFFSET = 0.5

    def __init__(self,
                 focal_length=1000,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(OpenCVCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.focal_length = focal_length

    def get_projection_matrix(self, width=None, height=None):
        cx = 0.5 * width
        cy = 0.5 * height

        right = (width - (cx + self.PIXEL_CENTER_OFFSET)) * (
            self.znear / self.focal_length)
        left = -(cx + self.PIXEL_CENTER_OFFSET) * (self.znear /
                                                   self.focal_length)
        top = -(height - (cy + self.PIXEL_CENTER_OFFSET)) * (
            self.znear / self.focal_length)
        bottom = (cy + self.PIXEL_CENTER_OFFSET) * (
            self.znear / self.focal_length)

        P = np.zeros([4, 4])

        P[0][0] = 2 * self.znear / (right - left)
        P[1, 1] = -2 * self.znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[2, 2] = - (self.zfar + self.znear) / (self.zfar - self.znear)
        P[3, 2] = -1.0
        P[2][3] = (2 * self.zfar * self.znear) / (self.znear - self.zfar)

        return P


class Renderer(object):
    def __init__(self, near=0.1, far=200, width=224, height=224,
                 bg_color=(0.0, 0.0, 0.0, 0.0), ambient_light=None,
                 use_raymond_lighting=True,
                 light_color=None, light_intensity=3.0):
        if light_color is None:
            light_color = np.ones(3)

        self.near = near
        self.far = far

        self.renderer = pyrender.OffscreenRenderer(viewport_width=width,
                                                   viewport_height=height,
                                                   point_size=1.0)

        if ambient_light is None:
            ambient_light = (0.1, 0.1, 0.1)

        self.scene = pyrender.Scene(bg_color=bg_color,
                                    ambient_light=ambient_light)

        pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.0,
                                        aspectRatio=float(width) / height)
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = np.array([0, 0, 2])
        self.scene.add(pc, pose=camera_pose)

        if use_raymond_lighting:
            light_nodes = self._create_raymond_lights()
            for node in light_nodes:
                self.scene.add_node(node)

    def _create_raymond_lights(self):
        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        nodes = []

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3, :3] = np.c_[x, y, z]
            nodes.append(
                pyrender.Node(
                    light=pyrender.DirectionalLight(color=np.ones(3),
                                                    intensity=1.0),
                    matrix=matrix
                ))

        return nodes

    def __call__(self, vertices, faces, img=None,
                 img_size=224,
                 body_color=(1.0, 1.0, 1.0, 1.0),
                 **kwargs):

        centered_verts = vertices - np.mean(vertices, axis=0, keepdims=True)
        meshes = self.create_mesh(centered_verts, faces,
                                  vertex_color=body_color)

        for node in self.scene.get_nodes():
            if node.name == 'mesh':
                self.scene.remove_node(node)
        for mesh in meshes:
            self.scene.add(mesh, name='mesh')

        color, _ = self.renderer.render(self.scene)

        return color.astype(np.uint8)

    def create_mesh(self, vertices, faces,
                    vertex_color=(0.9, 0.9, 0.7, 1.0)):

        tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        rot = trimesh.transformations.rotation_matrix(np.radians(180),
                                                      [1, 0, 0])
        tri_mesh.apply_transform(rot)

        meshes = []

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            baseColorFactor=vertex_color)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=material)
        meshes.append(mesh)
        return meshes


class WeakPerspectiveCamera(pyrender.Camera):
    PIXEL_CENTER_OFFSET = 0.5

    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=pyrender.camera.DEFAULT_Z_FAR,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale
        P[1, 1] = self.scale
        P[0, 3] = self.translation[0] * self.scale
        P[1, 3] = -self.translation[1] * self.scale
        P[2, 2] = -1

        return P


class WeakPerspectiveCameraNonSquare(pyrender.Camera):
    PIXEL_CENTER_OFFSET = 0.5

    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=pyrender.camera.DEFAULT_Z_FAR,
                 name=None):
        super(WeakPerspectiveCameraNonSquare, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1

        return P


class AbstractRenderer(object):
    def __init__(self, faces=None, img_size=224, use_raymond_lighting=True):
        super(AbstractRenderer, self).__init__()

        self.img_size = img_size
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=img_size,
            viewport_height=img_size,
            point_size=1.0)
        self.mat_constructor = pyrender.MetallicRoughnessMaterial
        self.mesh_constructor = trimesh.Trimesh
        self.trimesh_to_pymesh = pyrender.Mesh.from_trimesh
        self.transf = trimesh.transformations.rotation_matrix

        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                    ambient_light=(0.0, 0.0, 0.0))
        if use_raymond_lighting:
            light_nodes = self._create_raymond_lights()
            for node in light_nodes:
                self.scene.add_node(node)

    def _create_raymond_lights(self):
        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        nodes = []

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3, :3] = np.c_[x, y, z]
            nodes.append(
                pyrender.Node(
                    light=pyrender.DirectionalLight(color=np.ones(3),
                                                    intensity=1.0),
                    matrix=matrix
                ))

        return nodes

    def is_active(self):
        return self.viewer.is_active

    def close_viewer(self):
        if self.viewer.is_active:
            self.viewer.close_external()

    def create_mesh(self, vertices, faces, color=(0.3, 0.3, 0.3, 1.0),
                    wireframe=False, deg=0):

        material = self.mat_constructor(
            metallicFactor=0.0,
            alphaMode='BLEND',
            baseColorFactor=color)

        mesh = self.mesh_constructor(vertices, faces, process=False)

        curr_vertices = vertices.copy()
        mesh = self.mesh_constructor(
            curr_vertices, faces, process=False)
        if deg != 0:
            rot = self.transf(
                np.radians(deg), [0, 1, 0],
                point=np.mean(curr_vertices, axis=0))
            mesh.apply_transform(rot)

        rot = self.transf(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)

        return self.trimesh_to_pymesh(mesh, material=material)

    def update_mesh(self, vertices, faces, body_color=(1.0, 1.0, 1.0, 1.0),
                    deg=0):
        for node in self.scene.get_nodes():
            if node.name == 'body_mesh':
                self.scene.remove_node(node)
                break

        body_mesh = self.create_mesh(
            vertices, faces, color=body_color, deg=deg)
        self.scene.add(body_mesh, name='body_mesh')


class SMPLifyXRenderer(AbstractRenderer):
    def __init__(self, faces=None, img_size=224):
        super(SMPLifyXRenderer, self).__init__(faces=faces, img_size=img_size)

    def update_camera(self, translation, rotation=None, focal_length=5000,
                      camera_center=None):
        for node in self.scene.get_nodes():
            if node.name == 'camera':
                self.scene.remove_node(node)
        if rotation is None:
            rotation = np.eye(3, dtype=translation.dtype)
        if camera_center is None:
            camera_center = np.array(
                [self.img_size, self.img_size], dtype=translation.dtype) * 0.5

        camera_transl = translation.copy()
        camera_transl[0] *= -1.0
        pc = pyrender.camera.IntrinsicsCamera(
            fx=focal_length, fy=focal_length,
            cx=camera_center[0], cy=camera_center[1])
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = rotation
        camera_pose[:3, 3] = camera_transl
        self.scene.add(pc, pose=camera_pose, name='camera')

    @torch.no_grad()
    def __call__(self, vertices, faces,
                 camera_translation, bg_imgs=None,
                 body_color=(1.0, 1.0, 1.0),
                 upd_color=None,
                 **kwargs):
        if upd_color is None:
            upd_color = {}

        if torch.is_tensor(vertices):
            vertices = vertices.detach().cpu().numpy()
        if torch.is_tensor(camera_translation):
            camera_translation = camera_translation.cpu().numpy()
        batch_size = vertices.shape[0]

        output_imgs = []
        for bidx in range(batch_size):
            self.update_camera(camera_translation[bidx])

            curr_col = upd_color.get(bidx, None)
            if curr_col is None:
                curr_col = body_color
            self.update_mesh(vertices[bidx], faces, body_color=curr_col)

            flags = (pyrender.RenderFlags.RGBA |
                     pyrender.RenderFlags.SKIP_CULL_FACES)
            color, depth = self.renderer.render(self.scene, flags=flags)

            color = np.transpose(color, [2, 0, 1]).astype(np.float32) / 255.0
            color = np.clip(color, 0, 1)

            if bg_imgs is None:
                output_imgs.append(color[:-1])
            else:
                valid_mask = (color[3] > 0)[np.newaxis]

                output_img = (color[:-1] * valid_mask +
                              (1 - valid_mask) * bg_imgs[bidx])
                output_imgs.append(np.clip(output_img, 0, 1))
        return np.stack(output_imgs, axis=0)


class OverlayRenderer(AbstractRenderer):
    def __init__(self, faces=None, img_size=224, tex_size=1):
        super(OverlayRenderer, self).__init__(faces=faces, img_size=img_size)

    def update_camera(self, scale, translation):
        for node in self.scene.get_nodes():
            if node.name == 'camera':
                self.scene.remove_node(node)

        pc = WeakPerspectiveCamera(scale, translation,
                                   znear=1e-5,
                                   zfar=1000)
        camera_pose = np.eye(4)
        self.scene.add(pc, pose=camera_pose, name='camera')

    @torch.no_grad()
    def __call__(self, vertices, faces,
                 camera_scale, camera_translation, bg_imgs=None,
                 deg=0,
                 return_with_alpha=False,
                 body_color=None,
                 **kwargs):

        if torch.is_tensor(vertices):
            vertices = vertices.detach().cpu().numpy()
        if torch.is_tensor(camera_scale):
            camera_scale = camera_scale.detach().cpu().numpy()
        if torch.is_tensor(camera_translation):
            camera_translation = camera_translation.detach().cpu().numpy()
        batch_size = vertices.shape[0]

        output_imgs = []
        for bidx in range(batch_size):
            if body_color is None:
                body_color = COLORS['N']

            if bg_imgs is not None:
                _, H, W = bg_imgs[bidx].shape
                # Update the renderer's viewport
                self.renderer.viewport_height = H
                self.renderer.viewport_width = W

            self.update_camera(camera_scale[bidx], camera_translation[bidx])
            self.update_mesh(vertices[bidx], faces, body_color=body_color,
                             deg=deg)

            flags = (pyrender.RenderFlags.RGBA |
                     pyrender.RenderFlags.SKIP_CULL_FACES)
            color, depth = self.renderer.render(self.scene, flags=flags)
            color = np.transpose(color, [2, 0, 1]).astype(np.float32) / 255.0
            color = np.clip(color, 0, 1)

            if bg_imgs is None:
                if return_with_alpha:
                    output_imgs.append(color)
                else:
                    output_imgs.append(color[:-1])
            else:
                if return_with_alpha:
                    valid_mask = (color[3] > 0)[np.newaxis]

                    if bg_imgs[bidx].shape[0] < 4:
                        curr_bg_img = np.concatenate(
                            [bg_imgs[bidx],
                             np.ones_like(bg_imgs[bidx, [0], :, :])
                             ], axis=0)
                    else:
                        curr_bg_img = bg_imgs[bidx]

                    output_img = (color * valid_mask +
                                  (1 - valid_mask) * curr_bg_img)
                    output_imgs.append(np.clip(output_img, 0, 1))
                else:
                    valid_mask = (color[3] > 0)[np.newaxis]

                    output_img = (color[:-1] * valid_mask +
                                  (1 - valid_mask) * bg_imgs[bidx])
                    output_imgs.append(np.clip(output_img, 0, 1))
        return np.stack(output_imgs, axis=0)


class GTRenderer(AbstractRenderer):
    def __init__(self, faces=None, img_size=224):
        super(GTRenderer, self).__init__(faces=faces, img_size=img_size)

    def update_camera(self, intrinsics):
        for node in self.scene.get_nodes():
            if node.name == 'camera':
                self.scene.remove_node(node)
        pc = pyrender.IntrinsicsCamera(
            fx=intrinsics[0, 0],
            fy=intrinsics[1, 1],
            cx=intrinsics[0, 2],
            cy=intrinsics[1, 2],
            zfar=1000)
        camera_pose = np.eye(4)
        self.scene.add(pc, pose=camera_pose, name='camera')

    @torch.no_grad()
    def __call__(self, vertices, faces,
                 intrinsics, bg_imgs=None, deg=0,
                 return_with_alpha=False,
                 **kwargs):
        ''' Returns a B3xHxW batch of mesh overlays
        '''

        if torch.is_tensor(vertices):
            vertices = vertices.detach().cpu().numpy()
        if torch.is_tensor(intrinsics):
            intrinsics = intrinsics.detach().cpu().numpy()
        batch_size = vertices.shape[0]

        body_color = COLORS['GT']
        output_imgs = []
        for bidx in range(batch_size):
            if bg_imgs is not None:
                _, H, W = bg_imgs[bidx].shape
                # Update the renderer's viewport
                self.renderer.viewport_height = H
                self.renderer.viewport_width = W
            self.update_camera(intrinsics[bidx])
            self.update_mesh(vertices[bidx], faces, body_color=body_color,
                             deg=deg)

            flags = (pyrender.RenderFlags.RGBA |
                     pyrender.RenderFlags.SKIP_CULL_FACES)
            color, depth = self.renderer.render(self.scene, flags=flags)
            color = np.transpose(color, [2, 0, 1]).astype(np.float32) / 255.0
            color = np.clip(color, 0, 1)

            if bg_imgs is None:
                if return_with_alpha:
                    output_imgs.append(color)
                else:
                    output_imgs.append(color[:-1])
            else:
                if return_with_alpha:
                    valid_mask = (color[3] > 0)[np.newaxis]

                    if bg_imgs[bidx].shape[0] < 4:
                        curr_bg_img = np.concatenate(
                            [bg_imgs[bidx],
                             np.ones_like(bg_imgs[bidx, [0], :, :])
                             ], axis=0)
                    else:
                        curr_bg_img = bg_imgs[bidx]

                    output_img = (color * valid_mask +
                                  (1 - valid_mask) * curr_bg_img)
                    output_imgs.append(np.clip(output_img, 0, 1))
                else:
                    valid_mask = (color[3] > 0)[np.newaxis]

                    output_img = (color[:-1] * valid_mask +
                                  (1 - valid_mask) * bg_imgs[bidx])
                    output_imgs.append(np.clip(output_img, 0, 1))
        return np.stack(output_imgs, axis=0)


class HDRenderer(OverlayRenderer):
    def __init__(self, **kwargs):
        super(HDRenderer, self).__init__(**kwargs)

    def update_camera(self, focal_length, translation, center):
        for node in self.scene.get_nodes():
            if node.name == 'camera':
                self.scene.remove_node(node)

        pc = pyrender.IntrinsicsCamera(
            fx=focal_length,
            fy=focal_length,
            cx=center[0],
            cy=center[1],
        )
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = translation.copy()
        camera_pose[0, 3] *= (-1)
        self.scene.add(pc, pose=camera_pose, name='camera')

    @torch.no_grad()
    def __call__(self,
                 vertices: Tensor,
                 faces: Union[Tensor, Array],
                 focal_length: Union[Tensor, Array],
                 camera_translation: Union[Tensor, Array],
                 camera_center: Union[Tensor, Array],
                 bg_imgs: Array,
                 render_bg: bool = True,
                 deg: float = 0,
                 return_with_alpha: bool = False,
                 body_color: List[float] = None,
                 **kwargs):
        '''
            Parameters
            ----------
            vertices: BxVx3, torch.Tensor
                The torch Tensor that contains the current vertices to be drawn
            faces: Fx3, np.array
                The faces of the meshes to be drawn. Right now only support a
                batch of meshes with the same topology
            focal_length: B, torch.Tensor
                The focal length used by the perspective camera
            camera_translation: Bx3, torch.Tensor
                The translation of the camera estimated by the network
            camera_center: Bx2, torch.Tensor
                The center of the camera in pixels
            bg_imgs: np.ndarray
                Optional background images used for overlays
            render_bg: bool, optional
                Render on top of the background image
            deg: float, optional
                Degrees to rotate the mesh around itself. Used to render the
                same mesh from multiple viewpoints. Defaults to 0 degrees
            return_with_alpha: bool, optional
                Whether to return the rendered image with an alpha channel.
                Default value is False.
            body_color: list, optional
                The color used to render the image.
        '''
        if torch.is_tensor(vertices):
            vertices = vertices.detach().cpu().numpy()
        if torch.is_tensor(faces):
            faces = faces.detach().cpu().numpy()
        if torch.is_tensor(focal_length):
            focal_length = focal_length.detach().cpu().numpy()
        if torch.is_tensor(camera_translation):
            camera_translation = camera_translation.detach().cpu().numpy()
        if torch.is_tensor(camera_center):
            camera_center = camera_center.detach().cpu().numpy()
        batch_size = vertices.shape[0]

        output_imgs = []
        for bidx in range(batch_size):
            if body_color is None:
                body_color = COLORS['N']

            _, H, W = bg_imgs[bidx].shape
            # Update the renderer's viewport
            self.renderer.viewport_height = H
            self.renderer.viewport_width = W

            self.update_camera(
                focal_length=focal_length[bidx],
                translation=camera_translation[bidx],
                center=camera_center[bidx],
            )
            self.update_mesh(
                vertices[bidx], faces, body_color=body_color, deg=deg)

            flags = (pyrender.RenderFlags.RGBA |
                     pyrender.RenderFlags.SKIP_CULL_FACES)
            color, depth = self.renderer.render(self.scene, flags=flags)
            color = np.transpose(color, [2, 0, 1]).astype(np.float32) / 255.0
            color = np.clip(color, 0, 1)

            if render_bg:
                if return_with_alpha:
                    valid_mask = (color[3] > 0)[np.newaxis]

                    if bg_imgs[bidx].shape[0] < 4:
                        curr_bg_img = np.concatenate(
                            [bg_imgs[bidx],
                             np.ones_like(bg_imgs[bidx, [0], :, :])
                             ], axis=0)
                    else:
                        curr_bg_img = bg_imgs[bidx]

                    output_img = (color * valid_mask +
                                  (1 - valid_mask) * curr_bg_img)
                    output_imgs.append(np.clip(output_img, 0, 1))
                else:
                    valid_mask = (color[3] > 0)[np.newaxis]

                    output_img = (color[:-1] * valid_mask +
                                  (1 - valid_mask) * bg_imgs[bidx])
                    output_imgs.append(np.clip(output_img, 0, 1))
            else:
                if return_with_alpha:
                    output_imgs.append(color)
                else:
                    output_imgs.append(color[:-1])
        return np.stack(output_imgs, axis=0)
