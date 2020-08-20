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

from copy import deepcopy

import numpy as np

import torch

from .generic_target import GenericTarget

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class Vertices(GenericTarget):
    def __init__(self, vertices,
                 bc=None,
                 closest_faces=None,
                 flip=True,
                 flip_index=0, dtype=torch.float32):
        super(Vertices, self).__init__()
        self.vertices = vertices
        self.flip_index = flip_index
        self.closest_faces = closest_faces
        self.bc = bc
        self.flip = flip

    def __getitem__(self, key):
        if key == 'vertices':
            return self.vertices
        else:
            raise ValueError('Unknown key: {}'.format(key))

    def transpose(self, method):
        if not self.flip:
            return self
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        if self.closest_faces is None or self.bc is None:
            raise RuntimeError(f'Cannot support flip without correspondences')

        flipped_vertices = self.vertices.copy()
        flipped_vertices[:, self.flip_index] *= -1

        closest_tri_vertices = flipped_vertices[self.closest_faces].copy()
        #  flipped_vertices = flipped_vertices[
        #  self.flip_correspondences].copy()
        flipped_vertices = (
            self.bc[:, :, np.newaxis] * closest_tri_vertices).sum(axis=1)
        flipped_vertices = flipped_vertices.astype(self.vertices.dtype)

        vertices = type(self)(flipped_vertices, flip_index=self.flip_index,
                              bc=self.bc, closest_faces=self.closest_faces)

        for k, v in self.extra_fields.items():
            if isinstance(v, GenericTarget):
                v = v.transpose(method)
            vertices.add_field(k, v)
        self.add_field('is_flipped', True)
        return vertices

    def to_tensor(self, *args, **kwargs):
        self.vertices = torch.from_numpy(self.vertices)
        for k, v in self.extra_fields.items():
            if isinstance(v, GenericTarget):
                v.to_tensor(*args, **kwargs)

    def crop(self, *args, **kwargs):
        vertices = self.vertices.copy()
        field = type(self)(vertices, flip_index=self.flip_index,
                           bc=self.bc,
                           closest_faces=self.closest_faces)

        for k, v in self.extra_fields.items():
            if isinstance(v, GenericTarget):
                v = v.crop(*args, **kwargs)
            field.add_field(k, v)

        self.add_field('rot', kwargs.get('rot', 0))
        return field

    def rotate(self, rot=0, *args, **kwargs):
        if rot == 0:
            return self
        vertices = self.vertices.copy()
        R = np.array([[np.cos(np.deg2rad(-rot)),
                       -np.sin(np.deg2rad(-rot)), 0],
                      [np.sin(np.deg2rad(-rot)),
                       np.cos(np.deg2rad(-rot)), 0],
                      [0, 0, 1]], dtype=np.float32)
        vertices = np.dot(vertices, R.T)

        vertices = type(self)(vertices, flip_index=self.flip_index,
                              bc=self.bc, closest_faces=self.closest_faces)
        for k, v in self.extra_fields.items():
            if isinstance(v, GenericTarget):
                v = v.rotate(rot=rot, *args, **kwargs)
            vertices.add_field(k, v)

        self.add_field('rot', rot)
        return vertices

    def to(self, *args, **kwargs):
        vertices = type(self)(
            self.vertices.to(*args, **kwargs), flip_index=self.flip_index,
            bc=self.bc,
            closest_faces=self.closest_faces)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            vertices.add_field(k, v)
        return vertices
