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

import numpy as np
from copy import deepcopy

import torch
import cv2

from .generic_target import GenericTarget

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class Expression(GenericTarget):
    """ Stores the expression params
    """

    def __init__(self, expression, dtype=torch.float32, **kwargs):
        super(Expression, self).__init__()
        self.expression = expression

    def to_tensor(self, *args, **kwargs):
        if not torch.is_tensor(self.expression):
            self.expression = torch.from_numpy(self.expression)
        for k, v in self.extra_fields.items():
            if isinstance(v, GenericTarget):
                v.to_tensor(*args, **kwargs)

    def transpose(self, method):
        field = type(self)(expression=deepcopy(self.expression))
        for k, v in self.extra_fields.items():
            if isinstance(v, GenericTarget):
                v = v.transpose(method)
            field.add_field(k, v)
        self.add_field('is_flipped', True)
        return field

    def resize(self, size, *args, **kwargs):
        field = type(self)(expression=self.expression)
        for k, v in self.extra_fields.items():
            if isinstance(v, GenericTarget):
                v = v.resize(size, *args, **kwargs)
            field.add_field(k, v)
        return field

    def crop(self, rot=0, *args, **kwargs):
        field = type(self)(expression=self.expression)

        for k, v in self.extra_fields.items():
            if isinstance(v, GenericTarget):
                v = v.crop(rot=rot, *args, **kwargs)
            field.add_field(k, v)

        self.add_field('rot', rot)
        return field

    def to(self, *args, **kwargs):
        field = type(self)(expression=self.expression.to(*args, **kwargs))
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            field.add_field(k, v)
        return field
