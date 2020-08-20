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


from abc import ABC, abstractmethod

import torch
from loguru import logger


class GenericTarget(ABC):
    def __init__(self):
        super(GenericTarget, self).__init__()
        self.extra_fields = {}

    def __del__(self):
        if hasattr(self, 'extra_fields'):
            self.extra_fields.clear()

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def delete_field(self, field):
        if field in self.extra_fields:
            del self.extra_fields[field]

    def transpose(self, method):
        for k, v in self.extra_fields.items():
            if isinstance(v, GenericTarget):
                v = v.transpose(method)
            self.add_field(k, v)
        self.add_field('is_flipped', True)
        return self

    def rotate(self, *args, **kwargs):
        for k, v in self.extra_fields.items():
            if isinstance(v, GenericTarget):
                v = v.rotate(*args, **kwargs)
        self.add_field('rot', kwargs.get('rot', 0))
        return self

    def crop(self, *args, **kwargs):
        for k, v in self.extra_fields.items():
            if isinstance(v, GenericTarget):
                v = v.crop(*args, **kwargs)
        return self

    def resize(self, *args, **kwargs):
        for k, v in self.extra_fields.items():
            if isinstance(v, GenericTarget):
                v = v.resize(*args, **kwargs)
            self.add_field(k, v)
        return self

    def to_tensor(self, *args, **kwargs):
        for k, v in self.extra_fields.items():
            if isinstance(v, GenericTarget):
                v.to_tensor(*args, **kwargs)
            self.add_field(k, v)

    def to(self, *args, **kwargs):
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
        return self
