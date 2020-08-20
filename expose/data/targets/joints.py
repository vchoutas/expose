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

import torch
from .generic_target import GenericTarget


class Joints(GenericTarget):
    def __init__(self, joints, **kwargs):
        super(Joints, self).__init__()
        self.joints = joints

    def __repr__(self):
        s = self.__class__.__name__
        return s

    def to_tensor(self, *args, **kwargs):
        self.joints = torch.tensor(self.joints)

        for k, v in self.extra_fields.items():
            if isinstance(v, GenericTarget):
                v.to_tensor(*args, **kwargs)

    def __getitem__(self, key):
        if key == 'joints':
            return self.joints
        else:
            raise ValueError('Unknown key: {}'.format(key))

    def __len__(self):
        return 1

    def to(self, *args, **kwargs):
        joints = type(self)(self.joints.to(*args, **kwargs))
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            joints.add_field(k, v)
        return joints
