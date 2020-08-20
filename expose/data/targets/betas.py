
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


class Betas(GenericTarget):
    """ Stores the shape params
    """

    def __init__(self, betas, dtype=torch.float32, **kwargs):
        super(Betas, self).__init__()

        self.betas = betas

    def to_tensor(self, *args, **kwargs):
        if not torch.is_tensor(self.betas):
            self.betas = torch.from_numpy(self.betas)
        for k, v in self.extra_fields.items():
            if isinstance(v, GenericTarget):
                v.to_tensor(*args, **kwargs)

    def to(self, *args, **kwargs):
        field = type(self)(betas=self.betas.to(*args, **kwargs))
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            field.add_field(k, v)
        return field
