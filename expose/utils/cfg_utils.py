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

BUILTINS = [list, dict, tuple, set, str, int, float, bool]


def cfg_to_dict(cfg_node):
    if type(cfg_node) in BUILTINS:
        return cfg_node
    else:
        curr_dict = dict(cfg_node)
        for key, val in curr_dict.items():
            curr_dict[key] = cfg_to_dict(val)
        return curr_dict
