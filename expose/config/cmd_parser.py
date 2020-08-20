from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import sys
import os

import argparse
from loguru import logger

from yacs.config import CfgNode as CN
from . import cfg


def set_face_contour(node, use_face_contour=False):
    for key in node:
        if 'use_face_contour' in key:
            node[key] = use_face_contour
        if isinstance(node[key], CN):
            set_face_contour(node[key], use_face_contour=use_face_contour)


def parse_args(argv=None):
    arg_formatter = argparse.ArgumentDefaultsHelpFormatter

    description = 'PyTorch SMPL-X Regressor with Attention'
    parser = argparse.ArgumentParser(formatter_class=arg_formatter,
                                     description=description)

    parser.add_argument('--exp-cfg', type=str, dest='exp_cfg',
                        help='The configuration of the experiment')
    parser.add_argument('--exp-opts', default=[], dest='exp_opts',
                        nargs='*',
                        help='The configuration of the Detector')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--num-gpus', dest='num_gpus',
                        default=1, type=int,
                        help='Number of gpus')
    parser.add_argument('--backend', dest='backend',
                        default='nccl', type=str,
                        choices=['nccl', 'gloo'],
                        help='Backend used for multi-gpu training')

    cmd_args = parser.parse_args()

    cfg.merge_from_file(cmd_args.exp_cfg)
    cfg.merge_from_list(cmd_args.exp_opts)

    use_face_contour = cfg.datasets.use_face_contour
    set_face_contour(cfg, use_face_contour=use_face_contour)

    cfg.network.use_sync_bn = (cfg.network.use_sync_bn and
                               cmd_args.num_gpus > 1)
    cfg.local_rank = cmd_args.local_rank
    cfg.num_gpus = cmd_args.num_gpus

    return cfg
