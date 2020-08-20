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

import torch

from loguru import logger


class Checkpointer(object):
    def __init__(self, model, optimizer=None, scheduler=None,
                 adv_optimizer=None,
                 pretrained='',
                 distributed=False,
                 rank=0,
                 save_dir='/tmp/exp'):
        self.rank = rank
        self.distributed = distributed

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.adv_optimizer = adv_optimizer

        self.save_dir = save_dir
        if self.rank == 0:
            logger.info(f'Creating directory {self.save_dir}')
            os.makedirs(self.save_dir, exist_ok=True)
        self.pretrained = pretrained

    def save_checkpoint(self, name, **kwargs):
        if self.rank > 0:
            return
        ckpt_data = {}
        ckpt_data['model'] = self.model.state_dict()

        if self.optimizer is not None:
            logger.info('Adding optimizer state ...')
            ckpt_data['optimizer'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            logger.info('Adding scheduler state ...')
            ckpt_data['scheduler'] = self.scheduler.state_dict()
        if self.adv_optimizer is not None:
            logger.info('Adding discriminator optimizer state ...')
            ckpt_data['adv_optimizer'] = self.adv_optimizer.state_dict()

        ckpt_data.update(kwargs)

        curr_ckpt_fn = osp.join(self.save_dir, name)
        logger.info('Saving checkpoint to {}'.format(curr_ckpt_fn))
        torch.save(ckpt_data, curr_ckpt_fn)
        with open(osp.join(self.save_dir, 'latest_checkpoint'), 'w') as f:
            f.write(curr_ckpt_fn)
        ckpt_data.clear()

    def load_checkpoint(self):
        save_fn = osp.join(self.save_dir, 'latest_checkpoint')

        load_pretrained = False
        if not osp.exists(save_fn):
            # If no previous checkpoint exists, load from the pretrained model
            if len(self.pretrained) > 1:
                self.pretrained = osp.expandvars(self.pretrained)
                load_pretrained = True
                save_fn = osp.join(
                    self.pretrained, 'checkpoints', 'latest_checkpoint')
            # If neither the pretrained model exists nor there is a previous
            # checkpoint then initialize from scratch
            if not osp.exists(save_fn):
                logger.warning(f'No checkpoint found in {self.save_dir}!')
                return {}

        logger.info('Load pretrained: {}', load_pretrained)
        with open(save_fn, 'r') as f:
            latest_ckpt_fn = f.read().strip()
        logger.warning(f'Loading checkpoint from {latest_ckpt_fn}!')

        if self.distributed:
            map_location = torch.device(f'cuda:{self.rank}')
        else:
            map_location = torch.device('cpu')
        ckpt_data = torch.load(latest_ckpt_fn, map_location=map_location)

        if load_pretrained:
            if 'face_idxs' in ckpt_data['model']:
                del ckpt_data['model']['face_idxs']
            if 'smplx.smplx_loss.body_idxs' in ckpt_data['model']:
                del ckpt_data['model']['smplx.smplx_loss.body_idxs']
            if 'smplx.smplx_loss.hand_idxs' in ckpt_data['model']:
                del ckpt_data['model']['smplx.smplx_loss.hand_idxs']
            if 'smplx.smplx_loss.face_idxs' in ckpt_data['model']:
                del ckpt_data['model']['smplx.smplx_loss.face_idxs']
            if 'smplx.smplx_loss.left_hand_idxs' in ckpt_data['model']:
                del ckpt_data['model']['smplx.smplx_loss.left_hand_idxs']
            if 'smplx.smplx_loss.right_hand_idxs' in ckpt_data['model']:
                del ckpt_data['model']['smplx.smplx_loss.right_hand_idxs']
        if 'smplx.head_idxs' in ckpt_data['model']:
            del ckpt_data['model']['smplx.head_idxs']

        missing, unexpected = self.model.load_state_dict(
            #  ckpt_data['model'], strict=not load_pretrained)
            ckpt_data['model'], strict=False)
        if len(missing) > 0:
            logger.warning(
                f'The following keys were not found: {missing}')
        if len(unexpected):
            logger.warning(
                f'The following keys were not expected: {unexpected}')

        if self.optimizer is not None and 'optimizer' in ckpt_data:
            if not load_pretrained:
                logger.warning('Loading optimizer data from: {}'.format(
                    self.save_dir))
                self.optimizer.load_state_dict(ckpt_data['optimizer'])

        if self.scheduler is not None and 'scheduler' in ckpt_data:
            if not load_pretrained:
                logger.warning('Loading scheduler data from: {}'.format(
                    self.save_dir))
                self.scheduler.load_state_dict(ckpt_data['scheduler'])
        if self.adv_optimizer is not None and 'adv_optimizer' in ckpt_data:
            if not load_pretrained:
                logger.warning(
                    'Loading discriminator optim data from: {}'.format(
                        self.save_dir))
                self.adv_optimizer.load_state_dict(
                    ckpt_data['adv_optimizer'])

        if load_pretrained:
            ckpt_data['iteration'] = 0
            ckpt_data['epoch_number'] = 0

        return ckpt_data
