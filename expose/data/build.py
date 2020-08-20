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

from typing import NewType, List, Tuple, Union
import sys
import os.path as osp

from loguru import logger

import time
import functools
import torch
import torch.utils.data as dutils

from copy import deepcopy
from . import datasets
from .utils import EqualSampler

from .targets.generic_target import GenericTarget
from .targets.image_list import to_image_list, ImageList
from .transforms import build_transforms

from expose.utils.typing_utils import Tensor

DEFAULT_NUM_WORKERS = {
    'train': 0,
    'val': 0,
    'test': 0
}


def make_data_sampler(dataset, is_train=True,
                      shuffle=True, is_distributed=False):
    if is_train:
        sampler = dutils.RandomSampler(dataset)
    else:
        sampler = dutils.SequentialSampler(dataset)
    return sampler


def make_head_dataset(name, dataset_cfg, transforms,
                      num_betas=10, num_expression_coeffs=10,
                      **kwargs):
    if name == 'ehf':
        obj = datasets.EHF
    elif name == 'curated_fits':
        obj = datasets.CuratedFittings
    elif name == 'spinx':
        obj = datasets.SPINX
    elif name == 'ffhq':
        obj = datasets.FFHQ
    elif name == 'openpose':
        obj = datasets.OpenPose
    elif name == 'stirling3d':
        obj = datasets.Stirling3D
    else:
        raise ValueError('Unknown dataset: {}'.format(name))

    args = dict(**dataset_cfg[name])
    args.update(kwargs)

    vertex_flip_correspondences = osp.expandvars(dataset_cfg.get(
        'vertex_flip_correspondences', ''))
    dset_obj = obj(transforms=transforms,
                   head_only=True,
                   num_betas=num_betas,
                   num_expression_coeffs=num_expression_coeffs,
                   vertex_flip_correspondences=vertex_flip_correspondences,
                   **args)

    logger.info(f'Created head dataset: {dset_obj.name()}')
    return dset_obj


def make_hand_dataset(name, dataset_cfg, transforms,
                      num_betas=10, num_expression_coeffs=10,
                      **kwargs):
    if name == 'ehf':
        obj = datasets.EHF
    elif name == 'curated_fits':
        obj = datasets.CuratedFittings
    elif name == 'spinx':
        obj = datasets.SPINX
    elif name == 'openpose':
        obj = datasets.OpenPose
    elif name == 'freihand':
        obj = datasets.FreiHand
    else:
        raise ValueError(f'Unknown dataset: {name}')

    logger.info(f'Building dataset: {name}')
    args = dict(**dataset_cfg[name])
    args.update(kwargs)
    vertex_flip_correspondences = osp.expandvars(dataset_cfg.get(
        'vertex_flip_correspondences', ''))

    dset_obj = obj(transforms=transforms, num_betas=num_betas, hand_only=True,
                   num_expression_coeffs=num_expression_coeffs,
                   vertex_flip_correspondences=vertex_flip_correspondences,
                   **args)

    logger.info(f'Created dataset: {dset_obj.name()}')
    return dset_obj


def make_body_dataset(name, dataset_cfg, transforms,
                      num_betas=10,
                      num_expression_coeffs=10,
                      **kwargs):
    if name == 'ehf':
        obj = datasets.EHF
    elif name == 'curated_fits':
        obj = datasets.CuratedFittings
    elif name == 'threedpw':
        obj = datasets.ThreeDPW
    elif name == 'spin':
        obj = datasets.SPIN
    elif name == 'spinx':
        obj = datasets.SPINX
    elif name == 'lsp_test':
        obj = datasets.LSPTest
    elif name == 'openpose':
        obj = datasets.OpenPose
    elif name == 'tracks':
        obj = datasets.OpenPoseTracks
    else:
        raise ValueError(f'Unknown dataset: {name}')

    args = dict(**dataset_cfg[name])
    args.update(kwargs)

    vertex_flip_correspondences = osp.expandvars(dataset_cfg.get(
        'vertex_flip_correspondences', ''))
    dset_obj = obj(transforms=transforms, num_betas=num_betas,
                   vertex_flip_correspondences=vertex_flip_correspondences,
                   num_expression_coeffs=num_expression_coeffs,
                   **args)

    logger.info('Created dataset: {}', dset_obj.name())
    return dset_obj


class MemoryPinning(object):
    def __init__(
        self,
        full_img_list: Union[ImageList, List[Tensor]],
        images: Tensor,
        targets: List[GenericTarget]
    ):
        super(MemoryPinning, self).__init__()
        self.img_list = full_img_list
        self.images = images
        self.targets = targets

    def pin_memory(
            self
    ) -> Tuple[Union[ImageList, List[Tensor]], Tensor, List[GenericTarget]]:
        if self.img_list is not None:
            if isinstance(self.img_list, ImageList):
                self.img_list.pin_memory()
            elif isinstance(self.img_list, (list, tuple)):
                self.img_list = [x.pin_memory() for x in self.img_list]
        return (
            self.img_list,
            self.images.pin_memory(),
            self.targets,
        )


def collate_batch(batch, use_shared_memory=False, return_full_imgs=False,
                  pin_memory=True):
    if return_full_imgs:
        images, cropped_images, targets, _ = zip(*batch)
    else:
        _, cropped_images, targets, _ = zip(*batch)

    out_targets = []
    for t in targets:
        if t is None:
            continue
        if type(t) == list:
            out_targets += t
        else:
            out_targets.append(t)
    out_cropped_images = []
    for img in cropped_images:
        if img is None:
            continue
        if len(img.shape) < 4:
            img.unsqueeze_(dim=0)
        out_cropped_images.append(img.clone())

    if len(out_cropped_images) < 1:
        return None, None, None

    full_img_list = None
    if return_full_imgs:
        #  full_img_list = to_image_list(images)
        full_img_list = images
    out = None
    if use_shared_memory:
        numel = sum([x.numel() for x in out_cropped_images if x is not None])
        storage = out_cropped_images[0].storage()._new_shared(numel)
        out = out_cropped_images[0].new(storage)

    #  if not return_full_imgs:
        #  del images
        #  images = None

    batch.clear()
    #  del targets, batch
    if pin_memory:
        return MemoryPinning(
            full_img_list,
            torch.cat(out_cropped_images, 0, out=out),
            out_targets
        )
    else:
        return full_img_list, torch.cat(
            out_cropped_images, 0, out=out), out_targets


def make_equal_sampler(datasets, batch_size=32, shuffle=True, ratio_2d=0.5):
    batch_sampler = EqualSampler(
        datasets, batch_size=batch_size, shuffle=shuffle, ratio_2d=ratio_2d)
    out_dsets_lst = [dutils.ConcatDataset(datasets) if len(datasets) > 1 else
                     datasets[0]]
    return batch_sampler, out_dsets_lst


def make_data_loader(dataset, batch_size=32, num_workers=0,
                     is_train=True, sampler=None, collate_fn=None,
                     shuffle=True, is_distributed=False,
                     batch_sampler=None):
    if batch_sampler is None:
        sampler = make_data_sampler(
            dataset, is_train=is_train,
            shuffle=shuffle, is_distributed=is_distributed)

    if batch_sampler is None:
        assert sampler is not None, (
            'Batch sampler and sampler can\'t be "None" at the same time')
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            collate_fn=collate_fn,
            drop_last=True and is_train,
            pin_memory=True,
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            collate_fn=collate_fn,
            batch_sampler=batch_sampler,
            pin_memory=True,
        )
    return data_loader


def make_all_data_loaders(exp_cfg, split='train', start_iter=0, **kwargs):
    is_train = 'train' in split
    num_betas = exp_cfg.body_model.num_betas
    num_expression_coeffs = exp_cfg.body_model.num_expression_coeffs

    dataset_cfg = exp_cfg.get('datasets', {})

    body_dsets_cfg = dataset_cfg.get('body', {})
    body_dset_names = body_dsets_cfg.get('splits', {})[split]

    body_transfs_cfg = body_dsets_cfg.get('transforms', {})
    body_transforms = build_transforms(body_transfs_cfg, is_train=is_train)

    hand_dsets_cfg = dataset_cfg.get('hand', {})
    hand_dset_names = hand_dsets_cfg.get('splits', {})[split]
    hand_transfs_cfg = hand_dsets_cfg.get('transforms', {})
    hand_transforms = build_transforms(hand_transfs_cfg, is_train=is_train)

    head_dsets_cfg = dataset_cfg.get('head', {})
    head_dset_names = head_dsets_cfg.get('splits', {})[split]
    head_transfs_cfg = head_dsets_cfg.get('transforms', {})
    head_transforms = build_transforms(head_transfs_cfg, is_train=is_train)

    body_datasets = []
    for dataset_name in body_dset_names:
        dset = make_body_dataset(dataset_name, body_dsets_cfg,
                                 transforms=body_transforms,
                                 num_betas=num_betas,
                                 num_expression_coeffs=num_expression_coeffs,
                                 is_train=is_train, split=split, **kwargs)
        body_datasets.append(dset)

    hand_datasets = []
    for dataset_name in hand_dset_names:
        dset = make_hand_dataset(dataset_name, hand_dsets_cfg,
                                 transforms=hand_transforms,
                                 num_betas=num_betas,
                                 num_expression_coeffs=num_expression_coeffs,
                                 is_train=is_train, split=split, **kwargs)
        hand_datasets.append(dset)

    head_datasets = []
    for dataset_name in head_dset_names:
        dset = make_head_dataset(dataset_name, head_dsets_cfg,
                                 transforms=head_transforms,
                                 num_betas=num_betas,
                                 num_expression_coeffs=num_expression_coeffs,
                                 is_train=is_train, split=split, **kwargs)
        head_datasets.append(dset)

    use_equal_sampling = exp_cfg.datasets.use_equal_sampling

    # Hard-coded for now
    shuffle = is_train
    is_distributed = False

    body_batch_size = body_dsets_cfg.get('batch_size', 64)
    body_ratio_2d = body_dsets_cfg.get('ratio_2d', 0.5)

    hand_batch_size = hand_dsets_cfg.get('batch_size', 64)
    hand_ratio_2d = hand_dsets_cfg.get('ratio_2d', 0.5)

    head_batch_size = head_dsets_cfg.get('batch_size', 64)
    head_ratio_2d = head_dsets_cfg.get('ratio_2d', 0.5)

    body_num_workers = body_dsets_cfg.get(
        'num_workers', DEFAULT_NUM_WORKERS).get(split, 0)
    logger.info(f'{split.upper()} Body num workers: {body_num_workers}')

    network_cfg = exp_cfg.network
    return_full_imgs = (network_cfg.get('apply_hand_network_on_body', True) or
                        network_cfg.get('apply_head_network_on_body', True))
    logger.info(f'Return full resolution images: {return_full_imgs}')
    body_collate_fn = functools.partial(
        collate_batch, use_shared_memory=body_num_workers > 0,
        return_full_imgs=return_full_imgs)

    hand_num_workers = hand_dsets_cfg.get(
        'num_workers', DEFAULT_NUM_WORKERS).get(split, 0)
    hand_collate_fn = functools.partial(
        collate_batch, use_shared_memory=hand_num_workers > 0)
    #  collate_batch, use_shared_memory=False)

    head_num_workers = head_dsets_cfg.get(
        'num_workers', DEFAULT_NUM_WORKERS).get(split, 0)
    head_collate_fn = functools.partial(
        collate_batch, use_shared_memory=head_num_workers > 0)
    #  collate_batch, use_shared_memory=False)

    body_batch_sampler, hand_batch_sampler, head_batch_sampler = [None] * 3
    # Equal sampling should only be used during training and only if there
    # are multiple datasets
    if is_train and use_equal_sampling:
        body_batch_sampler, body_datasets = make_equal_sampler(
            body_datasets, batch_size=body_batch_size,
            shuffle=shuffle, ratio_2d=body_ratio_2d)
        if len(hand_datasets) > 0:
            hand_batch_sampler, hand_datasets = make_equal_sampler(
                hand_datasets, batch_size=hand_batch_size,
                shuffle=shuffle, ratio_2d=hand_ratio_2d)
        if len(head_datasets) > 0:
            head_batch_sampler, head_datasets = make_equal_sampler(
                head_datasets, batch_size=head_batch_size,
                shuffle=shuffle, ratio_2d=head_ratio_2d)

    body_data_loaders = []
    for body_dataset in body_datasets:
        body_data_loaders.append(
            make_data_loader(body_dataset, batch_size=body_batch_size,
                             num_workers=body_num_workers,
                             is_train=is_train,
                             batch_sampler=body_batch_sampler,
                             collate_fn=body_collate_fn,
                             shuffle=shuffle, is_distributed=is_distributed))
    hand_data_loaders = []
    for hand_dataset in hand_datasets:
        hand_data_loaders.append(
            make_data_loader(hand_dataset, batch_size=hand_batch_size,
                             num_workers=hand_num_workers,
                             is_train=is_train,
                             batch_sampler=hand_batch_sampler,
                             collate_fn=hand_collate_fn,
                             shuffle=shuffle, is_distributed=is_distributed))
    head_data_loaders = []
    for head_dataset in head_datasets:
        head_data_loaders.append(
            make_data_loader(head_dataset, batch_size=head_batch_size,
                             num_workers=head_num_workers,
                             is_train=is_train,
                             batch_sampler=head_batch_sampler,
                             collate_fn=head_collate_fn,
                             shuffle=shuffle, is_distributed=is_distributed))

    use_adv_training = exp_cfg.use_adv_training
    if is_train:
        assert len(body_data_loaders) == 1, (
            'There should be a single body loader,'
            f' not {len(body_data_loaders)}')
        #  assert len(hand_data_loaders) == 1, (
        #  'There should be a single hand loader,'
        #  f' not {len(hand_data_loaders)}')
        #  assert len(head_data_loaders) == 1, (
        #  'There should be a single head loader,'
        #  f' not {len(head_data_loaders)}')
        dloaders = {
            'body': body_data_loaders[0],
        }
        if len(hand_data_loaders) > 0:
            dloaders['hand'] = hand_data_loaders[0]
        if len(head_data_loaders) > 0:
            dloaders['head'] = head_data_loaders[0]
        if use_adv_training:
            raise NotImplementedError
        return dloaders

    return {
        'body': body_data_loaders,
        'hand': hand_data_loaders,
        'head': head_data_loaders,
    }
