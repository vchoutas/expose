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

from typing import List, Union
import sys
import numpy as np
import torch
from loguru import logger
from expose.utils.typing_utils import Tensor


class ImageList(object):
    def __init__(self, images: torch.Tensor,
                 img_sizes: List[torch.Size],
                 padding=None):
        self.images = images
        self.img_sizes = img_sizes
        self.sizes_tensor = torch.stack(
            [torch.tensor(s) if not torch.is_tensor(s) else s
             for s in img_sizes]).to(dtype=self.images.dtype)
        if padding is not None:
            self.padding_tensor = torch.stack(
                [torch.tensor(s) if not torch.is_tensor(s) else s
                 for s in padding]).to(dtype=self.images.dtype)
        self._shape = self.images.shape

    def as_image_list(self) -> List[Tensor]:
        return self.images

    def as_tensor(self) -> Tensor:
        return self.images

    @property
    def shape(self):
        return self._shape

    @property
    def device(self):
        return self.images.device

    @property
    def dtype(self):
        return self.images.dtype

    def pin_memory(self):
        if not self.images.is_pinned():
            self.images = self.images.pin_memory()
        return self

    def __del__(self):
        del self.images
        del self.sizes_tensor
        del self.img_sizes

    def to(self, *args, **kwargs):
        images = self.images.to(*args, **kwargs)
        sizes_tensor = self.sizes_tensor.to(*args, **kwargs)
        return ImageList(images, sizes_tensor)


class ImageListPacked(object):
    def __init__(
        self,
        packed_tensor: Tensor,
        starts: List[int],
        num_elements: List[int],
        img_sizes: List[torch.Size],
    ) -> None:
        '''
        '''
        self.packed_tensor = packed_tensor
        self.starts = starts
        self.num_elements = num_elements
        self.img_sizes = img_sizes

        self._shape = [len(starts)] + [max(s) for s in zip(*img_sizes)]

        _, self.heights, self.widths = zip(*img_sizes)

    def as_tensor(self):
        return self.packed_tensor

    def as_image_list(self):
        out_list = []

        sizes = [shape[1:] for shape in self.img_sizes]
        H, W = [max(s) for s in zip(*sizes)]

        out_shape = (3, H, W)
        for ii in range(len(self.img_sizes)):
            start = self.starts[ii]
            end = self.starts[ii] + self.num_elements[ii]
            c, h, w = self.img_sizes[ii]
            img = self.packed_tensor[start:end].reshape(c, h, w)
            out_img = torch.zeros(
                out_shape, device=self.device, dtype=self.dtype)
            out_img[:c, :h, :w] = img
            out_list.append(out_img.detach().cpu().numpy())

        return out_list

    @property
    def shape(self):
        return self._shape

    @property
    def device(self):
        return self.packed_tensor.device

    @property
    def dtype(self):
        return self.packed_tensor.dtype

    def pin_memory(self):
        if not self.images.is_pinned():
            self.images = self.images.pin_memory()
        return self

    def to(self, *args, **kwargs):
        self.packed_tensor = self.packed_tensor.to(*args, **kwargs)
        return self


def to_image_list_concat(
        images: List[Tensor]
) -> ImageList:
    if images is None:
        return images
    if isinstance(images, ImageList):
        return images
    sizes = [img.shape[1:] for img in images]
    #  logger.info(sizes)
    H, W = [max(s) for s in zip(*sizes)]

    batch_size = len(images)
    batched_shape = (batch_size, images[0].shape[0], H, W)
    batched = torch.zeros(
        batched_shape, device=images[0].device, dtype=images[0].dtype)

    #  for img, padded in zip(images, batched):
    #  shape = img.shape
    #  padded[:shape[0], :shape[1], :shape[2]] = img
    padding = None
    for ii, img in enumerate(images):
        shape = img.shape
        batched[ii, :shape[0], :shape[1], :shape[2]] = img

    return ImageList(batched, sizes, padding=padding)


def to_image_list_packed(images: List[Tensor]) -> ImageListPacked:
    if images is None:
        return images
    if isinstance(images, ImageListPacked):
        return images
    # Store the size of each image
    # Compute the number of elements in each image
    sizes = [img.shape for img in images]
    num_element_list = [np.prod(s) for s in sizes]
    # Compute the total number of elements

    packed = torch.cat([img.flatten() for img in images])
    # Compute the start index of each image tensor in the packed tensor
    starts = [0] + list(np.cumsum(num_element_list))[:-1]
    return ImageListPacked(packed, starts, num_element_list, sizes)


def to_image_list(
    images: List[Tensor],
    use_packed=False
) -> Union[ImageList, ImageListPacked]:
    '''
    '''
    func = to_image_list_packed if use_packed else to_image_list_concat
    return func(images)
