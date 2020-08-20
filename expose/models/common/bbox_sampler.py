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

from typing import Tuple, Union, Dict
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger

from expose.data.utils import points_to_bbox
from expose.data.targets import ImageList, ImageListPacked, GenericTarget
from expose.utils.typing_utils import Tensor


class ToCrops(nn.Module):
    def __init__(self) -> None:
        super(ToCrops, self).__init__()

    def forward(
        self,
        full_imgs: Union[ImageList, ImageListPacked],
        points: Tensor,
        targets: GenericTarget,
        scale_factor: float = 1.0,
        crop_size: int = 256
    ) -> Dict[str, Tensor]:
        num_imgs, _, H, W = full_imgs.shape
        device = points.device
        dtype = points.dtype

        # Get the image to crop transformations and bounding box sizes
        crop_transforms = []
        img_bbox_sizes = []
        for t in targets:
            crop_transforms.append(t.get_field('crop_transform'))
            img_bbox_sizes.append(t.get_field('bbox_size'))

        img_bbox_sizes = torch.tensor(
            img_bbox_sizes, dtype=dtype, device=device)

        crop_transforms = torch.tensor(
            crop_transforms, dtype=dtype, device=device)
        inv_crop_transforms = torch.inverse(crop_transforms)

        center_body_crop, bbox_size = points_to_bbox(
            points, bbox_scale_factor=scale_factor)

        orig_bbox_size = bbox_size / crop_size * img_bbox_sizes
        # Compute the center of the crop in the original image
        center = (torch.einsum(
            'bij,bj->bi', [inv_crop_transforms[:, :2, :2], center_body_crop]) +
            inv_crop_transforms[:, :2, 2])

        return {'center': center.reshape(-1, 2),
                'orig_bbox_size': orig_bbox_size,
                'bbox_size': bbox_size.reshape(-1),
                'inv_crop_transforms': inv_crop_transforms,
                'center_body_crop': 2 * center_body_crop / crop_size - 1,
                }


class CropSampler(nn.Module):
    def __init__(
        self,
        crop_size: int = 256
    ) -> None:
        ''' Uses bilinear sampling to extract square crops

            This module expects a high resolution image as input and a bounding
            box, described by its' center and size. It then proceeds to extract
            a sub-image using the provided information through bilinear
            interpolation.

            Parameters
            ----------
                crop_size: int
                    The desired size for the crop.
        '''
        super(CropSampler, self).__init__()

        self.crop_size = crop_size
        x = torch.arange(0, crop_size, dtype=torch.float32) / (crop_size - 1)
        grid_y, grid_x = torch.meshgrid(x, x)

        points = torch.stack([grid_y.flatten(), grid_x.flatten()], axis=1)

        self.register_buffer('grid', points.unsqueeze(dim=0))

    def extra_repr(self) -> str:
        return f'Crop size: {self.crop_size}'

    def bilinear_sampling(x0, x1, y0, y1):
        pass

    def _sample_packed(self, full_imgs: ImageListPacked, sampling_grid,
                       padding_mode='zeros'):
        device, dtype = sampling_grid.device, sampling_grid.dtype
        batch_size = sampling_grid.shape[0]
        tensor = full_imgs.as_tensor()

        flat_sampling_grid = sampling_grid.reshape(batch_size, -1, 2)
        x, y = flat_sampling_grid[:, :, 0], flat_sampling_grid[:, :, 1]

        # Get the closest spatial locations
        x0 = torch.floor(x).to(dtype=torch.long)
        x1 = x0 + 1

        y0 = torch.floor(y).to(dtype=torch.long)
        y1 = y0 + 1

        # Size: B
        start_idxs = torch.tensor(
            full_imgs.starts, dtype=torch.long, device=device)
        # Size: 3
        rgb_idxs = torch.arange(3, dtype=torch.long, device=device)
        # Size: B
        height_tensor = torch.tensor(
            full_imgs.heights, dtype=torch.long, device=device)
        # Size: B
        width_tensor = torch.tensor(
            full_imgs.widths, dtype=torch.long, device=device)

        # Size: BxP
        x0_in_bounds = x0.ge(0) & x0.le(width_tensor[:, None] - 1)
        x1_in_bounds = x0.ge(0) & x0.le(width_tensor[:, None] - 1)
        y0_in_bounds = y0.ge(0) & y0.le(height_tensor[:, None] - 1)
        y1_in_bounds = y0.ge(0) & y0.le(height_tensor[:, None] - 1)

        zero = torch.tensor(0, dtype=torch.long, device=device)
        x0 = torch.max(
            torch.min(x0, width_tensor[:, None] - 1), zero)
        x1 = torch.max(torch.min(x1, width_tensor[:, None] - 1), zero)
        y0 = torch.max(torch.min(y0, height_tensor[:, None] - 1), zero)
        y1 = torch.max(torch.min(y1, height_tensor[:, None] - 1), zero)

        flat_rgb_idxs = (
            rgb_idxs[None, :, None] * (width_tensor[:, None, None]) *
            height_tensor[:, None, None])
        x0_y0_in_bounds = (x0_in_bounds & y0_in_bounds).unsqueeze(
            dim=1).expand(-1, 3, -1)
        x1_y0_in_bounds = (x1_in_bounds & y0_in_bounds).unsqueeze(
            dim=1).expand(-1, 3, -1)
        x0_y1_in_bounds = (x0_in_bounds & y1_in_bounds).unsqueeze(
            dim=1).expand(-1, 3, -1)
        x1_y1_in_bounds = (x1_in_bounds & y1_in_bounds).unsqueeze(
            dim=1).expand(-1, 3, -1)

        idxs_x0_y0 = (start_idxs[:, None, None] +
                      flat_rgb_idxs +
                      y0[:, None, :] *
                      width_tensor[:, None, None] + x0[:, None, :])
        idxs_x1_y0 = (start_idxs[:, None, None] +
                      flat_rgb_idxs +
                      y0[:, None, :] *
                      width_tensor[:, None, None] + x1[:, None, :])
        idxs_x0_y1 = (start_idxs[:, None, None] +
                      flat_rgb_idxs +
                      y1[:, None, :] * width_tensor[:, None, None] +
                      x0[:, None, :])
        idxs_x1_y1 = (start_idxs[:, None, None] +
                      flat_rgb_idxs +
                      y1[:, None, :] * width_tensor[:, None, None] +
                      x1[:, None, :])

        Ia = torch.zeros(idxs_x0_y0.shape, dtype=dtype, device=device)
        Ia[x0_y0_in_bounds] = tensor[idxs_x0_y0[x0_y0_in_bounds]]

        Ib = torch.zeros(idxs_x1_y0.shape, dtype=dtype, device=device)
        Ib[x1_y0_in_bounds] = tensor[idxs_x1_y0[x1_y0_in_bounds]]

        Ic = torch.zeros(idxs_x0_y1.shape, dtype=dtype, device=device)
        Ic[x0_y1_in_bounds] = tensor[idxs_x0_y1[x0_y1_in_bounds]]

        Id = torch.zeros(idxs_x1_y1.shape, dtype=dtype, device=device)
        Id[x1_y1_in_bounds] = tensor[idxs_x1_y1[x1_y1_in_bounds]]

        f1 = (x1 - x)[:, None] * Ia + (x - x0)[:, None] * Ib
        f2 = (x1 - x)[:, None] * Ic + (x - x0)[:, None] * Id

        output = (y1 - y)[:, None] * f1 + (y - y0)[:, None] * f2
        return output.reshape(batch_size, 3, self.crop_size, self.crop_size)

    def _sample_padded(
        self,
        full_imgs: Union[ImageList, Tensor],
        sampling_grid: Tensor
    ) -> Tensor:
        '''
        '''
        tensor = (
            full_imgs.as_tensor() if isinstance(full_imgs, (ImageList,)) else
            full_imgs
        )
        # Get the sub-images using bilinear interpolation
        return F.grid_sample(tensor, sampling_grid, align_corners=True)

    def forward(
            self,
            full_imgs: Union[Tensor, ImageList, ImageListPacked],
            center: Tensor,
            bbox_size: Tensor
    ) -> Tuple[Tensor, Tensor]:
        ''' Crops the HD images using the provided bounding boxes

            Parameters
            ----------
                full_imgs: ImageList
                    An image list structure with the full resolution images
                center: torch.Tensor
                    A Bx2 tensor that contains the coordinates of the center of
                    the bounding box that will be cropped from the original
                    image
                bbox_size: torch.Tensor
                    A size B tensor that contains the size of the corp

            Returns
            -------
                cropped_images: torch.Tensoror
                    The images cropped from the high resolution input
                sampling_grid: torch.Tensor
                    The grid used to sample the crops
        '''

        batch_size, _, H, W = full_imgs.shape
        transforms = torch.eye(
            3, dtype=full_imgs.dtype, device=full_imgs.device).reshape(
            1, 3, 3).expand(batch_size, -1, -1).contiguous()

        hd_to_crop = torch.eye(
            3, dtype=full_imgs.dtype, device=full_imgs.device).reshape(
            1, 3, 3).expand(batch_size, -1, -1).contiguous()

        # Create the transformation that maps crop pixels to image coordinates,
        # i.e. pixel (0, 0) from the crop_size x crop_size grid gets mapped to
        # the top left of the bounding box, pixel
        # (crop_size - 1, crop_size - 1) to the bottom right corner of the
        # bounding box
        transforms[:, 0, 0] = bbox_size  # / (self.crop_size - 1)
        transforms[:, 1, 1] = bbox_size  # / (self.crop_size - 1)
        transforms[:, 0, 2] = center[:, 0] - bbox_size * 0.5
        transforms[:, 1, 2] = center[:, 1] - bbox_size * 0.5

        hd_to_crop[:, 0, 0] = 2 * (self.crop_size - 1) / bbox_size
        hd_to_crop[:, 1, 1] = 2 * (self.crop_size - 1) / bbox_size
        hd_to_crop[:, 0, 2] = -(
            center[:, 0] - bbox_size * 0.5) * hd_to_crop[:, 0, 0] - 1
        hd_to_crop[:, 1, 2] = -(
            center[:, 1] - bbox_size * 0.5) * hd_to_crop[:, 1, 1] - 1

        size_bbox_sizer = torch.eye(
            3, dtype=full_imgs.dtype, device=full_imgs.device).reshape(
            1, 3, 3).expand(batch_size, -1, -1).contiguous()

        if isinstance(full_imgs, (ImageList, torch.Tensor)):
            # Normalize the coordinates to [-1, 1] for the grid_sample function
            size_bbox_sizer[:, 0, 0] = 2.0 / (W - 1)
            size_bbox_sizer[:, 1, 1] = 2.0 / (H - 1)
            size_bbox_sizer[:, :2, 2] = -1

        #  full_transform = transforms
        full_transform = torch.bmm(size_bbox_sizer, transforms)

        batch_grid = self.grid.expand(batch_size, -1, -1)
        # Convert the grid to image coordinates using the transformations above
        sampling_grid = (torch.bmm(
            full_transform[:, :2, :2],
            batch_grid.transpose(1, 2)) +
            full_transform[:, :2, [2]]).transpose(1, 2)
        sampling_grid = sampling_grid.reshape(
            -1, self.crop_size, self.crop_size, 2).transpose(1, 2)

        if isinstance(full_imgs, (ImageList, torch.Tensor)):
            out_images = self._sample_padded(
                full_imgs, sampling_grid
            )
        elif isinstance(full_imgs, (ImageListPacked, )):
            out_images = self._sample_packed(full_imgs, sampling_grid)
        else:
            raise TypeError(
                f'Crop sampling not supported for type: {type(full_imgs)}')

        return {'images': out_images,
                'sampling_grid': sampling_grid.reshape(batch_size, -1, 2),
                'transform': transforms,
                'hd_to_crop': hd_to_crop,
                }
