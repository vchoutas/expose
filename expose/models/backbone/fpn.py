from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
from loguru import logger

from typing import Dict

import torch
import torch.nn as nn

from torchvision.models import resnet
from torchvision.models.detection.backbone_utils import (
    BackboneWithFPN as _BackboneWithFPN)

FPN_FEATURE_DIM = 256


class BackboneWithFPN(_BackboneWithFPN):
    def __init__(self, *args, **kwargs):
        super(BackboneWithFPN, self).__init__(*args, **kwargs)

    def forward(self, x):
        body_features = getattr(self, 'body')(x)

        output = getattr(self, 'fpn')(body_features)

        for key in body_features:
            output[f'body_{key}'] = body_features[key]
        return output


def resnet_fpn_backbone(backbone_name, pretrained=True, freeze=False):
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained)
    if freeze:
        # freeze layers
        for name, parameter in backbone.named_parameters():
            if ('layer2' not in name and 'layer3' not in name and
                    'layer4' not in name):
                parameter.requires_grad_(False)

    return_layers = {'layer1': 'layer1',
                     'layer2': 'layer2',
                     'layer3': 'layer3',
                     'layer4': 'layer4'}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list,
                           out_channels)


def build_fpn_backbone(backbone_cfg,
                       pretrained=True) -> nn.Module:
    backbone_type = backbone_cfg.get('type', 'resnet50')

    resnet_type = backbone_type.replace('fpn', '').replace('_', '').replace(
        '-', '')
    network = resnet_fpn_backbone(resnet_type, pretrained=pretrained)

    fpn_cfg = backbone_cfg.get('fpn', {})

    return RegressionFPN(network, fpn_cfg)


class SumAvgPooling(nn.Module):
    def __init__(self, pooling_type='avg', **kwargs) -> None:
        super(SumAvgPooling, self).__init__()

        if pooling_type == 'avg':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pooling_type == 'max':
            self.pooling = nn.AdaptiveMaxPool2d(1)
        else:
            raise ValueError(f'Unknown pooling function: {pooling_type}')

    def get_out_feature_dim(self) -> int:
        return FPN_FEATURE_DIM

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:

        pooled_features = {}
        # Pool each feature map
        for key in features:
            batch_size, feat_dim = features[key].shape[:2]
            pooled_features[key] = self.pooling(features[key]).view(
                batch_size, feat_dim)

        # Sum the individual features
        return sum(pooled_features.values())


class ConcatPooling(nn.Module):
    def __init__(self, use_max: bool = True, use_avg: bool = True,
                 **kwargs) -> None:
        super(ConcatPooling, self).__init__()
        assert use_avg or use_max, 'Either max or avg pooling should be on'

        self.use_avg = use_avg
        self.use_max = use_max
        if use_avg:
            self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        if use_max:
            self.max_pooling = nn.AdaptiveMaxPool2d(1)

    def extra_repr(self) -> str:
        msg = [f'Use average pooling: {self.use_avg}',
               f'Use max pooling: {self.use_max}']
        return '\n'.join(msg)

    def get_out_feature_dim(self) -> int:
        return 5 * (
            self.use_avg * FPN_FEATURE_DIM + self.use_max * FPN_FEATURE_DIM)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        pooled_features = []
        for key in features:
            batch_size, feat_dim = features[key].shape[:2]
            feats = []
            if self.use_avg:
                avg_pooled_features = self.avg_pooling(features[key]).view(
                    batch_size, feat_dim)
                feats.append(avg_pooled_features)
            if self.use_max:
                max_pooled_features = self.max_pooling(features[key]).view(
                    batch_size, feat_dim)
                feats.append(max_pooled_features)
            pooled_features.append(
                torch.cat(feats, dim=-1))
        return torch.cat(pooled_features, dim=-1)


class BilinearPooling(nn.Module):
    def __init__(self, pooling_type='avg', **kwargs) -> None:
        super(BilinearPooling, self).__init__()
        raise NotImplementedError
        if pooling_type == 'avg':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pooling_type == 'max':
            self.pooling = nn.AdaptiveMaxPool2d(1)
        else:
            raise ValueError(f'Unknown pooling function: {pooling_type}')

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        pooled_features = {}
        # Pool each feature map
        for key in features:
            batch_size, feat_dim = features[key].shape[:2]
            pooled_features[key] = self.pooling(features[key]).view(
                batch_size, feat_dim)
        # Should be BxNxK
        stacked_features = torch.stack(pooled_features.values(), dim=1)
        pass


#  class RegressionFPN(nn.Module):
class RegressionFPN(nn.Module):

    def __init__(self, backbone, fpn_cfg) -> None:
        super(RegressionFPN, self).__init__()
        self.feat_extractor = backbone

        pooling_type = fpn_cfg.get('pooling_type', 'sum_avg')
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        if pooling_type == 'sum_avg':
            sum_avg_cfg = fpn_cfg.get('sum_avg', {})
            self.pooling = SumAvgPooling(**sum_avg_cfg)
        elif pooling_type == 'concat':
            concat_cfg = fpn_cfg.get('concat', {})
            self.pooling = ConcatPooling(**concat_cfg)
        elif pooling_type == 'none':
            self.pooling = None
        else:
            raise ValueError(f'Unknown pooling type {pooling_type}')

    def get_output_dim(self) -> int:
        output = {
            'layer1': FPN_FEATURE_DIM,
            'layer2': FPN_FEATURE_DIM,
            'layer3': FPN_FEATURE_DIM,
            'layer4': FPN_FEATURE_DIM,
        }

        for key in output:
            output[f'{key}_avg_pooling'] = FPN_FEATURE_DIM
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feat_extractor(x)

        if self.pooling is not None:
            pass
        features['avg_pooling'] = self.avg_pooling(features['body_layer4'])
        return features
