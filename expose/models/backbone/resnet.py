from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
from loguru import logger

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from torchvision.models.resnet import (ResNet, Bottleneck, BasicBlock,
                                       model_urls)


class RegressionResNet(ResNet):

    def __init__(self, block, layers, forward_to=4,
                 num_classes=1000,
                 use_avgpool=True,
                 replace_stride_with_dilation=None,
                 zero_init_residual=False, **kwargs):
        super(RegressionResNet, self).__init__(
            block, layers,
            replace_stride_with_dilation=replace_stride_with_dilation)
        self.forward_to = forward_to
        msg = 'Forward to must be from 0 to 4'
        assert self.forward_to > 0 and self.forward_to <= 4, msg

        self.replace_stride_with_dilation = replace_stride_with_dilation

        self.expansion = block.expansion
        self.output_dim = block.expansion * 512
        self.use_avgpool = use_avgpool
        if not use_avgpool:
            del self.avgpool
        del self.fc

    def extra_repr(self):
        if self.replace_stride_with_dilation is None:
            msg = [
                f'Layer 1: {64 * self.expansion}, H / 4, W / 4',
                f'Layer 2: {64 * self.expansion * 2}, H / 8, W / 8',
                f'Layer 3: {64 * self.expansion * 4}, H / 16, W / 16',
                f'Layer 4: {64 * self.expansion * 8}, H / 32, W / 32'
            ]
        else:
            if not any(self.replace_stride_with_dilation):
                msg = [
                    f'Layer 1: {64 * self.expansion}, H / 4, W / 4',
                    f'Layer 2: {64 * self.expansion * 2}, H / 8, W / 8',
                    f'Layer 3: {64 * self.expansion * 4}, H / 16, W / 16',
                    f'Layer 4: {64 * self.expansion * 8}, H / 32, W / 32'
                ]
            else:
                layer2 = 4 * 2 ** (not self.replace_stride_with_dilation[0])
                layer3 = (layer2 *
                          2 ** (not self.replace_stride_with_dilation[1]))
                layer4 = (layer3 *
                          2 ** (not self.replace_stride_with_dilation[2]))
                msg = [
                    f'Layer 1: {64 * self.expansion}, H / 4, W / 4',
                    f'Layer 2: {64 * self.expansion * 2}, H / {layer2}, '
                    f'W / {layer2}',
                    f'Layer 3: {64 * self.expansion * 4}, H / {layer3}, '
                    f'W / {layer3}',
                    f'Layer 4: {64 * self.expansion * 8}, H / {layer4}, '
                    f'W / {layer4}'
                ]

        return '\n'.join(msg)

    def get_output_dim(self):
        return {
            'layer1': 64 * self.expansion,
            'layer2': 64 * self.expansion * 2,
            'layer3': 64 * self.expansion * 4,
            'layer4': 64 * self.expansion * 8,
            'avg_pooling': 64 * self.expansion * 8,
        }

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        output = {'maxpool': x}

        x = self.layer1(x)
        output['layer1'] = x
        x = self.layer2(x)
        output['layer2'] = x
        x = self.layer3(x)
        output['layer3'] = x
        x = self.layer4(x)
        output['layer4'] = x

        # Output size: BxC
        x = self.avgpool(x).view(x.size(0), -1)
        output['avg_pooling'] = x

        return output


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RegressionResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        logger.info('Loading pretrained ResNet-18')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']),
                              strict=False)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RegressionResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        logger.info('Loading pretrained ResNet-34')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']),
                              strict=False)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RegressionResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        logger.info('Loading pretrained ResNet-50')
        missing, unexpected = model.load_state_dict(
            model_zoo.load_url(model_urls['resnet50']), strict=False)
        if len(missing) > 0:
            logger.warning(
                f'The following keys were not found: {missing}')
        if len(unexpected):
            logger.warning(
                f'The following keys were not expected: {unexpected}')
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RegressionResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        logger.info('Loading pretrained ResNet-101')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']),
                              strict=False)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RegressionResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        logger.info('Loading pretrained ResNet-152')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']),
                              strict=False)
    return model


resnets = {'resnet18': resnet18,
           'resnet34': resnet34,
           'resnet50': resnet50,
           'resnet101': resnet101,
           'resnet152': resnet152}
