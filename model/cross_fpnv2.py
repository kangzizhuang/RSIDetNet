# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16

from mmrotate.models.builder import ROTATED_NECKS
from mmrotate.models.my_module.build import build_conv
from mmrotate.models.my_module.model.channel_mixer import *


def build_channel_mixer(mixer_type, in_features, hidden_features=None, out_features=None, act_type='GELU', bias=True,
                        drop=0.):
    if mixer_type is None:
        return nn.Identity()
    assert mixer_type in ['SE', 'TE1', 'TE2', 'TE3', 'TE4', 'ConvMlpV1', 'ConvMlpV2', 'ChannelAggregationFFN',
                          'ConvolutionalGLU', 'FFNActGLU']
    if mixer_type == 'ConvMlpV1':
        return ConvMlpV1(in_features=in_features, hidden_features=hidden_features, out_features=out_features,
                         act_type=act_type, bias=bias, drop=drop)
    elif mixer_type == 'ConvMlpV2':
        return ConvMlpV2(in_features=in_features, hidden_features=hidden_features, out_features=out_features,
                         act_type=act_type, bias=bias, drop=drop)
    elif mixer_type == 'ChannelAggregationFFN':
        return ChannelAggregationFFN(in_features=in_features, hidden_features=hidden_features,
                                     out_features=out_features, act_type=act_type, bias=bias, drop=drop)
    elif mixer_type == 'ConvolutionalGLU':
        return ConvolutionalGLU(in_features=in_features, hidden_features=hidden_features, out_features=out_features,
                                act_type=act_type, bias=bias, drop=drop)
    elif mixer_type == 'FFNActGLU':
        return FFNActGLU(in_features=in_features, hidden_features=hidden_features, out_features=out_features,
                         act_type=act_type, bias=bias, drop=drop)
    elif mixer_type == 'SE':
        return nn.Sequential(SEBlock(input_channels=in_features, internal_neurons=in_features // 4),
                             nn.Conv2d(in_features, out_features, 1, 1))
    elif mixer_type == 'TE1':
        return TextureExtracter1(in_channels=in_features, out_channels=out_features)
    elif mixer_type == 'TE2':
        return TextureExtracter2(in_channels=in_features, out_channels=out_features)
    elif mixer_type == 'TE3':
        return TextureExtracter3(in_channels=in_features, out_channels=out_features)
    elif mixer_type == 'TE4':
        return TextureExtracter4(in_channels=in_features, out_channels=out_features)


class TextureExtracter1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TextureExtracter1, self).__init__()
        self.conv1 = build_conv(in_channels, in_channels, 'DWC', 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = build_conv(in_channels, out_channels, 'NC', 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        return x


class TextureExtracter2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TextureExtracter2, self).__init__()
        hidden_channels = in_channels // 4
        self.down = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels,
                              kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels,
                            kernel_size=1, stride=1, bias=True)
        self.dw = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1,
                            padding=1, groups=hidden_channels)

    def forward(self, x):
        x_id = x
        x = self.down(x)
        x = self.dw(x)
        x = self.up(x)
        return x + x_id


class TextureExtracter3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TextureExtracter3, self).__init__()
        hidden_channels = in_channels // 4
        self.down = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels,
                              kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels,
                            kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        x_id = x
        x = self.down(x)
        x = self.up(x)
        return x + x_id


class TextureExtracter4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TextureExtracter4, self).__init__()
        self.dw1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1,
                             groups=in_channels)
        self.dw2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                             groups=out_channels)

    def forward(self, x):
        x_id = x
        x = self.dw1(x)
        x = self.dw2(x)
        return x + x_id


class CrossFusion(nn.Module):
    def __init__(self, dim, mixer_type, hidden_ratio=1, upsample_type='Transpose2d', use_norm=True):
        super(CrossFusion, self).__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        hidden_channels = int(dim * hidden_ratio)
        self.fe1 = build_channel_mixer(in_features=dim, hidden_features=hidden_channels, out_features=dim,
                                       mixer_type=mixer_type)
        self.fe2 = build_channel_mixer(in_features=dim, hidden_features=hidden_channels, out_features=dim,
                                       mixer_type=mixer_type)
        if upsample_type == 'Transpose2d':
            self.up_conv = nn.ConvTranspose2d(dim, dim, 3, 2, padding=1, output_padding=1, groups=dim)
        else:
            self.up_conv = nn.Upsample(scale_factor=2, mode=upsample_type)
        self.down = nn.Conv2d(in_channels=2 * dim, out_channels=dim,
                              kernel_size=1, stride=1, bias=True)
        self.res_layer_scale = nn.Parameter(1e-6 * torch.ones((1, dim, 1, 1)), requires_grad=True)
        self.use_norm = use_norm

    def forward(self, x1, x2):
        x2 = self.up_conv(x2)
        if self.use_norm:
            x2 = self.norm2(x2)
        x2 = self.fe2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.down(x)
        if self.use_norm:
            x = self.norm1(x)
        x = self.fe1(x)
        return x + self.res_layer_scale * x1


@ROTATED_NECKS.register_module()
class CrossEnhanceFpnv2(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 mixer_type='TE',
                 upsample_type='Transpose2d',
                 use_norm=True,
                 hidden_ratio=1):
        super(CrossEnhanceFpnv2, self).__init__()
        self.in_channels = in_channels
        self.num_outs = num_outs

        self.act_convs = nn.ModuleList()
        self.lateral_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            l_conv = build_conv(in_channels[i], out_channels, "NC", 1, 1)
            act_conv = build_conv(out_channels, out_channels, "NC", 3, 1)
            self.lateral_convs.append(l_conv)
            self.act_convs.append(act_conv)

        self.fusion_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.fusion_blocks.append(
                CrossFusion(out_channels, mixer_type=mixer_type, hidden_ratio=hidden_ratio, upsample_type=upsample_type,
                            use_norm=use_norm))

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        out = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        for i in range(len(out) - 1, 0, -1):
            out[i - 1] = self.fusion_blocks[i - 1](out[i - 1], out[i])

        for i in range(0, len(out)):
            out[i] = self.act_convs[i](out[i])

        if self.num_outs > len(out):
            for _ in range(self.num_outs - len(out)):
                out.append(F.max_pool2d(out[-1], 1, stride=2))

        return out
