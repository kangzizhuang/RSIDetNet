

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.helpers import checkpoint_seq
from timm.models.layers import trunc_normal_, DropPath
from mmcv.cnn.bricks import DropPath
from mmrotate.models.builder import ROTATED_BACKBONES
from mmengine.model import BaseModule


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class DecoupeLKConv(nn.Module):
    def __init__(self, dim, d_ks=(5, 3), d_dilation=(2, 3), band_ks=11, e=0.25):
        super().__init__()

        hidden_channel = int(dim * e)
        self.hd = hidden_channel

        self.split_indexes = (dim - 3 * hidden_channel, hidden_channel, hidden_channel, hidden_channel)
        self.band_w_conv = nn.Conv2d(hidden_channel, hidden_channel, (1, band_ks),
                                     padding=autopad((1, band_ks)), )
        self.band_h_conv = nn.Conv2d(hidden_channel, hidden_channel, (band_ks, 1),
                                     padding=autopad((band_ks, 1)), )
        self.band_1x1 = nn.Conv2d(hidden_channel, hidden_channel, 1, 1)

        self.d0_conv = nn.Conv2d(hidden_channel, hidden_channel, d_ks[0], padding=autopad(d_ks[0], d=d_dilation[0]),
                                 dilation=d_dilation[0], groups=hidden_channel)
        self.d1_conv = nn.Conv2d(hidden_channel, hidden_channel, d_ks[1], padding=autopad(d_ks[1], d=d_dilation[1]),
                                 groups=hidden_channel,
                                 dilation=d_dilation[1])
        self.fusion = nn.Conv2d(dim - 3 * hidden_channel, dim - 3 * hidden_channel, 1, 1)

    def forward(self, x):
        x_id, x_lk1, x_lk2, x_lk3 = torch.split(x, self.split_indexes, dim=1)
        x_lk1 = self.band_1x1(self.band_h_conv(self.band_w_conv(x_lk1)))
        x_lk2 = self.d0_conv(x_lk2)
        x_lk3 = self.d1_conv(x_lk3)
        x = torch.cat([self.fusion(x_id), x_lk1, x_lk2, x_lk3], dim=1)
        return x



class DecoupeLKInception(nn.Module):
    def __init__(self, in_channels, band_kernel_size=11,):
        super().__init__()
        if band_kernel_size == 11:
            self.d_kernel_size = (5, 3)
            self.d_dilation = (2, 3)
        self.conv = DecoupeLKConv(in_channels, self.d_kernel_size, self.d_dilation, band_kernel_size)

    def forward(self, x):
        return self.conv(x)


class SpliteInception(nn.Module):
    def __init__(self, dim, default_kernel_size=3):
        super().__init__()
        self.sk_conv = nn.Conv2d(dim, dim, default_kernel_size, padding=default_kernel_size // 2, groups=dim)

    def forward(self, x):
        return self.sk_conv(x)


class ConvMlp(nn.Module):

    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=True)
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class MetaNeXtBlock(nn.Module):
    """ MetaNeXtBlock Block
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            dim,
            token_mixer,
            norm_layer=nn.BatchNorm2d,
            mlp_layer=ConvMlp,
            mlp_ratio=4,
            mlp_drop=0.1,
            act_layer=nn.GELU,
            ls_init_value=1e-6,
            drop_path=0.,
    ):
        super().__init__()
        self.token_mixer = token_mixer(dim)
        self.norm = norm_layer(dim)
        # self.se = SEBlock(dim, dim // 4)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer, drop=mlp_drop)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        # x = self.se(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x


class MetaNeXtStage(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            token_mixer,
            ds_stride=2,
            depth=2,
            drop_path_rates=None,
            ls_init_value=1.0,
            act_layer=nn.GELU,
            norm_layer=None,
            mlp_ratio=4,
    ):
        super().__init__()
        self.grad_checkpointing = False
        if ds_stride > 1:
            self.downsample = nn.Sequential(
                norm_layer(in_chs),
                nn.Conv2d(in_chs, out_chs, kernel_size=ds_stride, stride=ds_stride),
            )
        else:
            self.downsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(MetaNeXtBlock(
                dim=out_chs,
                token_mixer=token_mixer,
                drop_path=drop_path_rates[i],
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                mlp_ratio=mlp_ratio,
            ))
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


@ROTATED_BACKBONES.register_module()
class RSInceptionNet(BaseModule):

    def __init__(
            self,
            in_chans=3,
            num_classes=9,
            depths=(3, 3, 9, 3),
            dims=(96, 192, 384, 768),
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.GELU,
            mlp_ratios=(4, 4, 4, 3),
            drop_rate=0.,
            drop_path_rate=0.1,
            ls_init_value=1e-6,
            out_indices=(0, 1, 2, 3),
            init_cfg=None,
            **kwargs,
    ):
        super().__init__()
        self.init_cfg = init_cfg
        num_stage = len(depths)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0] // 2, kernel_size=3, stride=2, padding=1),
            norm_layer(dims[0] // 2),
            nn.GELU(),
            nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
            norm_layer(dims[0])
        )

        self.out_indices = out_indices
        for i_emb, i_layer in enumerate(self.out_indices):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        self.stages = nn.Sequential()
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages = []
        prev_chs = dims[0]
        # feature resolution stages, each consisting of multiple residual blocks
        for i in range(num_stage):
            out_chs = dims[i]
            stages.append(MetaNeXtStage(
                prev_chs,
                out_chs,
                token_mixer=SpliteInception if i < 2 else DecoupeLKInception,
                ds_stride=2 if i > 0 else 1,
                depth=depths[i],
                drop_path_rates=dp_rates[i],
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                mlp_ratio=mlp_ratios[i],
            ))
            prev_chs = out_chs
        self.stages = nn.Sequential(*stages)
        self.num_features = prev_chs
        # self.apply(self._init_weights)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'norm'}

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return x

    def forward(self, x):
        return self.forward_det(x)

    def forward_det(self, x):
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        return tuple(outs)

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        else:
            super(RSInceptionNet, self).init_weights()


