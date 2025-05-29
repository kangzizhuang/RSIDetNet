import torch
import torch.nn as nn
from .act import build_act_layer
import torch.nn.functional as F
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block proposed in SENet (https://arxiv.org/abs/1709.01507)
    We assume the inputs to this layer are (N, C, H, W)
    """

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons,
                              kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels,
                            kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels
        self.nonlinear = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x = self.down(x)
        x = self.nonlinear(x)
        x = self.up(x)
        x = F.sigmoid(x)
        return inputs * x.view(-1, self.input_channels, 1, 1)

class ConvMlpV1(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    copied from timm: https://github.com/huggingface/pytorch-image-models/blob/v0.6.11/timm/models/layers/mlp.py
    """

    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_type='GELU', bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias)
        self.norm = nn.Identity()
        self.act = build_act_layer(act_type)
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class ConvMlpV2(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_type='GELU', bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1,bias=bias)
        self.dwconv = nn.Conv2d(hidden_features,hidden_features,3,1,1,bias=True,groups=hidden_features)
        self.act = build_act_layer(act_type)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1,bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ChannelAggregationFFN(nn.Module):
    """An implementation of FFN with Channel Aggregation.(MogaNet)

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        kernel_size (int): The depth-wise conv kernel size as the
            depth-wise convolution. Defaults to 3.
        act_type (str): The type of activation. Defaults to 'GELU'.
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_type='GELU', bias=True, drop=0.,decompose_act_type='SiLU'):
        super(ChannelAggregationFFN, self).__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.embed_dims = in_features
        self.feedforward_channels = hidden_features

        self.fc1 = nn.Conv2d(
            in_channels=in_features,
            out_channels=self.feedforward_channels,
            kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=self.feedforward_channels)
        self.act = build_act_layer(act_type)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_features,
            out_channels=out_features,
            kernel_size=1, bias=bias)
        self.drop = nn.Dropout(drop)

        self.decompose = nn.Conv2d(
            in_channels=self.feedforward_channels,  # C -> 1
            out_channels=1, kernel_size=1,
        )
        self.scale = nn.Parameter(
            1e-5 * torch.ones((1, hidden_features, 1, 1)),
            requires_grad=True
        )
        self.decompose_act = build_act_layer(decompose_act_type)

    def feat_decompose(self, x):
        # x_d: [B, C, H, W] -> [B, 1, H, W]
        identy = x
        return identy + self.scale * (x - self.decompose_act(self.decompose(x)))

    def forward(self, x):
        # proj 1
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        # proj 2
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvolutionalGLU(nn.Module):
    """from TransNeXt"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_type='GELU', bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(
            in_channels=in_features,
            out_channels=hidden_features,
            kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features//2, hidden_features//2, 3, stride=1,
                                padding=1, bias=bias, groups=hidden_features//2)
        self.act = build_act_layer(act_type)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_features//2,
            out_channels=out_features,
            kernel_size=1, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FFNActGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_type='GELU',bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_1 = nn.Conv2d(
            in_channels=in_features,
            out_channels=hidden_features,
            kernel_size=1, bias=bias)
        self.fc1_2 = nn.Conv2d(
            in_channels=in_features,
            out_channels=hidden_features,
            kernel_size=1, bias=bias)
        self.act = build_act_layer(act_type)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_features,
            out_channels=out_features,
            kernel_size=1, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.act(self.fc1_1(x)) * self.fc1_2(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
