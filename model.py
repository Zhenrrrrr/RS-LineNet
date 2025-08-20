import numpy as np
from typing import Optional
import torch.nn as nn
import torch

class ConvModule(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            groups: int = 1,
            norm_cfg: Optional[dict] = None,
            act_cfg: Optional[dict] = None):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=(norm_cfg is None)))
        if norm_cfg:
            norm_layer = self._get_norm_layer(out_channels, norm_cfg)
            layers.append(norm_layer)
        if act_cfg:
            act_layer = self._get_act_layer(act_cfg)
            layers.append(act_layer)
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

    def _get_norm_layer(self, num_features, norm_cfg):
        if norm_cfg['type'] == 'BN':
            return nn.BatchNorm2d(num_features, momentum=norm_cfg.get('momentum', 0.1), eps=norm_cfg.get('eps', 1e-5))
        raise NotImplementedError(f"Normalization layer '{norm_cfg['type']}' is not implemented.")

    def _get_act_layer(self, act_cfg):
        # 根据配置获取激活层
        if act_cfg['type'] == 'ReLU':
            return nn.ReLU(inplace=True)
        if act_cfg['type'] == 'SiLU':
            return nn.SiLU(inplace=True)
        raise NotImplementedError(f"Activation layer '{act_cfg['type']}' is not implemented.")

class CAA(nn.Module):
    def __init__(
            self,
            channels: int,
            h_kernel_size: int = 11,
            v_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU')):
        super().__init__()

        self.avg_pool = nn.AvgPool2d(7, 1, 3)

        self.conv1 = ConvModule(channels, channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1, (0, h_kernel_size // 2), groups=channels, norm_cfg=None, act_cfg=None)

        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), groups=channels, norm_cfg=None, act_cfg=None)

        self.conv2 = ConvModule(channels, channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor

class CS_CAA(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.conv_3x3 = Conv(dim, dim // 2, 3)
        self.conv_3x3_d1 = Conv(dim // 2, dim, 3, d=1)
        self.conv_3x3_d3 = Conv(dim // 2, dim // 2, 3, d=3)
        self.conv_3x3_d5 = Conv(dim // 2, dim // 2, 3, d=5)
        self.conv_1x1 = Conv(dim * 2, dim, k=1)
        self.attention = CAA(dim // 2)  # 添加注意力机制，通道数应与 conv_3x3 的输出一致

    def channel_shuffle(self, x, groups):
        batch_size, num_channels, height, width = x.size()
        # 计算每组的通道数
        channels_per_group = num_channels // groups
        # 重新排列通道
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # 交换通道和组的维度
        x = x.view(batch_size, num_channels, height, width)  # 恢复原始形状
        return x

    def forward(self, x):
        conv_3x3 = self.conv_3x3(x)
        conv_3x3 = self.channel_shuffle(conv_3x3, groups=2)  # 进行通道洗牌操作

        # 应用注意力机制
        conv_3x3 = self.attention(conv_3x3)

        x1 = self.conv_3x3_d1(conv_3x3)
        x2 = self.conv_3x3_d3(conv_3x3)
        x3 = self.conv_3x3_d5(conv_3x3)

        # 合并三个卷积结果
        x_out = torch.cat([x1, x2, x3], dim=1)
        x_out = self.conv_1x1(x_out) + x  # 假设 x 的形状与 x_out 兼容
        return x_out

class C2f_SCAA(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CS_CAA(self.c) for _ in range(n))
