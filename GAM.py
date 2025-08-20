# gam核心代码
import torch
import torch.nn as nn

'''
https://arxiv.org/abs/2112.05561
'''
__all__ = (
    "GAM",
)


class GAM(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4):
        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // rate),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // rate, in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // rate, kernel_size=7, padding=3),
            nn.BatchNorm2d(in_channels // rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // rate, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels)
        )

        # 自适应卷积核大小的层
        self.dynamic_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, skip_connection=None):
        b, c, h, w = x.shape

        # 通道注意力
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        # 通道注意力加权
        x = x * x_channel_att

        # 空间注意力
        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        # 引入跳跃连接
        if skip_connection is not None:
            out = out + skip_connection

        # 自适应卷积
        out = self.dynamic_conv(out)

        return out


if __name__ == '__main__':
    img = torch.rand(1, 64, 32, 48)
    b, c, h, w = img.shape
    net = GAM(in_channels=c, out_channels=c)
    output = net(img)
    print(output.shape)