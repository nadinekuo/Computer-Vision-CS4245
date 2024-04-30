import torch
import torch.nn as nn
import torch.nn.functional as F
from network_parts import DoubleConv, PoolAndConv, Up


class UNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = DoubleConv(3, 64)
        self.down_1 = PoolAndConv(in_channels=64, out_channels=128)
        self.down_2 = PoolAndConv(in_channels=128, out_channels=256)
        self.down_3 = PoolAndConv(in_channels=256, out_channels=512)

        self.up_1 = Up(in_channels=(512 + 256), out_channels=256)
        self.up_2 = Up(in_channels=(256 + 128), out_channels=128)
        self.up_3 = Up(in_channels=(128 + 64), out_channels=64)
        self.out_conv = nn.Conv2d(64, 2, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)

        x5 = self.up_1(x4, x3)
        x6 = self.up_2(x5, x2)
        x7 = self.up_3(x6, x1)
        x8 = self.out_conv(x7)

        return x8
