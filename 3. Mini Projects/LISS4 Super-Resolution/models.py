import math
import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, scale_factor, in_channels =3, maskOutput=False):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.maskOutput = maskOutput
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(32)
        self.block3 = ResidualBlock(32+32)
        self.block4 = ResidualBlock(32+32+32)
        self.block5 = ResidualBlock(32+32+32+32)
        self.block6 = ResidualBlock(32+32+32+32+32)
        self.block7 = nn.Sequential(
            nn.Conv2d(32+32+32+32+32+32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32)
        )
        if(self.maskOutput):
            self.maskLayers = nn.Sequential(
                nn.Conv2d(32, 1, kernel_size=9, padding=4),
                nn.Sigmoid()
            )
        block8 = [UpsampleBLock(32, 2) for _ in range(upsample_block_num)]
        self.block8 = nn.Sequential(*block8)
        self.final_layer = nn.Conv2d(32, 3, kernel_size=9, padding=4)

    def forward(self, x, return_dense = False):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(torch.cat((block1, block2), dim=1))
        block4 = self.block4(torch.cat((block1, block2, block3), dim=1))
        block5 = self.block5(torch.cat((block1, block2, block3, block4), dim=1))
        block6 = self.block6(torch.cat((block1, block2, block3, block4, block5), dim=1))
        block7 = self.block7(torch.cat((block1, block2, block3, block4, block5, block6), dim=1))
        block8 = self.block8(block1 + block7)
        out = self.final_layer(block8)
        if(self.maskOutput):
            mask = self.maskLayers(block8)
            if(return_dense):
                return (torch.tanh(out) + 1) / 2, block8, mask
            return (torch.tanh(out) + 1) / 2, mask
        
        if(return_dense):
            return (torch.tanh(out) + 1) / 2, block8
        return (torch.tanh(out) + 1) / 2







class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.finalConv = nn.Conv2d(channels, 32, kernel_size=1, padding=0)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = x + residual
        
        return self.finalConv(residual)


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
