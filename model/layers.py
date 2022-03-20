import torch
from torch import nn
import torch.nn.functional as F

# Bilinear downsampling using F.interpolate
class DownsamplingBilinear2d(nn.Module):
    def __init__(self, down_factor = 2):
        super().__init__()
        self.down_factor = down_factor

    def forward(self, x):
        _, _, h, w = x.shape
        new_h = h // self.down_factor
        new_w = w // self.down_factor
        return F.interpolate(x, size = (new_h, new_w), mode = 'bilinear',
                align_corners = False)

class UpsamplingBilinear2d(nn.Module):
    def __init__(self, up_factor):
        super().__init__()
        self.up_factor = up_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor = 2, mode = 'bilinear',
                align_corners = False)

class ResUpConvBlock(nn.Module):
    def __init__(self, fi, fo):
        super().__init__()

        assert fi == fo

        self.conv1 = nn.Conv2d(fi, fo, 4, 1, 1)
        self.conv2 = nn.Conv2d(fo, fo, 4, 1, 1)
        self.up = UpsamplingBilinear2d(2)
        self.bn = nn.BatchNorm2d(fo)

    def forward(self, x):
        x = self.up(x)
        skip = x
        _, _, h, w = skip.shape

        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn(x)
        x = self.conv2(x)

        # ensure size match for skip connection
        x = F.interpolate(x, size=(h,w))
        x = self.bn(x)
        x = x + skip
        x = F.relu(x)
        x = self.bn(x)
        
        return x

class UpConvBlock(nn.Module):
    def __init__(self, fi, fo):
        super().__init__()

        self.conv = nn.Conv2d(fi, fo, 4, 1, 1)
        self.up = UpsamplingBilinear2d(2)
        self.bn = nn.BatchNorm2d(fo)

    def forward(self, x):
        x = self.up(x)
        #_, _, h, w = x.shape
        x = self.conv(x)
        #x = F.interpolate(x, size=(h,w))
        x = F.relu(x)
        x = self.bn(x)

        return x

class ResDownConvBlock(nn.Module):
    def __init__(self, fi, fo):
        super().__init__()

        self.conv1 = nn.Conv2d(fi, fi, 4, 2, 1)
        self.conv2 = nn.Conv2d(fi, fo, 4, 1, 1)

        self.act = nn.LeakyReLU(0.2)
        self.down = DownsamplingBilinear2d(2)
        self.bn = nn.BatchNorm2d(fo)
    
    def forward(self, x):
        skip = self.down(x)
        _, _, h, w = skip.shape

        x = self.conv1(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.conv2(x)

        # ensure size match
        x = F.interpolate(x, size=(h,w))
        x = x + skip
        x = self.act(x)
        x = self.bn(x)

        return x

class DownConvBlock(nn.Module):
    def __init__(self, fi, fo):
        super().__init__()

        self.conv = nn.Conv2d(fi, fo, 4, 2, 1)

        self.bn = nn.BatchNorm2d(fo)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.conv(x)
        #x = F.interpolate(x, size=(h // 2,w // 2))
        x = self.act(x)
        x = self.bn(x)

        return x