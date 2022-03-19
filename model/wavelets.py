import torch
from torch import nn
from torch.nn import functional as F
from torch_utils.ops.upfirdn2d import upfirdn2d

import layers # Only uses Upsampling, Downsampling, Modblock, DiscConv

# These made StyleGAN train faster, so why not a VAE?

# Haar wavelet transform from SWAGAN
class WaveletTransform(nn.Module):
    def __init__(self, inverse = False):
        super().__init__()
        self.filters = self.get_haar_filters()
        self.inverse = inverse

    # returns [LL, LH, HL, HH] filters
    def get_haar_filters(self):
        l, h = torch.ones(2, 1, 2, device = 'cuda') * (2**-.5)
        h[0, 0] *= -1

        LL = l.T * l
        LH = h.T * l
        HL = l.T * h
        HH = h.T * h

        filters = [LL, LH, HL, HH]
        return torch.cat([f.unsqueeze(0) for f in filters])

    # Normal haar transform
    def baseForward(self, x):
        LL, LH, HL, HH = self.filters

        ll = upfirdn2d(x, LL, down = 2)
        lh = upfirdn2d(x, LH, down = 2)
        hl = upfirdn2d(x, HL, down = 2)
        hh = upfirdn2d(x, HH, down = 2)

        return torch.cat([ll, lh, hl, hh], 1)

    # Inverse haar transform
    def invForward(self, x):
        LL, LH, HL, HH = self.filters
        ll, lh, hl, hh = x.chunk(4, 1)

        p = (1, 0, 1, 0)
        ll = upfirdn2d(ll, LL, up = 2, padding = p)
        lh = upfirdn2d(lh, LH, up = 2, padding = p)
        hl = upfirdn2d(hl, HL, up = 2, padding = p)
        hh = upfirdn2d(hh, HH, up = 2, padding = p)

        return ll + lh + hl + hh

    def forward(self, x):
        if self.inverse:
            return self.invForward(x)
        else:
            return self.baseForward(x)

# SWAGAN version of tRGB layer
class tRGB(nn.Module):
    def __init__(self, fi, channels = 3):
        super().__init__()

        self.upWT = WaveletTransform(inverse = True)
        self.up = layers.UpsamplingBilinear2d(2)
        self.downWT = WaveletTransform(inverse = False)

        self.conv = nn.Conv2d(fi, channels * 4, 1, stride = 2, padding = 0)

    def forward(self, x, y_last = None):
        y = self.conv(x)
        if y_last is not None:
            y += self.downWT(self.up(self.upWT(y_last)))

        return y

# Gen block for SWAGAN
class GenBlock(nn.Module):
    def __init__(self, fi, fo, k = 3):
        super().__init__()
        
        self.conv1 = layers.modBlock(fi, fo, k, mode = "UP")
        self.conv2 = layers.modBlock(fo, fo, k)
        self.to_rgb = tRGB(fo)

    def forward(self, x, y_last = None):
        x = self.conv1(x)
        x = self.conv2(x)
        skip = self.to_rgb(x, y_last)

        return x, skip

# From RGB
class fRGB(nn.Module):
    def __init__(self, fo, channels = 3):
        super().__init__()

        self.upWT = WaveletTransform(inverse = True)
        self.down = layers.DownsamplingBilinear2d(2)
        self.downWT = WaveletTransform(inverse = False)

        self.conv = layers.discConv(channels * 4, fo, 1)

    def forward(self, x, x_last = None):
        if x_last is not None:
            x = self.upWT(x)
            x = self.down(x)
            x = self.downWT(x)

        y = self.conv(x)

        if x_last is not None:
            y += x_last

        return x, y

# Discriminator block for SWAGAN
class DiscBlock(nn.Module):
    def __init__(self, fi, fo):
        super().__init__()
    
        self.from_rgb = fRGB(fi)
        self.conv1 = layers.discConv(fi, fi, 3)
        self.conv2 = layers.discConv(fi, fo, 3, mode = "DOWN")
        
    def forward(self, x, x_last = None):
        skip, y = self.from_rgb(x, x_last)
        y = self.conv1(y)
        y = self.conv2(y)
        return skip, y