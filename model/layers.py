import torch
from torch import nn
import torch.nn.functional as F

from torch_utils.ops import upfirdn2d, conv2d_gradfix

# A lot of this code is copied from a private repo I have for StyleGAN 2
# So a lot of things might be unexplained/uncommented
# Names for layers might be confusing
# too lazy to rewrite code :(

# Normalize FIR kernel
def norm_fir_k(k):
    k = k[None, :] * k[:, None]
    k = k / k.sum()
    return k

# Simplified wrapper for upfirdn2d
# for pooling and upsampling
class upFirDn2D(nn.Module):
    # kernel size of associated convolution
    # and factor by which to upsample (down if < 1)
    # (should be power of 2)
    def __init__(self, conv_k, scale = 1):
        super().__init__()
        
        # FIR filter used in nvidia implementation
        self.f = torch.Tensor([1, 3, 3, 1]).cuda()
        self.f = norm_fir_k(self.f)
        # Get needed padding for desired scale
        if scale == 0.5: # Downsample
            p = 1 + conv_k
            padX = (p + 1) // 2
            padY = p // 2
        elif scale == 1: # Same
            p = 3 - conv_k
            padX = (p + 1) // 2
            padY = p // 2
        elif scale > 1: # upsample
            p = 3 - conv_k
            padX = (p + 1) // 2 + 1
            padY = p // 2 + 1
            self.f *= (scale ** 2)
        
        self.p = (padX, padY)

    def forward(self, x):
        return upfirdn2d.upfirdn2d(x, self.f, padding = self.p)

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

# Discriminator can be used as an encoder

# Conv layers for the discriminator
class discConv(nn.Module):
    # Mode can be "DOWN" or None
    def __init__(self, fi, fo, k, mode = None, use_bias = True, use_act = True):
        super().__init__()

        self.w = nn.Parameter(torch.randn(fo, fi, k, k))
        self.w_scale = (fi * k * k)**-.5
        self.b = nn.Parameter(torch.zeros(fo)) if use_bias else None
        self.act = nn.LeakyReLU(0.2) if use_act else None
        self.use_act = use_act

        self.s = 2 if mode == "DOWN" else 1
        self.p = 0 if mode == "DOWN" else k // 2

        self.fir = None
        if mode == "DOWN": self.fir = upFirDn2D(k, 0.5)

    def forward(self, x):
        if self.fir is not None:
            x = self.fir(x)
        x = conv2d_gradfix.conv2d(x, self.w * self.w_scale, bias = self.b, stride = self.s, padding = self.p)
        if self.use_act: x = self.act(x) 
        return x

# Generator can be used as a decoder

# Layer that adds noise with learnable weight
class Noise(nn.Module):
    def __init__(self):
        super().__init__()

        # "Learned per channel scaling factor" for noise
        self.w = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        n, _, h, w = x.shape
        noise = torch.randn(n, 1, h, w, device = 'cuda')

        return x + w * noise

# Modulated convolution
class modConv(nn.Module):
    # Mode can be "UP", "DOWN" or None
    def __init__(self, fi, fo, k, mode = None, do_demod = True, eps = 1e-6):
        super().__init__()

        self.fi = fi # Filters/channels in
        self.fo = fo # Filters/channels out
        self.k = k # Kernel size
        self.pad = k // 2
        self.mode = mode 
        self.do_demod = do_demod
        self.eps = eps # for weight demod

        self.w = nn.Parameter(torch.randn(1, fo, fi, k, k))# Weight matrix
        self.w_scale = (fi * k * k)**-.5
       
        self.fir = upFirDn2D(k, scale = 2 if mode == "UP" else 1)
    # takes input and style
    def forward(self, x):
        batch, channels, in_h, in_w = x.shape
        assert channels == self.fi

        # Make up for removing style code by
        # adding in tensor that is same shape as previous style tensor
        scale = self.w_scale * torch.ones(batch, 1, self.fi, 1, 1, device = 'cuda')

        w = self.w * scale

        # Demodulate
        if self.do_demod:
            # In the paper the sum is over indices "i" and "k"
            # Representing out channel (index 2 in self.w)
            # and kernel size, which is two indices (3 and 4)
            demod_mult = torch.rsqrt(w.pow(2).sum([2,3,4]) + self.eps).view(batch, self.fo, 1, 1, 1)
            w = w * demod_mult

        w = w.view(batch * self.fo, self.fi, self.k, self.k)

        x = x.view(1, batch * self.fi, in_h, in_w)
        if self.mode == "DOWN":
            x = self.fir(x)
            y = conv2d_gradfix.conv2d(x, w, padding = 0, stride = 2, groups = batch)
        elif self.mode == "UP":
            # For transposed convolution need filters out and in
            # swapped
            w = w.view(batch, self.fo, self.fi, self.k, self.k)
            w = w.transpose(1, 2)
            w = w.reshape(batch * self.fi, self.fo, self.k, self.k)
            y = conv2d_gradfix.conv_transpose2d(x, w, padding = 0, stride = 2, groups = batch)
        else:
            y = conv2d_gradfix.conv2d(x, w, padding = self.pad, groups = batch)

        _, _, out_h, out_w = y.shape
        y = y.view(batch, self.fo, out_h, out_w)
        
        if self.mode == "UP":
            y = self.fir(y)

        return y
        
# Single block in the generator
# i.e. modulated conv, noise, style, all put together
class modBlock(nn.Module):
    # As before mode can be "UP" "DOWN or None
    def __init__(self, fi, fo, k, mode = None, do_demod = True):
        super().__init__()
        
        self.conv = modConv(fi, fo, k, mode, do_demod)
        self.bias = nn.Parameter(torch.zeros(1, fo, 1, 1))
        self.noise = Noise()
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.noise(x)
        x = x + self.bias
        x = self.act(x)

        return x