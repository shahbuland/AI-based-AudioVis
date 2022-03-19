import torch
from torch import nn
import torch.nn.functional as F

import einops as eo

import wavelets
import layers

# Automatically create encoding layers from a desired input size
def construct_encoding_layers(img_size):
    size = img_size // 2 # Assume every layer halfs img size
    layers = nn.ModuleList()

    # next filter count
    step = lambda filter: filter * 2 if filter < 512 else filter
    f_in = 64

    while size > 4:
        f_out = step(f_in)

        layers.append(wavelets.DiscBlock(f_in, f_out))

        f_in = f_out
        size = size // 2
    
    return layers, f_out

# Same as above but for decoding
def construct_decoding_layers(img_size):
    # can't do same as before, make list of filters now
    step = lambda filter: filter * 2 if filter < 512 else filter
    f_in = 64
    filters = []
    size = img_size

    while size > 4:
        f_out = step(f_in)
        filters.append((f_out, f_in)) # reverse order
        f_in = f_out
        size = size // 2
    
    layers = nn.ModuleList()
    for (f_in, f_out) in reversed(filters):
        layers.append(wavelets.GenBlock(f_in, f_out))
    
    f_in = filters[-1][1] # filters going into first block
    return layers, f_in

# StyleGAN type ResNet
class ConvEncoder(nn.Module):
    def __init__(self, out_dim, img_size = 256):
        super().__init__()

        hidden_size = 1024
        hidden_layers = 4

        # Convolution steps
        self.downWT = wavelets.WaveletTransform(inverse = False)

        self.conv, f_out = construct_encoding_layers(img_size)
        self.end_frgb = wavelets.fRGB(f_out)

        # FC steps
        self.hidden_layers = nn.Sequential(*
                [nn.Linear(f_out * 4 * 4, hidden_size)] + \
                [nn.Linear(hidden_size, hidden_size) 
                    for _ in range(hidden_layers - 1)]
        ) if hidden_layers > 0 else None

        in_dim = hidden_size if hidden_layers > 0 else f_out * 4 * 4
        self.final_fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.downWT(x)

        skip = None
        for layer in self.conv:
            x, skip = layer(x, skip)
        _, x = self.end_frgb(x, skip)
        
        x = eo.rearrange(x, 'n c h w -> n (c h w)')
        if self.hidden_layers is not None:
            x = self.hidden_layers(x)
        x = self.final_fc(x)
        return x

# StyleGan type ResNet
class ConvDecoder(nn.Module):
    def __init__(self, in_dim, img_size = 256):
        super().__init__()

        hidden_size = 1024
        hidden_layers = 4

        # Convolution steps
        self.conv, f_in = construct_decoding_layers(img_size)

        self.first_conv = layers.modBlock(f_in, f_in, 3, mode = None)
        self.start_trgb = wavelets.tRGB(f_in)

        self.upWT = wavelets.WaveletTransform(inverse = True)

        # FC steps
        out_dim = hidden_size if hidden_layers > 0 else f_in * 4 * 4
        self.first_fc = nn.Linear(in_dim, out_dim)
        self.hidden_layers = nn.Sequential(*
            [nn.Linear(hidden_size, hidden_size)
                for _ in range(hidden_layers - 1)] + \
            [nn.Linear(hidden_size, f_in * 4 * 4)]
        ) if hidden_layers > 0 else None

    def forward(self, z):
        z = self.first_fc(z)
        if self.hidden_layers is not None:
            z = self.hidden_layers(z)
        z = eo.rearrange(z, 'n (c h w) -> n c h w', h = 4, w = 4)

        y = self.first_conv(z)
        skip = self.start_trgb(y)

        for layer in self.conv:
            y, skip = layer(y, skip)
        y = skip

        y = self.upWT(y)
        return y
