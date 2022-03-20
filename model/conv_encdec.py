import torch
from torch import nn
import torch.nn.functional as F

import einops as eo

import layers

# Any initializer used in below two funcs must ensure the following properties:
# 1. takes only filters in, filters out
# 2. layer must produce an image that is either twice or half its size (in h,w)

# Automatically create encoding layers from a desired input size
def construct_encoding_layers(img_size, layer_cls):
    size = img_size # Assume every layer halfs img size
    layers = nn.ModuleList()

    # next filter count
    step = lambda filter: filter * 2 if filter < 512 else filter
    f_in = 64

    while size > 4:
        f_out = step(f_in)

        layers.append(layer_cls(f_in, f_out))

        f_in = f_out
        size = size // 2
    
    return layers, f_out

# Same as above but for decoding
def construct_decoding_layers(img_size, layer_cls):
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
        layers.append(layer_cls(f_in, f_out))
    
    f_in = filters[-1][1] # filters going into first block
    return layers, f_in

# Returns a function that is a constructor for one of two classes
# It constructs a res block if fi == fo
# Otherwise it constructs a normal conv block
def res_switch(res_cls, normal_cls):
    return lambda fi, fo: res_cls(fi, fo) if fi == fo else normal_cls(fi, fo)

# StyleGAN type ResNet
class ConvEncoder(nn.Module):
    def __init__(self, out_dim, img_size = 256, hidden_layers = 4, hidden_size = 1024, channels = 3):
        super().__init__()

        # Convolution steps
        self.from_rgb = nn.Conv2d(channels, 64, 4, 2, 1)
        self.prep_shape = lambda x: F.interpolate(x, size=(img_size, img_size))

        switch = res_switch(layers.ResDownConvBlock, layers.DownConvBlock)
        switch = layers.DownConvBlock
        self.conv, f_out = construct_encoding_layers(img_size, switch)

        # FC steps
        self.hidden_layers = nn.Sequential(*
                [nn.Linear(f_out * 4 * 4, hidden_size)] + \
                [nn.Linear(hidden_size, hidden_size) 
                    for _ in range(hidden_layers - 1)]
        ) if hidden_layers > 0 else None

        in_dim = hidden_size if hidden_layers > 0 else f_out * 4 * 4
        self.final_fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.from_rgb(x)
        x = self.prep_shape(x)
        for layer in self.conv:
            x = layer(x)
        
        x = eo.rearrange(x, 'n c h w -> n (c h w)')
        if self.hidden_layers is not None:
            x = self.hidden_layers(x)
        x = self.final_fc(x)
        return x

# StyleGan type ResNet
class ConvDecoder(nn.Module):
    def __init__(self, in_dim, img_size = 256, hidden_layers = 4, hidden_size = 1024, channels = 3):
        super().__init__()

        # Convolution steps
        switch = res_switch(layers.ResUpConvBlock, layers.UpConvBlock)
        switch = layers.UpConvBlock # For some reason residual blocks are just *bad*?
        self.conv, f_in = construct_decoding_layers(img_size, switch)
        
        self.to_rgb = nn.Conv2d(64, channels, 4, 1, 1)
        self.prep_shape = lambda x: F.interpolate(x, size=(img_size, img_size))

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
        y = z.view(-1, 512, 4, 4)
        #z = eo.rearrange(z, 'n (c h w) -> n c h w', h = 4, w = 4)

        for layer in self.conv:
            y = layer(y)

        y = self.to_rgb(y)
        y = self.prep_shape(y)
        return y
