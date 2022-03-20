import torch
import torchvision

import einops as eo

# For visualizations
def make_grids(x, rec_x):
    # both assumed float (n, c, h, w)
    # assumed n is perfect square (if not, we will round down)
    
    n, c, h, w = x.shape 

    n_sqrt = int(n ** .5)
    n = n_sqrt * n_sqrt
    x = x[:n]
    rec_x = rec_x[:n]

    x_grid = torchvision.utils.make_grid(x, n_sqrt)
    rec_grid = torchvision.utils.make_grid(rec_x, n_sqrt)

    # renormalize both to be in [0,1]
    x_grid -= x_grid.min() # [0, ?]
    rec_grid -= rec_grid.min() # [0, ?]
    x_grid /= x_grid.max() # [0, 1]
    rec_grid /= rec_grid.max() # [0, 1]

    # reorder channels
    x_grid = eo.rearrange(x_grid, 'c h w -> h w c')
    rec_grid = eo.rearrange(rec_grid, 'c h w -> h w c')

    return x_grid.detach().cpu().numpy(), rec_grid.detach().cpu().numpy() 

# normalize to [0,1]
def normalize(x):
    x -= x.min()
    x /= x.max()
    return x