from torch.optim import AdamW
import torch

import wandb
import einops as eo

from vae import VAEModel
from conv_encdec import ConvEncoder, ConvDecoder
from utils import make_grids

log = True

def train_step(model, sample, opt):
    opt.zero_grad()
    loss, mse = model(sample)
    loss.backward()
    opt.step()
    return mse

def train(loader, img_size, save = False):
    model = VAEModel(512, ConvEncoder, ConvDecoder, bound_fc = 2048, kl_weight = 0.5,
        img_size = img_size, hidden_layers = 0, channels = 2)
    model.to('cuda')
    opt = AdamW(model.parameters(), lr = 2e-4, betas=(0.5,0.99), weight_decay = 1e-6)

    save_interval = 100 # save every 100 steps 
    best_loss = None # only if better than this 

    if log: 
        wandb.init(project="ConvVAE", entity = "shahbuland")
        wandb.watch(model)

    for step, batch in enumerate(loader):
        x = batch.to('cuda')
        rec_loss = train_step(model, x, opt)

        # Visualizations for logging
        if log:
            with torch.no_grad():
                rec_x = model(x, return_rec = True)
            x_grid, rec_grid = make_grids(x[:16], rec_x[:16])
            x_images = wandb.Image(x_grid.mean(2), caption="Original Images")
            rec_images = wandb.Image(rec_grid.mean(2), caption="Reconstructed Images")

            wandb.log({
                'Reconstruction Loss':rec_loss,
                'Samples':x_images,
                'Rec Samples':rec_images
            })
        
        if step % save_interval == 0:
            if best_loss is None or best_loss > rec_loss.item(): 
                model.save("best_audio.pt")
                best_loss = rec_loss.item()

from data import *

if __name__ == "__main__":
    loader = WAVFolderMemLoader("./data/songs", 512, 5, 16)
    train(loader, 512)