from torch.optim import AdamW
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
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

data_transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(0.5, 0.5)
        ])

if __name__ == "__main__":
    model = VAEModel(64, ConvEncoder, ConvDecoder, bound_fc = 2048, kl_weight = 1.0,
        img_size = 64, hidden_layers = 0)
    model.to('cuda')
    opt = AdamW(model.parameters(), lr = 2e-4, betas=(0.5,0.99), weight_decay = 1e-6)

    BATCH_SIZE = 128
    EPOCHS = 10

    dataset_train = MNIST("./data", train = True, download = True, transform = data_transform)
    train_loader = DataLoader(dataset_train, batch_size = BATCH_SIZE, shuffle = True)

    if log: wandb.init(project="ConvVAE", entity = "shahbuland")
    for epoch in range(EPOCHS):
        for batch in train_loader:
            x, _ = batch
            x = eo.repeat(x, 'n c h w -> n (repeat c) h w', repeat = 3)
            x = x.to('cuda')
            rec_loss = train_step(model, x, opt)

            # Visualizations for logging
            if log:
                with torch.no_grad():
                    rec_x = model(x, return_rec = True)
                x_grid, rec_grid = make_grids(x[:16], rec_x[:16])
                x_images = wandb.Image(x_grid, caption="Original Images")
                rec_images = wandb.Image(rec_grid, caption="Reconstructed Images")

                wandb.log({
                    'Reconstruction Loss':rec_loss,
                    'Samples':x_images,
                    'Rec Samples':rec_images
                })

