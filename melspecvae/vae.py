import torch
from torch import nn
import torch.nn.functional as F
import transformers

import einops as eo

# ===== LAYERS =====

class ConvBlock(nn.Module):
    def __init__(self, fi, fo):
        super().__init__()

        self.conv = nn.Conv2d(fi, fo, 3, padding = "same")
        self.pool = nn.MaxPool2d(2)
        self.bn = nn.BatchNorm2d(fo)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.bn(x)
        x = self.pool(x)

        return x

class DeconvBlock(nn.Module):
    def __init__(self, fi, fo):
        super().__init__()

        self.up = nn.Upsample(scale_factor = 2)
        self.deconv = nn.Conv2d(fi, fo, 3, padding = "same")
        self.bn = nn.BatchNorm2d(fo)

    def forward(self, x):
        x = self.up(x)
        x = self.deconv(x)
        x = F.relu(x)
        x = self.bn(x)

        return x

# ==== ENCODER/DECODER ====

class DCEncoder(nn.Module):
    def __init__(self, filters):
        super().__init__()

        self.conv = nn.Sequential(
            *[ConvBlock(fi, fo) for fi, fo in zip(filters[:-1], filters[1:])]
        )
        self.last_fo = filters[-1]

    def forward(self, x):
        x = self.conv(x)
        # -> [B, 512, 4, 4]
        x = eo.rearrange(x, 'B C H W -> B (C H W)') # flatten

        return x
    
class DCDecoder(nn.Module):
    def __init__(self, filters):
        super().__init__()

        self.deconv = nn.Sequential(
            *[DeconvBlock(fi, fo) for fi, fo in zip(filters[:-1], filters[1:])]
        )
        self.first_fi = filters[0]

    def forward(self, x):
        x = eo.rearrange(x, 'B (C H W) -> B C H W', C = 512, H = 4, W = 4)
        x = self.deconv(x)
        return x

# ====== OVERALL MODEL ======

class VAE(nn.Module):
    def __init__(self, latent_dim = 512, kl_weight = 10):
        super().__init__()

        hidden_dim = 512 * 4 * 4
        self.kl_weight = kl_weight

        # Make encoding layers
        filters = [1, 32, 64, 128, 256, 512, 512]
        self.encoder = DCEncoder(filters)
        self.mu_fc = nn.Linear(hidden_dim, latent_dim)
        self.logvar_fc = nn.Linear(hidden_dim, latent_dim)

        # Make decoding layers
        filters.reverse()
        self.proj = nn.Linear(latent_dim, hidden_dim)
        self.decoder = DCDecoder(filters)

        self.opt = None
    
    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x):
        x = self.encoder(x)
        mu, logvar = self.mu_fc(x), self.logvar_fc(x)
        return self.sample(mu, logvar)
    
    def decode(self, z):
        z = self.proj(z)
        z = self.decoder(z)
        return z
    
    def forward(self, x):
        x = self.encoder(x)
        mu, logvar = self.mu_fc(x), self.logvar_fc(x)
        z = self.sample(mu, logvar)

        x = self.proj(z)
        x_rec = self.decoder(x)
        return x_rec, mu, logvar
    
    def loss(self, x, x_rec, mu, logvar):
        # first the KL term
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # then the reconstruction term
        rec = F.mse_loss(x_rec, x, reduction = "sum")

        return self.kl_weight * kl + rec, rec
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))

    def init_train_state(self, lr = 2e-4):
        self.opt = torch.optim.AdamW(self.parameters(), lr = lr)

    def train_step(self, x):
        assert self.opt

        self.opt.zero_grad()
        x_rec, mu, logvar = self(x)
        loss, rec_loss = self.loss(x, x_rec, mu, logvar)
        loss.backward()
        self.opt.step()

        return rec_loss.item() / x.shape[0]

# Test train on a dataset of one song
if __name__ == "__main__":
    from data import *
    model = VAE()
    model.cuda()
    loader = WavFolderLoader("wav_data")
    data_unshuffled = loader.sample_waveforms_local(ind = 3, shuffle = False)
    data = data_unshuffled[torch.randperm(len(data_unshuffled))]

    batch_size = 64

    data = batchify(data, batch_size)

    # Evaluate by trying to reconstruct first 10 seconds of song
    eval_data = data_unshuffled[:sec_to_samp(10)].clone()
    def eval_step(i):
        chunks = batchify(eval_data, batch_size)
        for chunk in chunks:
            x = wf_to_spec(chunk)
            _, h, w = x.shape
            x = F.interpolate(x[:,None], (256, 256)).to('cuda')
            with torch.no_grad():
                x_rec, _, _ = model(x)
            x_rec = F.interpolate(x_rec, (h, w))
            x_rec = x_rec.cpu()
            x_rec = spec_to_wf(x_rec)
            x_rec = merge(x_rec)
            x_rec = to_stereo(x_rec)
            torchaudio.save('x_rec_{i}.wav', x_rec, sr)

    model.init_train_state()
    for i in range(10000):
        batch = data[i]
        batch = wf_to_spec(batch)
        batch = batch[:,None] # add a channel dim
        batch = F.interpolate(batch, (256, 256)).to('cuda')

        loss = model.train_step(batch)
        print(loss)

        if i % 100 == 0:
            eval_step(i)