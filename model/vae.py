import torch
from torch import nn
import torch.nn.functional as F

# latent_dim: size of representation

# both of below can also be custom constructor functions
# encoder_cls: class to instantiate for encoder
# decoder_cls: class to instantiate for decoder

# kl_weight: weight given to kl term in loss
# bound_fc: n neurons in last enc layer/ first dec layer
class VAEModel(nn.Module):
    def __init__(self, latent_dim, encoder_cls, decoder_cls, kl_weight = 10, bound_fc = None):
        super().__init__()

        if bound_fc is None: bound_fc = latent_dim

        self.encoder = encoder_cls(bound_fc)
        self.decoder = decoder_cls(bound_fc)

        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.bound_fc = bound_fc

        self.mu_fc = nn.Linear(bound_fc, latent_dim)
        # logvar = log(sigma ** 2)
        # sigma = sqrt(e ** logvar)
        self.logvar_fc = nn.Linear(bound_fc, latent_dim)

    # given tensor of means and logvars, sample random vector
    def sample(self, mu, logvar):
        sigma = logvar.exp().sqrt()
        return torch.normal(mu, sigma)

    # given input x, get representation
    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_fc(h)
        logvar = self.logvar_fc(h)
        return self.sample(mu, logvar)

    # given representation get reconstructed sample
    def decode(self, z):
        return self.decoder(x)

    # evaluates VAE loss (also returns raw mse for eval)
    def forward(self, x):
        h = self.encoder(x)
        
        mu = self.mu_fc(h)
        logvar = self.logvar_fc(h)
        sigma = logvar.exp().sqrt()

        z = self.sample(mu, logvar)
        rec_x = self.decoder(z)

        # KL Loss
        # Formula for D_{KL}(N(mu, sigma) || N(0, 1)) in multivar case works out to below
        kl_div = (sigma.log() + sigma).sum() + (mu * mu).sum()
        
        # Reconstruction Loss
        rec_loss = ((x - rec_x)**2).sum()

        return self.kl_weight * kl_div + rec_loss, rec_loss


        

