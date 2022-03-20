import torch
from torch import nn
import torch.nn.functional as F

# latent_dim: size of representation

# both of below can also be custom constructor functions
# encoder_cls: class to instantiate for encoder
# decoder_cls: class to instantiate for decoder

# kl_weight: weight given to kl term in loss
# bound_fc: n neurons in last enc layer
class VAEModel(nn.Module):
    def __init__(self, latent_dim, encoder_cls, decoder_cls, kl_weight = 10, bound_fc = None, **kwargs):
        super().__init__()

        if bound_fc is None: bound_fc = latent_dim

        self.encoder = encoder_cls(bound_fc, **kwargs)
        self.decoder = decoder_cls(latent_dim, **kwargs)

        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.bound_fc = bound_fc

        self.mu_fc = nn.Linear(bound_fc, latent_dim)
        # logvar = log(sigma ** 2)
        # sigma = sqrt(e ** logvar)
        self.logvar_fc = nn.Linear(bound_fc, latent_dim)

    # given tensor of means and logvars, sample random vector
    def sample(self, mu, logvar):
        sigma = (logvar / 2).exp()
        # torch.normal bugs in backprop, need to use an intermediate variable
        # that isnt trained on
        eps = torch.randn_like(mu)
        return sigma * eps + mu

    # given input x, get representation
    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_fc(h)
        logvar = self.logvar_fc(h)
        return self.sample(mu, logvar)

    # given representation get reconstructed sample
    def decode(self, z):
        return self.decoder(z)

    # evaluates VAE loss (also returns raw mse for eval)
    def forward(self, x, return_rec = False):
        h = self.encoder(x)
        
        mu = self.mu_fc(h)
        logvar = self.logvar_fc(h)

        z = self.sample(mu, logvar)
        rec_x = self.decoder(z)

        if return_rec: return rec_x

        # KL Loss
        # Formula for D_{KL}(N(mu, sigma) || N(0, 1)) in multivar case works out to below
        kl_div = 0.5 * torch.sum(-1 - logvar + mu.pow(2) + logvar.exp())
        
        # Reconstruction Loss
        rec_loss = ((x - rec_x)**2).sum()

        return self.kl_weight * kl_div + rec_loss, rec_loss

    # checkpoints
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        try:
            self.load_state_dict(torch.load(path))
        except:
            print("Failed to load model checkpoint")
        

