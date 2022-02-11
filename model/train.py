from torch.optim import AdamW

from vae import VAEModel
from encoders import ConvEncoder

def train_step(model, sample, opt):
    opt.zero_grad()
    loss, mse = model(sample)
    loss.backward()
    opt.step()
    return mse

if __name__ == "__main__"
    #model = VAEModel(2048, ConvEncoder
