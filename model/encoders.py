import torch
from torch import nn
import torch.nn.functional as F

class ConvEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__

        conv_out = 512*4*4
        hidden_size = 1024
        conv_layers = 4
        hidden_layers = 4
        
        self.conv = nn.Sequential([])

        self.hidden_layers = nn.Sequential(*[
                nn.Linear(conv_out, hidden_size] + \
                [nn.Linear(hidden_size, hidden_size) 
                    for _ in range(hidden_layers - 1))])

        in_dim = hidden_size if hidden_layers > 0 else conv_out
        self.final_fc = nn.Linear(in_dim, out_dim)

    def forward(self, x)
        x = self.conv(x)
        x = self.hidden_layers(x)
        x = self.final_fc(x)
        return x
