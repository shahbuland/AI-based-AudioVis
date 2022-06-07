import torch
from torch import nn
import torch.nn.functional as F

import transformers as tf

options = {
    "base" : "facebook/wav2vec2-base-960h",
    "large" : "facebook/wav2vec2-large-robust",
    "data2vec" : "facebook/data2vec-audio-base-960h"
}

class Encoder(nn.Module):
    def __init__(self, mode, sr = 16000, device = 'cuda'):
        super(Encoder, self).__init__()

        self.sr = sr
        self.preprocess =  tf.AutoFeatureExtractor.from_pretrained(options[mode])
        self.model = tf.AutoModelForCTC.from_pretrained(options[mode])
        self.model = self.model.to(device)
        
        self.logit_scale = nn.Parameter(
            torch.ones([], device = device)
            * torch.log(torch.tensor([1 / 0.07], device = device))
        )

        self.loss_fn = nn.MSELoss()

        self.device = device

    def prepare(self, x):
        """
        Process waveform into tensor.
        """
        with torch.no_grad():
            x  =  self.preprocess(x, sampling_rate = self.sr, return_tensors = "pt").input_values[0]
        return x.to(self.device)

    def encode(self, wf):
        """
        Directly turn waveform into an embedding.
        """
        x = self.prepare(wf)
        return self.forward(x)
    
    def forward(self, x, masks = None):
        """
        Turn processed waveform  into embedding.
        """
        if masks is None:
            y = self.model(x, output_hidden_states = True)["hidden_states"]
        else:
            y = self.model(x, attention_mask = masks, output_hidden_states = True)["hidden_states"]
        h = y[-1] # output of last hidden layer
        h = h[:, -1, :] # output for last sample
        h = F.normalize(h, dim = 1)
        return h

    def c_loss(self, x, y):
        """
        Calculate contrastive loss between embeddings.
        """
        n = x.shape[0]

        logits = torch.abs(x @ y.t()) * self.logit_scale.exp()
        labels = torch.arange(n, device = self.device)

        loss = F.cross_entropy(logits, labels)

        return loss
        

