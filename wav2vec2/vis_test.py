from model import Encoder

import torch
import torchaudio
import torchaudio.transforms as tatf

import numpy as np

from datasets import load_dataset
import datasets

import utils

from umap.umap_ import UMAP

# This script gets embeddings from the model over 1.5 second chunks of audio file
# Then plots a umap reduction of said embeddings in 2d space

audio, sr = torchaudio.load("./wav/1.wav")
sample_time = 0.25
audio = tatf.Resample(sr, 16000)(audio)
sr = 16000
audio = utils.to_mono(audio)

audio_batch = [utils.slice(audio, i * sample_time, (i + 1) * sample_time, sr) for i in range(0, 200)] # 1.5 seconds per batch
audio_batch = torch.stack(audio_batch)
print(audio_batch.shape)

try:
    z = np.load("z.npy")
    assert False
except:
    mode = "data2vec"

    model = Encoder(mode, sr = sr)
    with torch.no_grad():
        h = model.encode(audio_batch.to('cuda'))
        h = h.cpu().numpy()

    # run dimensionality reduction over each
    print("UMAP time")
    umap = UMAP(n_neighbors = 10, n_epochs = 10, min_dist = 0.1, n_components = 2)
    z = umap.fit_transform(h)
    print(z.shape)
    np.save("z.npy", z)

from interactive_scatter import *

scatter_with_sounds(z, audio_batch, sr, labels = [0] * len(audio_batch))

