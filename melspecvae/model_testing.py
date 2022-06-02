import torch
from data import *

from vae import VAE

# This script tries to reconstruct a waveform from encoder/decoder model

def rec(spec, model):
    z = model.encode(spec)
    rec = model.decode(z)
    return rec

if __name__ == "__main__":
    model = VAE()
    model.load("vae.pt")

    # Load test song
    wf, sr = torchaudio.load("test.wav")
    wf = wf[:sec_to_samp(20)] # try to reconstruct first 20 seconds

    x = wf
    x = trim_silence(wf)
    x = split(x, hop, time_shape)
    x = torch.stack(x)

    spec = wf_to_spec(x)
    spec = rec(spec, model)
    wf_rec = spec_to_wf(spec)
    wf_rec = merge(wf_rec, hop, time_shape)
    wf_rec = to_stereo(wf_rec)
    torchaudio.save('test_vae_rec.wav', wf_rec, sr)