from importlib.abc import PathEntryFinder
import torch
import torch.nn.functional as F
from torchaudio.transforms import Vad, MelSpectrogram
import torchaudio.functional as AF
import torchaudio
from torch import nn

from tqdm import tqdm
import numpy as np
import librosa 
import os
import random

# Basic librosa utilities

def pow_to_db(x):
    x = x.numpy()
    x = librosa.power_to_db(x)
    return torch.from_numpy(x)

def db_to_pow(x):
    x = x.numpy()
    x = librosa.db_to_power(x)
    return torch.from_numpy(x)

# domain conversion utilities

def sec_to_samp(sec):
    return int(44100 * sec)

def samp_to_sec(samp):
    return samp / 44100

def to_mono(wf):
    return wf.sum(0) / 2

def to_stereo(wf):
    return torch.stack([wf, wf])

# ======== ACTING ON SINGLE WAVEFORM =========

# constants
hop = 128
sr = 44100
min_db = -100
ref_db = 20
time_shape = 2048 # time axis length

"""
spec_tform = Spectrogram(
    n_fft = 4 * hop,
    win_length = time_shape,
    hop_length = hop,
    pad = 0,
    power = 2,
    normalized = False
)

mel_tform = MelScale(
    n_mels = hop,
    sample_rate = sr,
    f_min = 0
)
"""
melspec_tform = MelSpectrogram(
    n_mels = time_shape // 4,
    sample_rate = sr,
    f_min = 0,
    n_fft = time_shape,
    win_length = time_shape,
    hop_length = hop,
    pad = 0,
    power = 2,
    normalized = False
)

def spectral_convergence(input, target):
    return 20 * ((input - target).norm().log10() - target.norm().log10())

def normalize(spec):
    spec = (spec - min_db) / (-1 * min_db)
    spec = (spec * 2) - 1
    spec = torch.clip(spec, -1, 1)

    return spec

def denormalize(spec):
    spec = torch.clip(spec, -1, 1)
    spec = (spec + 1) / 2
    spec = spec * (-1 * min_db)
    spec = spec + min_db

    return spec

def wf_to_spec(wf):
    s = melspec_tform(wf)
    s = pow_to_db(s) - ref_db
    return normalize(s)

# Gradient based method to recover waveform
def approximate_waveform(spec, maxiter = 250, verbose = False):

    samples = (spec.shape[-1] * hop) - hop
    n_spec = spec.shape[0]

    device = 'cuda'
    spec = spec.to(device)
    tform = melspec_tform.to(device)
    x = nn.Parameter(data = torch.randn(n_spec, samples, device = device)  * 1e-6)

    loss_fn = nn.L1Loss()

    lr = 0.01
    opt = torch.optim.AdamW([x], lr = lr)

    for i in tqdm(range(maxiter)) if verbose else range(maxiter):
        opt.zero_grad()
        spec_pred = tform(x)
        loss = loss_fn(spec_pred, spec)

        loss.backward()
        opt.step()

    return x.cpu().detach().squeeze()
    
def spec_to_wf(spec_arr, verbose = True):
    # add a batch dim if not there already
    if spec_arr.ndim == 2:
        spec_arr = spec_arr.unsqueeze(0)

    spec_arr = [denormalize(spec) + ref_db for spec in spec_arr]
    spec_arr = [db_to_pow(spec) for spec in spec_arr]
    spec_arr = torch.stack(spec_arr)
    wf_arr = approximate_waveform(spec_arr, verbose = True)
    #wf_arr = torch.stack([approximate_waveform(spec[None,:]) for spec in tqdm(spec_arr)])
    return wf_arr

# Trim silence from front and back of a waveform
def trim_silence(wf):
    tform = Vad(44100)
    
    # trim from front
    wf = tform(wf)

    # reverse to trim from back
    wf = wf.flip(0)
    wf = tform(wf)

    # reverse again to get back to original order
    wf = wf.flip(0)

    return wf

# Split waveform into chunks 
def split(wf, hop, window_size):
    # assume wf is [n]
    pos = 0
    n = len(wf)
    res = []

    while pos + window_size < n:
        res.append(wf[pos:pos + window_size])
        pos += hop
    return res

# invert the above, merge chunked waveforms into one
# assumes chunks have given hop between them (i.e. possible overlap)
# and that the chunks are of the same length (window_size)
def merge(wf_arr, hop = hop, window_size = time_shape):
    # want to get the total number of samples in the resulting waveform
    
    # first window contributes window_size samples
    # every window after that contributes those same samples + hop more
    #   (because window_size - hop is the overlap)
    total_samples = window_size + (wf_arr.shape[0] - 1) * hop

    res_wf = torch.zeros(total_samples)

    pos = 0
    for wf in wf_arr:
        res_wf[pos:pos + window_size] += wf
        pos += hop
    
    # because we added them together, any place two windows overlap becomes louder
    # than intended. this can be fixed by dividing each segment by the number of windows
    # that overlapped on it. the segments correspond to hops
    # for all but the first window, every segment was overlapped by same number of windows
    # so we just have to do the individual calculations for the first window

    pos = 0
    div = 0
    while pos < window_size:
        div += 1
        res_wf[pos:pos + hop] /= div
        pos += hop
    res_wf[pos:] /= div

    return res_wf

# ==== FILE I/O ====

# Take a path to directory of wav files
# convert each to a waveform array
# returns a list of waveform slices for every file
def wav_dir_to_wf_arr(path, hop = hop, window = time_shape):
    wf_arr = []
    for file in os.listdir(path):
        if file.endswith(".wav"):
            wf, sr = torchaudio.load(os.path.join(path, file))
            assert sr == 44100

            wf = to_mono(wf)
            wf = trim_silence(wf)
            wf = split(wf, hop, window)
            wf = torch.stack(wf)
            wf_arr += wf

    return wf_arr # -> [n slices overall, window_size]

# Given tensor, returns list of batches of specified batch size or less
def batchify(tensor, batch_size):
    res = []
    for i in range(0, tensor.shape[0], batch_size):
        res.append(tensor[i:i+batch_size])
    return res

# This loader discards sequential info
class WavFolderLoader:
    def __init__(self, path):
        files = os.listdir(path)
        # filter for files that are wav
        is_wav = lambda x: x.endswith(".wav")
        self.files = [file for file in filter(is_wav, files)]

        # paths of all wav files that will be in dataset
        self.paths = [os.path.join(path, file) for file in self.files]
        self.N = len(self.paths)
    
    # Pick a random file and sample waveforms from it
    def sample_waveforms_local(self, hop = hop, window = time_shape, ind = None, shuffle = True):
        # pick a random file

        if ind is None:
            path = random.choice(self.paths)
        else:
            path = self.paths[ind]

        wf, sr = torchaudio.load(path)
        assert sr == 44100

        wf = to_mono(wf)
        wf = trim_silence(wf)

        wf = split(wf, hop, window)
        wf = torch.stack(wf)

        if shuffle: wf = wf[torch.randperm(wf.shape[0])]

        return wf

    # get a batch of waveforms, randomly sampled from all files
    def sample_waveforms_global(self, n, hop = hop, window = time_shape):
        # pick a file index for each randomly
        file_inds = torch.randint(0, self.N, (n,))
        inds, freqs = torch.unique(file_inds, return_counts = True)
        res = []
        for ind, freq in tqdm(zip(inds, freqs)):
            wf, sr = torchaudio.load(self.paths[ind])
            assert sr == 44100

            wf = to_mono(wf)
            wf = trim_silence(wf)

            start_pos = torch.randint(0, len(wf) - window, (freq,))
            end_pos = start_pos + window

            wf_arr = [wf[start:end] for start, end in zip(start_pos, end_pos)]
            wf_arr = torch.stack(wf_arr)
            res.append(wf_arr)
        
        return torch.cat(res)

# Wrapper for above that will load new data when needed
class BufferedLoader:
    def __init__(self, path, batch_size = 32, buffer_size = 1024):
        self.loader = WavFolderLoader(path)
        self.buffer = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size
    
    def sample(self, n, hop = hop, window = time_shape):
        if not self.buffer:
            samples = self.loader.sample_waveforms_local(self.buffer_size, hop, window)
            self.buffer = batchify(samples, self.batch_size)
        else:
            return self.buffer.pop(0)


# Tests for the above functions
if __name__ == '__main__':
    wf, sr = torchaudio.load('test.wav')
    wf = to_mono(wf)
    wf = wf[:sec_to_samp(5)]

    x = wf
    x = trim_silence(wf)
    x = split(x, hop, time_shape)
    x = torch.stack(x)

    spec = wf_to_spec(x)

    print(spec.shape)
    wf_rec = spec_to_wf(spec)
    wf_rec = merge(wf_rec, hop, time_shape)
    wf_rec = to_stereo(wf_rec)
    torchaudio.save('test_rec.wav', wf_rec, sr)