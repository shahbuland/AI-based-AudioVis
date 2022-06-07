import os
import torch
import torchaudio as ta
import torchaudio.transforms as tatf

from utils import to_mono, sec_to_samp, slice, \
    get_length, chunk

class DataLoader:
    def __init__(self, data_path = "./wav", sr = 16000):
        self.target_sr = sr

        ds = self.dir2dataset(data_path)
        self.N = len(ds)

        self.index = None
        self.inds = None
    
    # Takes a single song and returns list of samples
    # wf : tensor of shape (n_channels, n_samples)
    # sr : sample rate of wf
    # window_size : size of window to split wf into (in seconds)
    # samples_per_window : to increase dataset diversity, 
    def prep(self, wf, sr, window_size = 0.25, hops_per_window = 4):
        """
        Takes a single song and returns list of samples

        :param wf: tensor of shape (n_channels, n_samples)
        :param sr: sample rate of wf
        :param window_size: size of window to split wf into (in seconds)
        :param hops_per_window: to increase dataset diversity, have multiple hops per window
        """
        wf = tatf.Resample(sr, self.target_sr)(wf)
        wf = to_mono(wf) # [n,]

        length = get_length(wf, self.target_sr) # length in seconds
        length = sec_to_samp(length, self.target_sr) # length in samples

        N = length
        W = sec_to_samp(window_size, self.target_sr) # window size in samples
        H = W // hops_per_window

        # Code ahead looks weird, but is just
        # splitting into samples of size W, with hop size H
        WF = [
            [wf_i[i : (i + W)] for i in range(0, N - 2*W, W)]
            for wf_i in \
                [wf[i : N] for i in range(0, W, H)]
        ]
        # [list of lists of samples]

        WF[:] = list(map(torch.stack, WF))
        # -> [list of tensors]

        WF = torch.cat(WF, dim = 0)
        # -> [n_windows * hops_per_window, W] tensor

        print(WF.shape)
        print(((N // W - 1) * hops_per_window), W)
        return WF

    def dir2dataset(self, path):
        """
        Convert directory of wav files into data tensor

        :param path: path to directory of wav files
        """
        dataset = []
        for filename in os.listdir(path):
            if filename.endswith(".wav"):
                wf, sr = ta.load(os.path.join(path, filename))
                wf = self.prep(wf, sr)
                dataset.append(wf)
        return torch.cat(dataset, dim = 0)

    def shuffle(self):
        """
        Initialize indices for shuffling dataset. Call to start an epoch.
        """
        self.index = 0
        self.inds = torch.randperm(self.N)
    
    def sample(self, batch_size, tform = None):
        """
        Get a batch of data from dataset. Return -1 if data is exhausted.

        :param batch_size: size of batch
        """
        if self.index is None or \
            (self.index + batch_size) >= self.N:
            raise Exception("No more samples")
        
        i = self.index 
        self.index += batch_size
        res = self.ds[i : i + batch_size]
        if tform is not None:
            res = tform(res)
        return res

# Tests for the dataloader object
if __name__ == "__main__":
    loader = DataLoader()
    wf, sr = ta.load("./wav/1.wav")
    loader.prep(wf, sr)
