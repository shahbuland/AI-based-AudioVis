from data import *

if __name__ == "__main__":
    loader = WavFolderLoader("wav_data")
    sample = loader.sample_waveforms(32)
    print(sample.shape)