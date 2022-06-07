# seconds to samples
def sec_to_samp(seconds, sr = 44100):
    return int(seconds * sr)

# get a subsection of wf, using seconds as an index
def slice(wf, start, end, sr = 44100):
    start = sec_to_samp(start)
    end = sec_to_samp(end)
    if len(wf.shape) == 2:
        return wf[:, start:end]
    elif len(wf.shape) == 1:
        return wf[start:end]
    else:
        raise Exception("Invalid shape going into slice")

# get length of wf in seconds
def get_length(wf, sr = 44100):
    if len(wf.shape) == 1:
        return len(wf) / sr
    elif len(wf.shape) == 2:
        return wf.shape[1] / sr

def to_mono(wf):
    n_ch, _ = wf.shape
    if n_ch == 1:
        return wf[0]
    else:
        return wf.mean(0)

# Chunk a tensor into list of tensors, where each chunk has size chunk_size
def chunk(t, chunk_size):
    return [t[i:i + chunk_size] for i in range(0, len(t), chunk_size)]