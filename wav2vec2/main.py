import transformers as tf
import torch
from datasets import load_dataset
import datasets

from IPython.display import Audio, display

options = {
    "base" : "facebook/wav2vec2-base-960h",
    "large" : "facebook/wav2vec2-large-robust"
}

model = tf.AutoModelForCTC.from_pretrained(options["base"])
tokenizer = tf.AutoTokenizer.from_pretrained(options["base"])
feature_extractor = tf.AutoFeatureExtractor.from_pretrained(options["base"])

cnt = 0
for p in model.parameters():
    cnt += p.numel()
print(cnt)

dataset = load_dataset("common_voice", "en", split="train", streaming = True)
dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate = 16_000))
dataset_iter = iter(dataset)
sample = next(dataset_iter)

with torch.no_grad():
    # Generate a random sample
    print(type(sample))

