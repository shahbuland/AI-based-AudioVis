import os
import ffmpy

# Convert all mp3 files in a directory to wav files
def mp3towav(path, res_folder = "wav"):
    # check if  res folder exists already
    if not os.path.exists(res_folder):
        os.mkdir(res_folder)

    paths = os.listdir(path)
    paths = [os.path.join(path, p) for p in paths if p.endswith(".mp3")]

    for p in paths:
        ff = ffmpy.FFmpeg(
            inputs = {p: None},
            outputs = {os.path.join(res_folder, os.path.basename(p).split(".")[0] + ".wav"): None}
        )
        ff.run()

if __name__ == "__main__":
    mp3towav("./mp3", "./wav")
