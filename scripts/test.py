import torchaudio
import torch
from pathlib import Path
import os
import ipdb
import numpy as np
import json

if __name__ == "__main__":
    path = '/home/s6kogase/data/real/A_ljspeech'
    path_list = list(Path(path).glob(f"./*.wav"))
    path_list_alt = [file for file in os.listdir(path) if file.endswith(".wav")]

    parent = path_list[0].parent

    seconds = 1

    frame_dict = {}
    audio_list = []
    frame_list = []
    winsize_list = []

    # building lists for data loader for one label
    for file_name in path_list:
        meta = torchaudio.info(file_name)
        frame_dict[file_name] = meta.num_frames
        
        num_windows = meta.num_frames // int(seconds * meta.sample_rate)
        for i in range(num_windows):
            audio_list.append(file_name)
            frame_list.append(i)
            winsize_list.append(int(seconds * meta.sample_rate))

    frames_array = np.asarray([audio_list, frame_list, winsize_list], dtype=object).transpose()

    ipdb.set_trace()
    # testing
    for i in range(frames_array.shape[0]):
        try:
            audio, _ = torchaudio.load(frames_array[i, 0], frame_offset=frames_array[i, 1] * frames_array[i, 2], num_frames=frames_array[i, 2])
            if audio.shape[-1] != frames_array[i, 2]:
                print("Manno")
        except:
            print("aua")
            continue
    ipdb.set_trace()
    with open(f"{str(parent)}_meta_{seconds}", "w") as file:
        np.save(frames_array, file)