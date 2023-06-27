"""Clean up train, test and validation sets to be exactly equally distributed in all labels."""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample
from tqdm import tqdm

from src.data_loader import LearnWavefakeDataset

if __name__ == "__main__":
    sample_rate = 22050
    resample_rate = 16000
    batch_size = 4096

    path = "/home/s6kogase/data/run6"  # Replace with the path to the directory where the prepared audiofiles are
    new_path = "/home/s6kogase/data/run6_" + str(resample_rate)

    if not os.path.exists(new_path):
        os.makedirs(new_path)

    resampler = Resample(
        sample_rate,
        resample_rate,
        resampling_method="sinc_interp_kaiser",
        dtype=torch.float32,
    ).cuda()

    for end in ["_val", "_test", "_train"]:
        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            new_folder_path = os.path.join(new_path, folder)
            if (
                os.path.isdir(folder_path)
                and folder.endswith(end)
                and "all" in folder_path
            ):
                if not os.path.exists(new_folder_path):
                    os.makedirs(new_folder_path)

                dataset = LearnWavefakeDataset(folder_path)

                data_loader = DataLoader(
                    dataset,
                    shuffle=False,
                    drop_last=False,
                    batch_size=batch_size,
                    num_workers=4,
                )
                file_count = 0

                for batch in tqdm(
                    iter(data_loader),
                    desc="resample",
                    total=len(data_loader),
                ):
                    waveform = batch["audio"].cuda()
                    resampled_waveform = resampler(waveform).cpu()

                    for pre_processed_audio in resampled_waveform:
                        with open(
                            f"{new_folder_path}/{file_count:06}.npy", "wb"
                        ) as numpy_file:
                            np.save(numpy_file, pre_processed_audio)

                        file_count += 1

                with open(f"{new_folder_path}/labels.npy", "wb") as label_file:
                    np.save(label_file, np.array(dataset.labels))
