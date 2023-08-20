import torchaudio
import torch
from pathlib import Path
import os
import ipdb
import numpy as np
import json
from tqdm import tqdm
from typing import Optional
from torch.utils.data import Dataset
from src.data_loader import get_ds_label
from torch.utils.data import DataLoader


class CustomDataset(Dataset):
    """Create a data loader for custom paths."""

    def __init__(
        self,
        paths: list,
        labels: list,
        ds_type: str = 'train',
        save_path = '/home/s6kogase/data/run1',
        source_name: str = "fake",
        seconds: int = 1,
        resample_rate: int = 16000,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        key: Optional[str] = "audio",
        limit: tuple = (-1, -1, -1),    # (train, val, test) per label
        verbose: Optional[bool] = False,
    ):
        """Create a Wavefake-dataset object.

        Dataset with additional transforming in cwt space.

        Args:
            data_dir: A path to a pre-processed folder with numpy files.
            key: The key for the input or 'x' component of the dataset.
                Defaults to "audio".

        Raises:
            ValueError: If an unexpected file name is given or directory is empty.

        # noqa: DAR401
        """
        if verbose:
            print("Loading ", ds_type, paths, flush=True)

        names = []
        for path in paths:
            names.append(path.split("/")[-1].split("_")[-1])
        
        destination = f"{save_path}/dataset_{'-'.join(names)}_meta_{seconds}sec"
        if (os.path.exists(f"{destination}_train.npy") and 
            os.path.exists(f"{destination}_val.npy") and 
            os.path.exists(f"{destination}_test.npy")
        ):
            result_train = np.load(f"{destination}_train.npy", allow_pickle=True)
            result_val = np.load(f"{destination}_val.npy", allow_pickle=True)
            result_test = np.load(f"{destination}_test.npy", allow_pickle=True)
        else:
            train_data = []
            val_data = []
            test_data = []
            min_sample_rate = 192000     # assuming this will be the highest sample rate ever
            sample_count = []
            path_num = 0
            for path in tqdm(paths, desc="Process paths"):
                names.append(path.split("/")[-1].split("_")[-1])
                path_list = list(Path(path).glob(f"./*.wav"))

                frame_dict = {}
                audio_list = []
                frame_list = []
                winsize_list = []

                # building lists for data loader for one label
                for file_name in tqdm(path_list, desc="Process files in path"):
                    meta = torchaudio.info(file_name)
                    frame_dict[file_name] = meta.num_frames
                    min_sample_rate = min(meta.sample_rate, min_sample_rate)
                    
                    num_windows = meta.num_frames // int(seconds * meta.sample_rate)
                    for i in range(num_windows):
                        audio_list.append(file_name)
                        frame_list.append(i)
                        winsize_list.append(int(seconds * meta.sample_rate))

                frames_array = np.asarray(
                    [
                        audio_list,
                        frame_list,
                        winsize_list,
                        [labels[path_num]] * len(audio_list)
                    ],
                    dtype=object
                ).transpose()
                num_samples = frames_array.shape[0]
                num_train = int(train_ratio * num_samples)
                num_val = int(val_ratio * num_samples)
                num_test = num_samples - num_train - num_val

                train_data.append(frames_array[:num_train])
                val_data.append(frames_array[num_train:num_train + num_val])
                test_data.append(frames_array[num_train + num_val:])

                sample_count.append([num_train, num_val, num_test])
                path_num += 1

            sample_count = np.asarray(sample_count)
            min_len = sample_count.transpose().min(axis=1)
            result_train = self.get_result_set(train_data, min_len[0])
            result_val = self.get_result_set(val_data, min_len[1])
            result_test = self.get_result_set(test_data, min_len[2])

            np.save(f"{destination}_train.npy", result_train, allow_pickle=True)
            np.save(f"{destination}_val.npy", result_val, allow_pickle=True)
            np.save(f"{destination}_test.npy", result_test, allow_pickle=True)

        # proceed with preparation for loading
        
        # apply limit per label
        result_train = result_train[:, :limit[0]]
        result_val = result_val[:, :limit[1]]
        result_test = result_test[:, :limit[2]]

        # make sure, all frames will result in the same window size after resampling
        min_sample_rate = min(result_train[:,:,2].min(), result_val[:,:,2].min(), result_test[:,:,2].min())
        if resample_rate > min_sample_rate:
            raise RuntimeError("Sample rate is smaller than desired sample rate. No upsampling possible here.")

        audio_data = None
        if ds_type == "train":
            result = result_train
        elif ds_type == "val":
            result = result_val
        elif ds_type == "test":
            result = result_test
        else:
            raise RuntimeError("Dataset type does not exists.")

        for i in range(result.shape[0]):
            if audio_data is None:
                audio_data = result[i]
            else:
                audio_data = np.vstack([audio_data, result[i]])

        self.audio_data = audio_data    # shape (number of samples, 4)

        self.ds_type = ds_type
        self.key = key
        self.resample_rate = resample_rate
        self.label_names = {0: "original", get_ds_label(self.audio_data[:,3]): source_name}

    def get_result_set(self, frames, min_len):
        result = None   # will be of shape (len(paths), frame_number, individual_window_size, label)
        for frame_array in frames:
            if result is None:
                result = frame_array[:min_len]
            else:
                result = np.stack([result, frame_array[:min_len]])
        return result

    def get_label_name(self, key):
        if key in self.label_names.keys():
            return self.label_names[key]
        else:
            return f"John Doe Generator {key}"

    def __len__(self) -> int:
        """Return the data set length."""
        return int(self.audio_data.shape[0])

    def __getitem__(self, idx: int) -> dict:
        """Get a dataset element.

        Args:
            idx (int): The element index of the data pair to return.

        Returns:
            sample (dict): Returns a dictionary with the self.key
                    default ("audio") and "label" keys.
        """
        audio, sample_rate = torchaudio.load(
            self.audio_data[idx, 0],
            frame_offset=self.audio_data[idx, 1] * self.audio_data[idx, 2],
            num_frames=self.audio_data[idx, 2]
        )
        
        if sample_rate > self.resample_rate:
            audio = torchaudio.functional.resample(audio, sample_rate, self.resample_rate)
        elif sample_rate < self.resample_rate:
            raise RuntimeError("Sample rate is smaller than desired sample rate. No upsampling possible here.")

        label = torch.tensor(self.audio_data[idx, 3])
        sample = {self.key: audio, "label": label}
        return sample
    

def get_costum_dataset(ds_set, ds_type):
    save_path = '/home/s6kogase/data/run1'
    train_path = '/home/s6kogase/data/data/train'
    cross_test_path = '/home/s6kogase/data/data/cross_test'
    if ds_set == "train":
        paths = list(Path(train_path).glob(f"./*_*"))
    elif ds_set == "cross_test":
        paths = list(Path(cross_test_path).glob(f"./*_*"))
        ds_type = "only_test"
    else:
        raise TypeError("Set name does not exist. Choose on of [train, cross_test].")

    labels = []
    for path in paths:
        labels.append(ord(path.name.split("_")[0]) - 65)

    ipdb.set_trace()
    seconds = 1
    resample_rate = 22050
    # limit = (55500, 7500, 15500)   # pro label
    
    return CustomDataset(
        paths=paths,
        labels=labels,
        save_path=save_path,
        seconds=seconds,
        resample_rate=resample_rate,
        verbose=True,
        ds_type=ds_type
    )


if __name__ == "__main__":
    get_costum_dataset("cross_test", "train")
    