"""Data loading utility for WaveFake audio classification.

Holds a custom torch Dataset class implementation that prepares the audio
and cuts it to frames and transforms it with CWT. Dataloader methods
can be found here to.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm


def get_ds_label(labels):
    """Retrieve current label from binary dataset."""
    for label in labels:
        if label != 0:
            return label
    return np.int64(0)


class WelfordEstimator:
    """Compute running mean and standard deviations.

    The Welford approach greatly reduces memory consumption.
    Port of: https://github.com/gan-police/frequency-forensics/blob/main/src/freqdect/prepare_dataset.py
    """

    def __init__(self) -> None:
        """Create a Welfordestimator."""
        self.collapsed_axis: Optional[Tuple[int, ...]] = None

    # estimate running mean and std
    # average all axis except the color channel
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    def update(self, batch_vals: torch.Tensor) -> None:
        """Update the running estimation.

        Args:
            batch_vals (torch.Tensor): The current batch element.
        """
        if not self.collapsed_axis:
            self.collapsed_axis = tuple(np.arange(len(batch_vals.shape[:-1])))
            self.count = torch.zeros(1, device=batch_vals.device, dtype=torch.float32)
            self.mean = torch.zeros(
                batch_vals.shape[-1], device=batch_vals.device, dtype=torch.float32
            )
            self.std = torch.zeros(
                batch_vals.shape[-1], device=batch_vals.device, dtype=torch.float32
            )
            self.m2 = torch.zeros(
                batch_vals.shape[-1], device=batch_vals.device, dtype=torch.float32
            )
        self.count += torch.prod(torch.tensor(batch_vals.shape[:-1]))
        delta = torch.sub(batch_vals, self.mean)
        self.mean += torch.sum(delta / self.count, self.collapsed_axis)
        delta2 = torch.sub(batch_vals, self.mean)
        self.m2 += torch.sum(delta * delta2, self.collapsed_axis)

    def finalize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Finish the estimation and return the computed mean and std.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Estimated mean and variance.
        """
        return self.mean, torch.sqrt(self.m2 / self.count)


class CustomDataset(Dataset):
    """Create a data loader for custom paths.

    This class will traverse the given directories and create an equally distributed dataset (all labels
    are represented with same amount) over these folders. If two folders with real and fake audios are given,
    this class will represent a binary dataset. You can give any amount of folders and set the limit for each label
    as well. The dataset will contain information about each audio file (where it is saved), about the used samples
    from this audio, the corresponding label and the desired resample rate. All these infos will be applied when
    a data sample is retrieved (e.g. by torch dataloader).
    """

    def __init__(
        self,
        paths: list,
        labels: list,
        save_path: str,
        only_test_folders: Optional[list] = None,
        abort_on_save: bool = False,
        ds_type: str = "train",
        seconds: float = 1,
        resample_rate: int = 16000,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        key: Optional[str] = "audio",
        limit: int = 555000,
        verbose: Optional[bool] = False,
        filetype: str = "wav",
        asvspoof_name: str | None = None,
    ):
        """Create a custom-dataset object.

        Args:
            paths (list): Paths to audio files to include.
            labels (list): Labels for given audio paths.
            save_path (str): Path to save the dataset in.
            only_test_folders (Optional[list]): List of folders that only contain audios for testing. Defaults to None.
            abort_on_save (bool): Abort after saving. Use this to only prepare the datasets. Defaults to False.
            ds_type (str): Type of this dataset. Can be train, test or val. Defaults to "train".
            seconds (float): Length of desired audio snippets. This will be used to cut the audio files
                             into same-length frames. Defaults to 1.
            resample_rate (int): Desired sample rate. This will later on be used to resample the given audio.
                                 Defaults to 16000.
            train_ratio (float): Size of train set as decimal number. Defaults to 0.7.
            val_ratio (float): Size of validation set as decimal number. Defaults to 0.1.
            key (Optional[str]): Key for the audio data in the resulting get item method. Defaults to "audio".
            limit (int): Limit of maximum audios per label. Defaults to 555000.
            verbose (Optional[bool]): If more verbose messages should be displayed. Defaults to False.
            filetype (str): The file ending of the audio files. Defaults to "wav".
            asvspoof_name (str | None): The file prefix if asvspoof files are provided, e.g. DF_E. Defaults to None.

        Raises:
            RuntimeError: If desired resample rate for auido files is bigger than the actual sample rate (no upsampling
                          possible).
            ValueError: If only_test_folders are specified and ds_type is "train". This is contradictory.
            RuntimeError: If ds_type is not "train", "test" or "val".
        """
        if verbose:
            print("Loading ", ds_type, paths, flush=True)

        names = []
        self.label_names = {}
        for i in range(len(paths)):
            names.append(paths[i].split("/")[-1].split("_")[-1])
            self.label_names[labels[i]] = names[-1]

        destination = f"{save_path}/dataset_{'-'.join(names)}_meta_{seconds}sec"
        if os.path.exists(f"{destination}_train.npy") and ds_type == "train":
            result_set = np.load(f"{destination}_train.npy", allow_pickle=True)
        elif os.path.exists(f"{destination}_val.npy") and ds_type == "val":
            result_set = np.load(f"{destination}_val.npy", allow_pickle=True)
        elif os.path.exists(f"{destination}_test.npy") and ds_type == "test":
            result_set = np.load(f"{destination}_test.npy", allow_pickle=True)
        else:
            print(
                "Reading dataset. This may take more than 45 minutes, so maybe grab a coffee."
                "The result will be saved to your hard drive to speed up the dataloading"
                "in further experiments."
            )
            train_data = []
            val_data = []
            test_data = []
            min_sample_rate = (
                192000  # assuming this will be the highest sample rate ever
            )
            sample_count: list = []
            path_num = 0
            for path in tqdm(paths, desc="Process paths"):
                name = path.split("/")[-1].split("_")[-1]
                names.append(name)
                if asvspoof_name is not None:
                    path_list = list(Path(path).glob(f"./{asvspoof_name}*.{filetype}"))
                else:
                    path_list = list(Path(path).glob(f"./*.{filetype}"))
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
                        [labels[path_num]] * len(audio_list),
                    ],
                    dtype=object,
                ).transpose()
                num_samples = frames_array.shape[0]

                if only_test_folders is None or name not in only_test_folders:
                    num_train = int(train_ratio * num_samples)
                    num_val = int(val_ratio * num_samples)
                    num_test = num_samples - num_train - num_val
                else:
                    num_train = 0
                    # use previous calculated lengths for val and test sets if enough samples are provided
                    if (
                        len(sample_count) != 0
                        and num_samples >= sample_count[-1][1] + sample_count[-1][2]
                    ):
                        num_val = sample_count[-1][1]
                        num_test = sample_count[-1][2]
                    else:
                        # else use maximum samples available in same ratio
                        num_val = int(val_ratio / (1.0 - train_ratio) * num_samples)
                        num_test = num_samples - num_val

                train_data.append(frames_array[:num_train])
                val_data.append(frames_array[num_train : num_train + num_val])
                test_data.append(frames_array[num_train + num_val :])

                if only_test_folders is not None and name in only_test_folders:
                    if len(sample_count) != 0:
                        num_train = sample_count[-1][0]
                    else:
                        print(
                            "Warning: Only test folder came first. Defaulting to given limit for train set."
                        )
                        if limit == -1:
                            num_train = 55500
                        else:
                            num_train = limit  # TODO: Test!

                sample_count.append([num_train, num_val, num_test])
                path_num += 1

            min_len = np.asarray(sample_count).transpose().min(axis=1)

            if ds_type == "train":
                if only_test_folders is not None and len(only_test_folders) != 0:
                    result_set = np.zeros([0, 0, 0])
                else:
                    result_set = self.get_result_set(train_data, min_len[0])
            elif ds_type == "val":
                result_set = self.get_result_set(val_data, min_len[1])
            else:
                result_set = self.get_result_set(test_data, min_len[2])

            np.save(f"{destination}_{ds_type}.npy", result_set, allow_pickle=True)

            if abort_on_save:
                print("Aborting on dataset saving.")
                quit()

        # proceed with preparation for loading
        # apply limit per label
        result_set = result_set[:, :limit]

        # make sure, all frames will result in the same window size after resampling

        min_sample_rate = result_set[:, :, 2].min()
        if resample_rate > min_sample_rate:
            raise RuntimeError(
                "Sample rate is smaller than desired sample rate. No upsampling possible here."
            )

        audio_data = np.zeros(0)
        if ds_type == "train":
            if only_test_folders is not None and len(only_test_folders) != 0:
                raise ValueError(
                    "Since there are folders in only_test_folders this cannot be a train dataset."
                )
        elif ds_type != "val" and ds_type != "test":
            raise RuntimeError("Dataset type does not exists.")

        result = result_set
        for i in range(result.shape[0]):
            if len(audio_data) != 0:
                audio_data = np.vstack([audio_data, result[i]])
            else:
                audio_data = result[i]

        self.audio_data = audio_data  # shape (number of samples, 4)
        self.ds_type = ds_type
        self.key = key
        self.resample_rate = resample_rate

    def get_result_set(self, frames: List[np.ndarray], min_len: int):
        """Build up frame array from frames list using only `min_len` frames.

        Args:
            frames (List[np.ndarray]): List containing arrays with file names, window sizes
                                       and labels.
            min_len (int): Used frames per result set. This is used to limit the output size
                           per label.

        Returns:
            np.ndarray: Array of shape (len(paths), frame_number, individual_window_size, label).
        """
        result = None
        for frame_array in frames:
            if result is None:
                result = np.expand_dims(frame_array[:min_len], 0)
            else:
                result = np.concatenate(
                    [result, np.expand_dims(frame_array[:min_len], 0)]
                )
        return result

    def get_label_name(self, key: Union[int, str]) -> str:
        """Get name to corresponding label.

        Args:
            key (Union[int, str]): Key of the label.

        Returns:
            str: Name of the label.
        """
        if key in self.label_names.keys():
            return self.label_names[key]
        else:
            return f"John Doe Generator {key}"

    def __len__(self) -> int:
        """Return the data set length."""
        return int(len(self.audio_data))

    def __getitem__(self, idx: int) -> dict:
        """Get a dataset element.

        Args:
            idx (int): The element index of the data pair to return.

        Returns:
            sample (dict): Returns a dictionary with the self.key
                           default ("audio") and "label" keys.

        Raises:
            RuntimeError: If sample rate is smaller than desired resample rate.
        """
        audio, sample_rate = torchaudio.load(
            self.audio_data[idx, 0],
            frame_offset=self.audio_data[idx, 1] * self.audio_data[idx, 2],
            num_frames=self.audio_data[idx, 2],
        )

        if sample_rate > self.resample_rate:
            audio = torchaudio.functional.resample(
                audio, sample_rate, self.resample_rate
            )
        elif sample_rate < self.resample_rate:
            raise RuntimeError(
                "Sample rate is smaller than desired sample rate. No upsampling possible here."
            )

        label = torch.tensor(self.audio_data[idx, 3])
        sample = {self.key: audio, "label": label}
        return sample


class CustomDatasetDetailed(CustomDataset):
    """Create a data loader for custom paths.

    This class wraps the CustomDataset class to override __getitem__ and return a more detailed sample.
    """

    def __getitem__(self, idx: int) -> dict:
        """Get a dataset element.

        Args:
            idx (int): The element index of the data pair to return.

        Returns:
            sample (dict): Returns a dictionary with the self.key default ("audio") and "label" key
            and the corresponding data path in "path" and frame offset in "offset" and the number
            of frames in "num_frames".

        Raises:
            RuntimeError: If sample rate is smaller than desired resample rate.
        """
        audio, sample_rate = torchaudio.load(
            self.audio_data[idx, 0],
            frame_offset=self.audio_data[idx, 1] * self.audio_data[idx, 2],
            num_frames=self.audio_data[idx, 2],
        )

        if sample_rate > self.resample_rate:
            audio = torchaudio.functional.resample(
                audio, sample_rate, self.resample_rate
            )
        elif sample_rate < self.resample_rate:
            raise RuntimeError(
                "Sample rate is smaller than desired sample rate. No upsampling possible here."
            )

        label = torch.tensor(self.audio_data[idx, 3])
        sample = {
            self.key: audio,
            "label": label,
            "path": self.audio_data[idx, 0],
            "offset": self.audio_data[idx, 1],
            "num_frames": self.audio_data[idx, 2],
        }
        return sample


def get_costum_dataset(
    data_path: str,
    save_path: str,
    ds_type: str,
    only_test_folders: Optional[list] = None,
    only_use: Optional[list] = None,
    seconds: float = 1,
    resample_rate: int = 22050,
    limit: int = 55504,
    abort_on_save: bool = False,
    asvspoof_name: str | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    file_type: str = "wav",
    get_details: bool = False,
) -> CustomDataset:
    """Wrap custom dataset creation.

    Args:
        data_path (str): Path to the actual data folders.
        save_path (str): Path to the desired destination folder where the metadata will be saved.
        ds_type (str): Description of dataset purpose. Should be one of: "train", "val", "test".
        only_test_folders (Optional[list]): Names of folders which should only be used
                                                      for the test set. Defaults to None.
        only_use (Optional[list]): Names of folders which will be used, all else will
                                             be ignored. Defaults to None.
        seconds (float): Length of one frame of one data sample. Defaults to 1.
        resample_rate (int): Desired sample rate of audio to be fed into training procedure.
                                       This is stored alongside the filename and length and will be
                                       used when loading the audio in the torch dataloader.
                                       Defaults to 22050.
        limit (int): Limit for dataset.
        abort_on_save (bool): Only save metadata to drive, don't return torch dataset.
                                        Defaults to False.
        asvspoof_name (str): Name of asvspoof data split, e.g. DF_E. Defaults to None.
        train_ratio (float): Percentage of train files in dataset as decimal.
                                       Defaults to 0.7.
        val_ratio (float): Percentage of validation files in dataset as decimal.
                                     Defaults to 0.1.
        file_type (str): File type of all audio files in the folders. Defaults to "wav".
        get_details (bool): If __getitem__ should return details about the audio file, the frame
                            count and the offset.

    Raises:
        RuntimeError: If the given `data_path` does not contain subfolders.
        RuntimeError: If 0 is not in the labels when `ds_type` is "train".

    Returns:
        CustomDataset: Torch custom dataset which can be passed into a torch dataloader instance.
    """
    paths = list(Path(data_path).glob("./*_*"))
    if len(paths) == 0:
        raise RuntimeError("Given data_path is empty.")

    labels: list = []
    str_paths = []

    for path in paths:
        if (
            only_use is not None
            and str(path).split("/")[-1].split("_")[-1] not in only_use
        ):
            continue
        desired_label = ord(path.name.split("_")[0]) - 65
        if desired_label in labels:
            for i in range(len(labels)):
                new_des_label = desired_label + i + 1
                if new_des_label in labels:
                    continue
                else:
                    desired_label = new_des_label
                    break
        labels.append(desired_label)
        str_paths.append(str(path))

    if 0 not in labels and ds_type == "train":
        raise RuntimeError("No real training data. Aborting...")

    if get_details:
        return CustomDatasetDetailed(
            paths=str_paths,
            labels=labels,
            save_path=save_path,
            abort_on_save=abort_on_save,
            seconds=seconds,
            resample_rate=resample_rate,
            verbose=False,
            limit=limit,
            ds_type=ds_type,
            only_test_folders=only_test_folders,
            asvspoof_name=asvspoof_name,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            filetype=file_type,
        )
    else:
        return CustomDataset(
            paths=str_paths,
            labels=labels,
            save_path=save_path,
            abort_on_save=abort_on_save,
            seconds=seconds,
            resample_rate=resample_rate,
            verbose=False,
            limit=limit,
            ds_type=ds_type,
            only_test_folders=only_test_folders,
            asvspoof_name=asvspoof_name,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            filetype=file_type,
        )
