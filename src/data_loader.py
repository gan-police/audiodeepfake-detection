"""Data loading utility for WaveFake audio classification.

Holds a custom torch Dataset class implementation that prepares the audio
and cuts it to frames and transforms it with CWT. Dataloader methods
can be found here to.
"""
import os
import pickle
from pathlib import Path
from typing import Optional, Tuple

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


class LearnWavefakeDataset(Dataset):
    """Create a data loader to load pre-processed numpy arrays into memory."""

    def __init__(
        self,
        data_dir: str,
        source_name: str = "fake",
        key: Optional[str] = "audio",
        limit: int = -1,
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
        self.data_dir = data_dir
        self.file_lst = sorted(Path(data_dir).glob("./*.npy"))
        if verbose:
            print("Loading ", data_dir, flush=True)
        if len(self.file_lst) == 0:
            raise ValueError("empty directory")
        if self.file_lst[-1].name != "labels.npy":
            raise ValueError("unexpected file name for label file.")

        self.labels = np.load(self.file_lst[-1])
        self.audios = np.array(self.file_lst[:-1])

        if limit == -1:
            limit = len(self.audios)
        limit -= (limit // 128) % 8  # same batches if trained with 1, 2, 4, 8 GPUs

        # keep equally distributed structure
        del_num = len(self.audios) - limit
        pos_count = del_num // 2
        neg_count = del_num // 2

        for i in range(len(self.audios)):
            if self.labels[i] == 0 and pos_count > 0:
                pos_count -= 1
                self.labels = np.delete(self.labels, i)
                self.audios = np.delete(self.audios, i)
            elif self.labels[i] != 0 and neg_count > 0:
                neg_count -= 1
                self.labels = np.delete(self.labels, i)
                self.audios = np.delete(self.audios, i)
            elif pos_count == 0 and neg_count == 0:
                break

        self.labels = self.labels[:limit]
        self.audios = self.audios[:limit]

        sizes = [self.labels[self.labels == 0].size, self.labels[self.labels != 0].size]
        diff = abs(sizes[0] - sizes[1])
        label = np.argmax(sizes)
        for i in range(len(self.audios)):
            if self.labels[i] == 0 and diff > 0 and label == 0:
                diff -= 1
                self.labels = np.delete(self.labels, i)
                self.audios = np.delete(self.audios, i)
            elif self.labels[i] != 0 and diff > 0 and label != 0:
                diff -= 1
                self.labels = np.delete(self.labels, i)
                self.audios = np.delete(self.audios, i)
            elif diff == 0:
                break

        self.key = key
        self.label_names = {0: "original", get_ds_label(self.labels): source_name}

    def _load_mean_std(self):
        with open(self.data_dir + "/mean_std.pkl", "rb") as f:
            return pickle.load(f)

    def get_label_name(self, key):
        if key in self.label_names.keys():
            return self.label_names[key]
        else:
            return f"John Doe Generator {key}"

    def __len__(self) -> int:
        """Return the data set length."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        """Get a dataset element.

        Args:
            idx (int): The element index of the data pair to return.

        Returns:
            sample (dict): Returns a dictionary with the self.key
                    default ("audio") and "label" keys.
        """
        audio_path = self.audios[idx]
        audio = np.load(audio_path)
        audio = torch.from_numpy(audio.astype(np.float32))

        label = self.labels[idx]
        label = torch.tensor(int(label))
        sample = {self.key: audio, "label": label}
        return sample


class CrossWavefakeDataset(Dataset):
    """Create a data loader to load pre-processed numpy arrays into memory."""

    def __init__(
        self,
        sources: list,
        base_path: str = "/home/s6kogase/data/run6",
        postfix: str = "_test",
        prefix: str = "fake_22050_22050_0.7_",
        limit: int = -1,
        key: Optional[str] = "audio",
        verbose: Optional[bool] = False,
    ):
        """Create a Wavefake-dataset object.

        Dataset with additional transforming in cwt space.

        Args:
            key: The key for the input or 'x' component of the dataset.
                Defaults to "audio".

        Raises:
            RuntimeError: If there are problems with the given data paths.

        # noqa: DAR401
        """
        if verbose:
            print("Loading cross dataset...", flush=True)
        legit_folders = [f"{prefix}{source}{postfix}" for source in sources]
        init_size = None
        required_num = 0
        real_done = False
        unique_labels = None
        real_count = 0

        labels = []
        paths = []

        for folder in os.listdir(base_path):
            if folder not in legit_folders:
                continue
            required_num += 1

        label_source_names = {}
        label_source_names[0] = "original"

        def get_free_label(unique_labels):
            if len(unique_labels) == 0:
                raise RuntimeError(
                    "Function call only possible on arrays with positive length."
                )

            if 0 in unique_labels:
                unique_labels = np.delete(unique_labels, 0)

            unique_labels.sort()

            for i in range(len(unique_labels)):
                if i + 1 == unique_labels[i]:
                    continue
                else:
                    return np.int64(i + 1)
            return np.int64(i + 2)

        def get_source_name(folder, sources):
            for source in sources:
                if f"_{source}_" in folder:
                    return source

        for folder in os.listdir(base_path):
            if folder not in legit_folders:
                continue
            dataset = LearnWavefakeDataset(f"{base_path}/{folder}", verbose=False)
            if init_size is None:
                init_size = len(dataset)
                if limit > 0 and limit <= init_size:
                    required_samples = limit
                else:
                    required_samples = init_size
                real_count = 0
            elif init_size != len(dataset):
                raise RuntimeError(f"{folder} contains to little samples.")
            else:
                unique_labels = np.unique(np.asarray(labels))

            file_count = 0

            # make sure that there will only be unique values in dataset
            target_label = get_ds_label(dataset.labels)
            if unique_labels is not None:
                this_label = target_label
                if this_label in unique_labels:
                    target_label = get_free_label(unique_labels)
                    if verbose:
                        print(
                            f"Setting target label from {this_label} to {target_label}."
                        )

            label_source_names[target_label] = get_source_name(folder, sources)

            for path, label in zip(dataset.audios, dataset.labels):
                if real_done and file_count >= required_samples:
                    break
                if label == 0 and not real_done:
                    real_count += 1
                    paths.append(path)
                    labels.append(label)
                    if real_count == required_samples:
                        real_done = True
                elif label != 0 and file_count < required_samples:
                    paths.append(path)
                    labels.append(target_label)
                    file_count += 1

        self.labels = np.asarray(labels)
        self.audios = np.asarray(paths)

        self.label_names = label_source_names

        self.key = key

    def get_label_name(self, key):
        if key in self.label_names.keys():
            return self.label_names[key]
        else:
            return f"John Doe Generator {key}"

    def __len__(self) -> int:
        """Return the data set length."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        """Get a dataset element.

        Args:
            idx (int): The element index of the data pair to return.

        Returns:
            sample (dict): Returns a dictionary with the self.key
                    default ("audio") and "label" keys.
        """
        audio_path = self.audios[idx]
        audio = np.load(audio_path)
        audio = torch.from_numpy(audio.astype(np.float32))

        label = self.labels[idx]
        label = torch.tensor(int(label))
        sample = {self.key: audio, "label": label}
        return sample


class CustomDataset(Dataset):
    """Create a data loader for custom paths."""

    def __init__(
        self,
        paths: list,
        labels: list,
        save_path: str,
        only_test_folders: Optional[list] = [],
        abort_on_save: bool = False,
        ds_type: str = "train",
        seconds: int = 1,
        resample_rate: int = 16000,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        key: Optional[str] = "audio",
        limit: tuple = (555000, 7500, 15500),  # (train, val, test) per label
        verbose: Optional[bool] = False,
        filetype: str = "wav",
        asvspoof_name: str = None,
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
                "Reading dataset. This may take up to 45 min, so maybe grab a coffee. The result will be saved to your hard drive to speed up the dataloading in further experiments."
            )
            train_data = []
            val_data = []
            test_data = []
            min_sample_rate = (
                192000  # assuming this will be the highest sample rate ever
            )
            sample_count = []
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
                            num_train = limit[0]

                sample_count.append([num_train, num_val, num_test])
                path_num += 1

            sample_count = np.asarray(sample_count)
            min_len = sample_count.transpose().min(axis=1)

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
        import pdb; pdb.set_trace()
        result_set = result_set[:, : limit]

        # make sure, all frames will result in the same window size after resampling
        
        min_sample_rate = result_set[:, :, 2].min()
        if resample_rate > min_sample_rate:
            raise RuntimeError(
                "Sample rate is smaller than desired sample rate. No upsampling possible here."
            )

        audio_data = None
        if ds_type == "train":
            if only_test_folders is not None and len(only_test_folders) != 0:
                raise ValueError(
                    "Since there are folders in only_test_folders this cannot be a train dataset."
                )
        elif ds_type != "val" and ds_type != "test":
            raise RuntimeError("Dataset type does not exists.")

        result = result_set
        for i in range(result.shape[0]):
            if audio_data is None:
                audio_data = result[i]
            else:
                audio_data = np.vstack([audio_data, result[i]])

        self.audio_data = audio_data  # shape (number of samples, 4)
        self.ds_type = ds_type
        self.key = key
        self.resample_rate = resample_rate

    def get_result_set(self, frames, min_len):
        result = None  # will be of shape (len(paths), frame_number, individual_window_size, label)
        for frame_array in frames:
            if result is None:
                result = np.expand_dims(frame_array[:min_len], 0)
            else:
                result = np.concatenate(
                    [result, np.expand_dims(frame_array[:min_len], 0)]
                )
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


def get_costum_dataset(
    data_path: str,
    save_path: str,
    ds_type: str,
    only_test_folders: Optional[list] = None,
    only_use: Optional[list] = None,
    seconds: float = 1,
    resample_rate: int = 22050,
    limit: tuple = (55504, 7504, 15504),
    abort_on_save: bool = False,
    asvspoof_name: str = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    file_type: str = "wav"
) -> CustomDataset:
    paths = list(Path(data_path).glob(f"./*_*"))
    if len(paths) == 0:
        raise RuntimeError("Given data_path is empty.")

    labels = []
    str_paths = []
    for path in paths:
        if (
            only_use is not None
            and str(path).split("/")[-1].split("_")[-1] not in only_use
        ):
            continue
        desired_label = ord(path.name.split("_")[0]) - 65
        if len(labels) > 0 and desired_label != max(labels) + 1:
            desired_label = max(labels) + 1
        labels.append(desired_label)
        str_paths.append(str(path))

    if 0 not in labels and ds_type == "train":
        raise RuntimeError("No real training data. Aborting...")

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
        filetype=file_type
    )
