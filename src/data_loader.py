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
from torch.utils.data import Dataset


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
        key: Optional[str] = "audio",
        verbose: Optional[bool] = True,
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
        self.key = key

        self.label_names = {0: "original", get_ds_label(self.labels): "fake"}

    def _load_mean_std(self):
        with open(self.data_dir + "/mean_std.pkl", "rb") as f:
            return pickle.load(f)

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
        verbose: Optional[bool] = True,
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
                if limit > 0 and limit < init_size // 2:
                    required_samples = limit
                else:
                    required_samples = init_size // 2
                real_count = 0
            elif init_size != len(dataset):
                raise RuntimeError(
                    f"{folder} contains to little samples."
                )
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
