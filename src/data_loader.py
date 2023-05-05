"""Data loading utility for WaveFake audio classification.

Holds a custom torch Dataset class implementation that prepares the audio
and cuts it to frames and transforms it with CWT. Dataloader methods
can be found here to.
"""
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


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
        print("Loading ", data_dir, flush=True)
        if len(self.file_lst) == 0:
            raise ValueError("empty directory")
        if self.file_lst[-1].name != "labels.npy":
            raise ValueError("unexpected file name")
        self.labels = np.load(self.file_lst[-1])
        self.audios = np.array(self.file_lst[:-1])
        self.key = key

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
