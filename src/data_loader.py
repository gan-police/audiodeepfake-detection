"""Data loading utility for WaveFake audio classification.

Holds a custom torch Dataset class implementation that prepares the audio
and cuts it to frames and transforms it with CWT. Dataloader methods
can be found here to.
"""
import functools
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from src.old.cwt import CWT
from src.old.old_train_classifier import LOGGER

from .ptwt_continuous_transform import (
    _ComplexMorletWavelet,
    _DifferentiableContinuousWavelet,
    _frequency2scale,
)
from .wavelet_math import wavelet_direct

SOX_SILENCE = [
    # trim all silence that is longer than 0.2s and louder than 1% volume (relative to the file)
    # from beginning and middle/end
    ["silence", "1", "0.1", "1%", "-1", "0.1", "1%"],
]


class TransformDataset(torch.utils.data.Dataset):
    """Torch dataset to load data from a provided directory.

    Port of: https://github.com/RUB-SysSec/WaveFake/blob/main/dfadetect/datasets.py

    Windowing happens as the file is loaded, because torchaudio.load can load file sample-wise. This
    might lead to border effects as soon as cwt transform is applied afterwards.
    """

    def __init__(
        self,
        directory_or_path_list: Union[str, Path, list],
        label: int = 0,
        device: Optional[str] = "cpu",
        sample_rate: float = 16000.0,
        max_length: int = 64600,
        amount: Optional[int] = None,
        normalize: bool = True,
        resolution: int = 50,
        f_min: float = 80.0,
        f_max: float = 1000.0,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        frame_size: int = 1024,
        wavelet: str = "cmor0.5-1.0",
        from_path: int = 0,
        to_path: int = 10,
        channels: int = 1,
        out_classes: int = 2,
    ) -> None:
        """Initialize Audioloader.

        For each given all .wav file paths are put into a list and iterated. The first max_length
        samples of each file is cut into frames of length frame_size until the maximum given amount
        of training samples is reached. If (to_path - from_path) * (max_length // frame_size) is very
        small then it might happen that only on audio file is enough to fill up all the training samples.

        Args:
            directory_or_path_list: Path to the directory containing wav files to load. Or a list of paths.
            device (optional, str): Device on which tensors shall be stored, can be cpu or gpu.
            sample_rate (float): Sample rate that audio signal shall have. If it sample rate is not the
                            same as the sample rate of the loaded signal, it will be down- or upsampled.
            max_length (int): Maximum number of samples that will be loaded from each audio file.
            amount (optinal, int): Maximum number of paths being considerd when searching folders for audios.
            normalize (bool): True if audio signal should be loaded with normalized amplitude (in [-1,1]).
            resolution (int): Number of scales of CWT that are computed -> first dimension of input tensor.
            f_min (float): Smallest frequency that will be analyzed.
            f_max (float): Biggest frequency that will be analyzed.
            mean (optional, float): Mean of dataset.
            std (optional, float): Standard deviation of dataset.
            frame_size (int): Number of samples per frame -> second dimension of input tensor.

        Raises:
            IOError: If directory does not exist, or does not contain wav files.
            TypeError: If given path is weird.
        """
        super().__init__()

        self.sample_rate = sample_rate
        self.normalize = normalize
        self.device = device

        self.max_length = max_length
        self.resolution = resolution
        self.mean = mean
        self.std = std
        self.frame_size = frame_size
        self.frames_per_file = self.max_length // self.frame_size

        self.channels = channels
        self.label: int = label

        self.transform = CWT(
            sample_rate=self.sample_rate,
            n_lin=resolution,
            f_min=f_min,
            f_max=f_max,
            wavelet=wavelet,
        )

        self.binary_classification = True
        if out_classes > 2:
            self.binary_classification = False

        paths = []
        self.labels = {}
        if from_path >= to_path:
            from_path = 0
            to_path = 10
        if (
            isinstance(directory_or_path_list, Path)
            or isinstance(directory_or_path_list, str)
            or isinstance(directory_or_path_list, list)
        ):
            if (
                isinstance(directory_or_path_list, list)
                and len(directory_or_path_list) > 1
            ):
                dir_num = len(directory_or_path_list)
                amount = to_path - from_path
                ind = amount // dir_num
                if ind == 0:
                    ind = 1
                for p in range(dir_num):
                    directory = Path(directory_or_path_list[p])
                    if not directory.exists():
                        raise IOError(f"Directory does not exists: {directory}")
                    path_list = find_wav_files(directory)
                    if path_list is None:
                        raise IOError(
                            f"Directory did not contain wav files: {directory}"
                        )

                    if from_path is not None and to_path is not None:
                        # take equally spread number of audios from each directory
                        # assuming that all directories shall be taken into account equally
                        if p == dir_num - 1 and amount > 1:
                            ind += amount % dir_num
                        path_list = path_list[from_path : from_path + ind]

                    paths.extend(path_list)
                    self.labels[directory.parts[-1]] = p
            else:
                if isinstance(directory_or_path_list, list):  # must be of size 1
                    directory = Path(directory_or_path_list[0])
                else:
                    directory = Path(directory_or_path_list)
                if not directory.exists():
                    raise IOError(f"Directory does not exists: {directory}")

                paths.extend(find_wav_files(directory))
                if paths is None:
                    raise IOError(f"Directory did not contain wav files: {directory}")
                if from_path is not None and to_path is not None:
                    paths = paths[from_path:to_path]
        else:
            raise TypeError(
                f"Supplied unsupported type for argument directory_or_path_list {type(directory_or_path_list)}!"
            )

        path_offsets = []
        path_list = []
        for path in paths:
            for i in range(self.frames_per_file):
                # assuming every audio file has max. max_length samples
                path_list.append(path)
                path_offsets.append(i * self.frame_size)
        self._path_list = path_list
        self._path_offsets = path_offsets
        LOGGER.info(f"Set with Label: {self.label}")
        # LOGGER.info(self._path_list)
        # LOGGER.info(self._path_offsets)
        LOGGER.info(f"length of set: {len(self._path_list)}")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load signal from .wav.

        Audio files are loaded in frames of given frame size. They will be down- or upsampled if sample rate
        differs from given sample rate. Afterwards the specified transform will be applied.
        """
        path = self._path_list[index]
        offset = self._path_offsets[index]

        meta = torchaudio.info(path)
        if meta.num_frames < offset + self.frame_size:
            offset = 0  # if audio file is smaller than max_length samples

        if meta.sample_rate != self.sample_rate:
            waveform, _sample_rate = torchaudio.sox_effects.apply_effects_file(
                path,
                [["rate", f"{self.sample_rate}"]],
                normalize=self.normalize,
            )
            # cut silences longer than 0.2s
            waveform, _sample_rate = torchaudio.sox_effects.apply_effects_tensor(
                waveform, int(self.sample_rate), SOX_SILENCE
            )

            if self.device == "cuda":
                waveform = waveform.cuda()
            waveform = waveform[:, offset : offset + self.frame_size]
        else:
            # only load small window of audio -> faster than slicing afterwards and more convenient
            waveform, _sample_rate = torchaudio.load(
                path,
                normalize=self.normalize,
                num_frames=self.frame_size,
                frame_offset=offset,
            )
            if self.device == "cuda":
                waveform = waveform.cuda()

        if self.transform:
            waveform = self.transform(waveform)

        # normalize data
        if self.mean is not None:
            waveform = (waveform - self.mean) / self.std

        shape = waveform.shape
        if self.channels > shape[0]:
            padding = torch.zeros(
                self.channels - shape[0], shape[1], shape[2], device=self.device
            )
            for i in range(self.channels - shape[0]):
                padding[i] = waveform  # all channels have the same data
            waveform = torch.cat((waveform, padding), 0)

        label = get_label(
            self, path=path, binary_classification=self.binary_classification
        )
        label_t = torch.tensor(int(label), device=self.device)

        return waveform, label_t

    def __len__(self) -> int:
        """Length of path list."""
        return len(self._path_list)

    def set_mean_std(self, mean, std) -> None:
        """Setter for mean and standard deviation."""
        self.mean = mean
        self.std = std


def find_wav_files(path_to_dir: Union[Path, str]) -> list[Path]:
    """Find all wav files in the directory and its subtree.

    Port of: https://github.com/RUB-SysSec/WaveFake/blob/main/dfadetect/utils.py
    Args:
        path_to_dir: Path top directory.
    Returns:
        List containing Path objects or None (nothing found).
    """
    paths = list(sorted(Path(path_to_dir).glob("**/*.wav")))

    return paths


def get_label(self, path: Path, binary_classification: bool) -> int:
    """Get the label of the audios in a folder based on the folder path.

    We assume:
        wavs: Orignal data,
        all others: different gans
    A working folder structure could look like:
        wavs ljspeech_melgan ljspeech_melgan_large ljspeech_multi_band_melgan
        ljspeech_full_band_melgan ljspeech_waveglow ljspeech_parallel_wavegan
        ljspeech_hifiGAN jsut_parallel_wavegan jsut_multi_band_melgan
        common_voices_prompts_from_conformer_fastspeech2_pwg_ljspeech
    With each folder containing the audios from the corresponding source.

    Raises:
        NotImplementedError: If folder name is not implemented as label.
        ValueError: If in binary classification self.label is not set correctly.
    """
    if binary_classification:
        if self.label is None:
            raise ValueError(
                "self.label is None and out_classes is 2. Set label to specific label."
            )
        return self.label
    else:
        # the the label based on the path, As are 0s, Bs are 1, etc.
        label_str = path.parts[-2]
        if label_str in self.labels:
            label = self.labels[label_str]
        else:
            raise NotImplementedError(label_str)
        return label


def get_frames_list(
    path_to_dir: Union[Path, str], amount: Optional[int] = None, frame_size: int = 224
) -> tuple[list, list]:
    """Return list of all paths in given directory with frame offset list.

    Raises:
        ValueError: If frame_size is to low.
    """
    paths = find_wav_files(path_to_dir)
    if amount:
        paths = paths[:amount]

    if frame_size <= 0:
        raise ValueError("Frame_size must be positive.")

    path_num_frames = []
    path_offsets = []
    path_list = []
    for i in range(len(paths)):
        frames = torchaudio.info(paths[i]).num_frames
        path_num_frames.append(frames)
        for j in range(frames // frame_size):
            # assuming every audio file has max. max_length samples
            path_list.append(paths[i])
            path_offsets.append(j * frame_size)

    return path_list, path_offsets


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


def get_mean_std_welford(train_data_set) -> tuple[float, float]:
    """Calculate mean and standard deviation of dataset."""
    LOGGER.info("Calculating mean and standard deviation...")
    welford = WelfordEstimator()
    for aud_no in range(train_data_set.__len__()):
        welford.update(train_data_set.__getitem__(aud_no)[0])
    mean, std = welford.finalize()

    return mean.mean().item(), std.mean().item()


def prepare_dataloaders(
    args,
    device,
    fake_data_folder,
    real_data_folder,
    fake_test_data_folder,
    amount,
    input_channels,
    f_max,
    f_min,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return normalized training, validation and test dataloaders."""
    train_data_real, train_data_fake, train_data = load_dist_data(
        args,
        device,
        fake_data_folder,
        real_data_folder,
        input_channels,
        f_max,
        f_min,
        from_path=0,
        to_path=amount,
    )

    test_size = int(amount * 0.2)  # 20% of total training samples
    from_idx = amount + 1
    to_idx = from_idx + test_size

    val_data_real, val_data_fake, val_data = load_dist_data(
        args,
        device,
        fake_data_folder,
        real_data_folder,
        input_channels,
        f_max,
        f_min,
        from_path=from_idx,
        to_path=to_idx,
    )

    test_data_real, test_data_fake, test_data = load_dist_data(
        args,
        device,
        fake_test_data_folder,
        real_data_folder,
        input_channels,
        f_max,
        f_min,
        from_path=to_idx + 1,
        to_path=to_idx + 1 + test_size,
    )

    LOGGER.info(f"Train set length: {len(train_data)}")
    LOGGER.info(f"Val set length: {len(val_data)}")
    LOGGER.info(f"Test set length: {len(test_data)}")

    if args.mean and args.std:
        mean, std = args.mean, args.std
    else:
        mean, std = get_mean_std_welford(
            train_data
        )  # calculate mean, std only on train data
    LOGGER.info(f"Mean: {mean}, Std: {std}")
    train_data_real.set_mean_std(mean, std)
    train_data_fake.set_mean_std(mean, std)
    val_data_real.set_mean_std(mean, std)
    val_data_fake.set_mean_std(mean, std)
    test_data_real.set_mean_std(mean, std)
    test_data_fake.set_mean_std(mean, std)

    trn_dl = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )
    val_dl = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    test_dl = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    return trn_dl, val_dl, test_dl


def load_dist_data(
    args,
    device,
    fake_data_folder,
    real_data_folder,
    input_channels,
    f_max,
    f_min,
    from_path,
    to_path,
) -> tuple[
    TransformDataset,
    TransformDataset,
    torch.utils.data.ConcatDataset | TransformDataset,
]:
    """Return ConcatDataset that holds 50% fake and 50% real audio files at given paths.

    Raises:
        Warning: If length of real dataset is uneqal to length of fake dataset.
    """
    if args.out_classes == 2:
        from_path = from_path // 2
        to_path = to_path // 2
        data_real = TransformDataset(
            real_data_folder,
            device=device,
            sample_rate=args.sample_rate,
            max_length=args.max_length,
            frame_size=args.frame_size,
            f_min=f_min,
            f_max=f_max,
            resolution=args.scales,
            wavelet=args.wavelet,
            from_path=from_path,
            to_path=to_path,
            channels=input_channels,
            out_classes=args.out_classes,
            label=1,
        )
    data_fake = TransformDataset(
        fake_data_folder,
        device=device,
        sample_rate=args.sample_rate,
        max_length=args.max_length,
        frame_size=args.frame_size,
        f_min=f_min,
        f_max=f_max,
        resolution=args.scales,
        wavelet=args.wavelet,
        from_path=from_path,
        to_path=to_path,
        channels=input_channels,
        out_classes=args.out_classes,
        label=0,
    )
    if args.out_classes == 2:
        if len(data_real) != len(data_fake):
            raise Warning("Data not distributed equally bewteen fake and real.")

        train_data: torch.utils.data.ConcatDataset = torch.utils.data.ConcatDataset(
            [data_real, data_fake]
        )

        return data_real, data_fake, train_data
    else:
        return data_fake, data_fake, data_fake


class NumpyDataset(Dataset):
    """Create a data loader to load pre-processed numpy arrays into memory."""

    def __init__(
        self,
        data_dir: str,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        key: Optional[str] = "audio",
    ):
        """Create a Numpy-dataset object.

        Args:
            data_dir: A path to a pre-processed folder with numpy files.
            mean: Pre-computed mean to normalize with. Defaults to None.
            std: Pre-computed standard deviation to normalize with. Defaults to None.
            key: The key for the input or 'x' component of the dataset.
                Defaults to "audio".

        Raises:
            ValueError: If an unexpected file name is given or directory is empty.

        # noqa: DAR401
        """
        self.data_dir = data_dir
        self.file_lst = sorted(Path(data_dir).glob("./*.npy"))
        print("Loading ", data_dir)
        if len(self.file_lst) == 0:
            raise ValueError("empty directory")
        if self.file_lst[-1].name != "labels.npy":
            raise ValueError("unexpected file name")
        self.labels = np.load(self.file_lst[-1])
        self.audios = self.file_lst[:-1]
        # self.labels = self.labels.repeat(len(self.audios) // self.labels.shape[0])
        self.mean = mean
        self.std = std
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

        # normalize the data.
        if self.mean is not None:
            audio = (audio - self.mean) / self.std

        label = self.labels[idx]
        label = torch.tensor(int(label))
        sample = {self.key: audio, "label": label}
        return sample


class CombinedDataset(Dataset):
    """Load data from multiple Numpy-Data sets using a singe object."""

    def __init__(self, sets: list):
        """Create an merged dataset, combining many numpy datasets.

        Args:
            sets (list): A list of NumpyDataset objects.
        """
        self.sets = sets
        self.len = len(sets[0])
        # assert not any(self.len != len(s) for s in sets)

    @property
    def key(self) -> list:
        """Return the keys for all features in this dataset."""
        return [d.key for d in self.sets]

    def __len__(self) -> int:
        """Return the data set length."""
        return self.len

    def __getitem__(self, idx: int) -> dict:
        """Get a dataset element.

        Args:
            idx (int): The element index of the data pair to return.

        Returns:
            sample (dict): Returns a dictionary with the self.key
                           default ("audio") and "label" keys.
                           The key property will return a keylist.
        """
        label_list = [s.__getitem__(idx)["label"] for s in self.sets]
        # the labels should all be the same
        # assert not any([label_list[0] != l for l in label_list])
        label = label_list[0]
        dict = {set.key: set.__getitem__(idx)[set.key] for set in self.sets}
        dict["label"] = label
        return dict


class WavefakeDataset(Dataset):
    """Create a data loader to load pre-processed numpy arrays into memory."""

    def __init__(
        self,
        data_dir: str,
        wavelet: Optional[_DifferentiableContinuousWavelet],
        mean: Optional[float] = None,
        std: Optional[float] = None,
        key: Optional[str] = "audio",
        sample_rate: int = 8000,
        f_min: float = 2000,
        f_max: float = 4000,
        num_of_scales: int = 224,
    ):
        """Create a Wavefake-dataset object.

        Dataset with additional transforming in cwt space.

        Args:
            data_dir: A path to a pre-processed folder with numpy files.
            mean: Pre-computed mean to normalize with. Defaults to None.
            std: Pre-computed standard deviation to normalize with. Defaults to None.
            key: The key for the input or 'x' component of the dataset.
                Defaults to "audio".

        Raises:
            ValueError: If an unexpected file name is given or directory is empty.

        # noqa: DAR401
        """
        self.data_dir = data_dir
        self.file_lst = sorted(Path(data_dir).glob("./*.npy"))
        print("Loading ", data_dir)
        if len(self.file_lst) == 0:
            raise ValueError("empty directory")
        if self.file_lst[-1].name != "labels.npy":
            raise ValueError("unexpected file name")
        self.labels = np.load(self.file_lst[-1])
        self.audios = np.array(self.file_lst[:-1])
        self.mean = mean
        self.std = std
        self.key = key

        freqs = torch.linspace(f_max, f_min, num_of_scales, device="cuda") / sample_rate
        scales = _frequency2scale(wavelet, freqs).detach()
        # scales = pywt.frequency2scale("cmor4.6-0.87", freqs)
        self.transform = functools.partial(
            wavelet_direct,
            wavelet=wavelet,
            scales=scales,
            cuda=True,
            log_scale=True,
        )

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

        # jit_cwt = torch.jit.trace(_to_jit_cwt, (audio), strict=False)
        audio = self.transform(audio)

        # normalize the data.
        if self.mean is not None:
            audio = (audio - self.mean) / self.std

        label = self.labels[idx]
        label = torch.tensor(int(label))
        sample = {self.key: audio, "label": label}
        return sample


class LearnWavefakeDataset(Dataset):
    """Create a data loader to load pre-processed numpy arrays into memory."""

    def __init__(
        self,
        data_dir: str,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        key: Optional[str] = "audio",
    ):
        """Create a Wavefake-dataset object.

        Dataset with additional transforming in cwt space.

        Args:
            data_dir: A path to a pre-processed folder with numpy files.
            mean: Pre-computed mean to normalize with. Defaults to None.
            std: Pre-computed standard deviation to normalize with. Defaults to None.
            key: The key for the input or 'x' component of the dataset.
                Defaults to "audio".

        Raises:
            ValueError: If an unexpected file name is given or directory is empty.

        # noqa: DAR401
        """
        self.data_dir = data_dir
        self.file_lst = sorted(Path(data_dir).glob("./*.npy"))
        print("Loading ", data_dir)
        if len(self.file_lst) == 0:
            raise ValueError("empty directory")
        if self.file_lst[-1].name != "labels.npy":
            raise ValueError("unexpected file name")
        self.labels = np.load(self.file_lst[-1])
        self.audios = np.array(self.file_lst[:-1])
        self.mean = mean
        self.std = std
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

        # normalize the data.
        if self.mean is not None:
            audio = (audio - self.mean) / self.std

        label = self.labels[idx]
        label = torch.tensor(int(label))
        sample = {self.key: audio, "label": label}
        return sample


class LearnDirectDataset(Dataset):
    """Create a data loader to load pre-processed numpy arrays into memory."""

    def __init__(
        self,
        data_set: np.ndarray,
        labels: np.ndarray,
        window_size: int = 11025,
        sample_rate: int = 22050,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        key: Optional[str] = "audio",
    ):
        """Create a Wavefake-dataset object.

        Dataset with additional transforming in cwt space.

        Args:
            data_set: np.ndarray paths to wav files.
            labels: np.ndarray labels corresponding to these paths.
            window_size (int): Size of window that audios are cute to.
            sample_rate (int): Sample rate of output audio for get_item in Hz.
            mean: Pre-computed mean to normalize with. Defaults to None.
            std: Pre-computed standard deviation to normalize with. Defaults to None.
            key: The key for the input or 'x' component of the dataset. Defaults to "audio".
        """
        self.labels = labels
        self.audios = data_set[0]
        self.audio_frames = data_set[1]
        self.window_size = int(window_size)
        self.sample_rate = int(sample_rate)
        self.mean = mean
        self.std = std
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

        audio, sample_rate = torchaudio.load(
            audio_path,
            normalize=True,
            num_frames=self.window_size,
            frame_offset=self.audio_frames[idx],
        )
        # resample with better window (a bit slower than default hann window)
        if int(sample_rate) != self.sample_rate:
            audio = torchaudio.functional.resample(
                audio, sample_rate, self.sample_rate, resampling_method="kaiser_window"
            )
        import pdb

        pdb.set_trace()
        # normalize the data.
        if self.mean is not None:
            audio = (audio - self.mean) / self.std

        label = self.labels[idx]
        label = torch.tensor(int(label))
        sample = {self.key: audio, "label": label}
        return sample


def _to_jit_cwt(sig):
    sample_rate = 8000
    f_max = 4000
    f_min = 2000
    num_of_scales = 224
    wavelet = "cmor4.6-0.87"
    if not isinstance(wavelet, _DifferentiableContinuousWavelet):
        wavelet = _ComplexMorletWavelet(wavelet)
    freqs = torch.linspace(f_max, f_min, num_of_scales, device="cuda") / sample_rate
    scales = _frequency2scale(wavelet, freqs).detach()
    # scales = pywt.frequency2scale("cmor4.6-0.87", freqs)
    return wavelet_direct(
        sig, scales=scales, wavelet=wavelet, log_scale=True, cuda=True
    )
