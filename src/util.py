"""Audio File and Spectrogramm utility.

Means to reproduce plots of Frank, Schönherr (2021): WaveFake: A Data Set to Facilitate
Audio Deepfake Detection [FS21].

Simple script that reproduces spectrograms in the paper that show apperent differences between
original audio files and the corresponding audio samples that [FS21] generated with different
Deep Learning methods and architectures that already exist, e.g. MelGAN, Waveglow, Hifi-GAN.

This script is mainly inspired by:
https://github.com/pytorch/tutorials/blob/master/beginner_source/audio_feature_extractions_tutorial.py

"""

import os
from os import path as pth
from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ptwt
import tikzplotlib as tikz
import torch
import torchaudio
import torchaudio.transforms as tf

from src.cwt import CWT

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# LJSpeech-1.1 specific audio file format
SAMPLE_RATE = 22050
NUM_CHANNELS = 1
BITS_PER_SAMPLE = 16
ENCODING = "PCM_S"
SOX_SILENCE = [
    # trim all silence that is longer than 0.2s and louder than 1% volume (relative to the file)
    # from beginning and middle/end
    ["silence", "1", "0.2", "1%", "-1", "0.2", "1%"],
]


class AudioDataset(torch.utils.data.Dataset):
    """Torch dataset to load data from a provided directory.

    Port of: https://github.com/RUB-SysSec/WaveFake/blob/main/dfadetect/datasets.py
    Args:
        directory_or_path_list: Path to the directory containing wav files to load. Or a list of paths.
    Raises:
        IOError: If the directory does ot exists or the directory did not contain any wav files.
    """

    def __init__(
        self,
        directory_or_path_list: Union[str, Path, list],
        sample_rate: int = 16_000,
        amount: Optional[int] = None,
        normalize: bool = True,
    ) -> None:
        """Initialize Audioloader.

        Raises:
            IOError: If directory does not exist, or does not contain wav files.
            TypeError: If given path is weird.
        """
        super().__init__()

        self.sample_rate = sample_rate
        self.normalize = normalize

        paths = []
        if (
            isinstance(directory_or_path_list, Path)
            or isinstance(directory_or_path_list, str)
            or isinstance(directory_or_path_list, list)
        ):
            if isinstance(directory_or_path_list, list):
                for path in directory_or_path_list:
                    directory = Path(path)
                    if not directory.exists():
                        raise IOError(f"Directory does not exists: {directory}")
                    path_list = find_wav_files(directory)
                    paths.append(path_list)
                    if path_list is None:
                        raise IOError(
                            f"Directory did not contain wav files: {directory}"
                        )
            else:
                directory = Path(directory_or_path_list)
                if not directory.exists():
                    raise IOError(f"Directory does not exists: {directory}")

                paths.append(find_wav_files(directory))
                if path_list is None:
                    raise IOError(f"Directory did not contain wav files: {directory}")
        else:
            raise TypeError(
                f"Supplied unsupported type for argument directory_or_path_list {type(directory_or_path_list)}!"
            )

        if amount is not None:
            paths = paths[:amount]

        self._paths = paths

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Load signal from .wav."""
        path = self._paths[index]

        waveform, sample_rate = torchaudio.load(path, normalize=self.normalize)

        if sample_rate != self.sample_rate:
            waveform, sample_rate = torchaudio.sox_effects.apply_effects_file(
                path, [["rate", f"{self.sample_rate}"]], normalize=self.normalize
            )
        return waveform, sample_rate

    def __len__(self) -> int:
        """Length of path list."""
        return len(self._paths)


class TransformDataset(torch.utils.data.Dataset):
    """Torch dataset to load data from a provided directory.

    Port of: https://github.com/RUB-SysSec/WaveFake/blob/main/dfadetect/datasets.py

    Windowing happens as the file is loaded, because torchaudio.load can load file sample-wise. This
    might lead to border effects as soon as cwt transform is applied afterwards.
    """

    def __init__(
        self,
        directory_or_path_list: Union[str, Path, list],
        device: Optional[str] = "cpu",
        sample_rate: float = 16000.0,
        max_length: int = 64600,
        amount: Optional[int] = None,
        normalize: bool = True,
        resolution: int = 50,
        lfcc_filter: Optional[int] = 40,
        f_min: float = 80.0,
        f_max: float = 1000.0,
        transform: str = "cwt",
        mean: Optional[float] = None,
        std: Optional[float] = None,
        frame_size: int = 1024,
        wavelet: str = "cmor0.5-1.0",
        from_path: int = 0,
        to_path: int = 10,
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
            lfcc_filter (optional, int): Number of lfccs that will be computed.
            f_min (float): Smallest frequency that will be analyzed.
            f_max (float): Biggest frequency that will be analyzed.
            transform (str): Transformation that will be used. Can be 'cwt' or 'lfcc'.
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

        if transform == "lfcc":
            n_fft = frame_size
            hop_length = 512

            self.transform = tf.LFCC(
                sample_rate=self.sample_rate,
                n_lfcc=lfcc_filter,
                speckwargs={
                    "n_fft": n_fft,
                    "hop_length": hop_length,
                },
            )
        elif transform == "cwt":
            self.transform = CWT(
                sample_rate=self.sample_rate,
                n_lin=resolution,
                f_min=f_min,
                f_max=f_max,
                wavelet=wavelet,
            )
        else:
            self.transform = None

        paths = []
        if from_path >= to_path:
            from_path = 0
            to_path = 10
        if (
            isinstance(directory_or_path_list, Path)
            or isinstance(directory_or_path_list, str)
            or isinstance(directory_or_path_list, list)
        ):
            if isinstance(directory_or_path_list, list):
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
            if waveform.shape[0] < offset + self.frame_size:
                offset = 0  # if audio file is smaller than max_length samples
            waveform = waveform[:, offset : offset + self.frame_size]
        else:
            # only load small window of audio -> faster than slicing afterwards and more convenient
            waveform, _sample_rate = torchaudio.load(
                path,
                normalize=self.normalize,
                num_frames=self.frame_size,
                frame_offset=offset,
            )

        """
        # cut all silences > 0.2s
        waveform_trimmed, sample_rate_trimmed = torchaudio.sox_effects.apply_effects_tensor(
            waveform, sample_rate, SOX_SILENCE
        )

        if waveform_trimmed.size()[1] > 0:
            waveform = waveform_trimmed
            sample_rate = sample_rate_trimmed
        """
        if self.transform:
            waveform = self.transform(waveform)

        # normalize data
        if self.mean is not None:
            waveform = (waveform - self.mean) / self.std

        # a bit hardcoded sorry
        path_str = str(path)
        if "generated" in path_str or "gen" in path_str:
            label = torch.tensor(1.0, dtype=torch.int64)
        else:
            label = torch.tensor(0.0, dtype=torch.int64)

        return waveform.to(self.device), label.to(self.device)

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
            self.count = torch.zeros(1, device=batch_vals.device, dtype=torch.float64)
            self.mean = torch.zeros(
                batch_vals.shape[-1], device=batch_vals.device, dtype=torch.float64
            )
            self.std = torch.zeros(
                batch_vals.shape[-1], device=batch_vals.device, dtype=torch.float64
            )
            self.m2 = torch.zeros(
                batch_vals.shape[-1], device=batch_vals.device, dtype=torch.float64
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


def load_from_wav(
    path: str, start_frame: int = 0, end_frame: int = -1, normalize: bool = True
) -> torch.Tensor:
    """Load signal waveform and meta data from *.wav file.

    With no normalization it does not return float32 as by default in torchaudio.load
    (see torchaudio.backend). For comparable results the audio file is tested, if it
    has the same format as files in LJSpeech-1.1.

    Args:
        path (str): The path to .wav audio file.
        start_frame (int): Start frame index of part of audio wav sample that is of interest.
                           Default is 0. (Optional)
        end_frame (int): End frame index of part of audio wav sample that the spectrogram shows.
                         Default is last frame. (Optional)
        normalize (bool): Normalized signal. Then waveform is float32 in [-1.0,1.0] (Default). (Optinal)

    Raises:
        IOError: If audio file does not have the same specs as the ones in LJSpeech-1.1.
        FileExistsError: If given path is not a file or does not exist.
        ValueError: If file at path is not a mono audio signal (if it has to many channels)

    Returns:
        torch.Tensor: Waveform of file as tensor at specified path.
    """
    if not pth.isfile(path) or not pth.exists(path):
        raise FileExistsError("File Path leads nowhere reasonable: ", path)

    # get sample_rate, num_frames, num_channels, bits_per_sample, encoding
    meta = torchaudio.info(path)

    # Test if audio file is of comparable format as wavs in LJSpeech-1.1
    is_correct_format = (
        meta.sample_rate == SAMPLE_RATE
        and meta.num_channels == NUM_CHANNELS
        and meta.bits_per_sample == BITS_PER_SAMPLE
        and meta.encoding == ENCODING
    )
    if not is_correct_format:
        raise IOError("Audio file is not in the same format as LJSpeech-1.1 Dataset.")

    # framerate_in_sec = meta.num_frames / meta.sample_rate
    # print("Total length in seconds: ", framerate_in_sec)
    # print("Frames: ", meta.num_frames)

    waveform, sample_rate = torchaudio.load(
        path, normalize=normalize
    )  # returns torch tensor

    if meta.num_channels == 1:
        if start_frame >= meta.num_frames:
            start_frame = 0
            print("Frame start param too high. Set to first frame.")
        if end_frame >= meta.num_frames:
            end_frame = meta.num_frames - 1
            print("Frame end param too high. Set to last frame.")
        if end_frame == -1:
            end_frame = meta.num_frames  # set to last frame

        # cut waveform to given window
        waveform = waveform[0][start_frame:end_frame]
    else:
        raise ValueError("To many channels in data. Should be 1-D Audio, no stereo.")

    return waveform


def compute_spectogram(
    path: str, from_frame: int = 0, to_frame: int = -1, n_fft: int = 1024
) -> Tuple[torch.Tensor, int]:
    """Compute spectrogram of file given at path.

    Uses torchaudio implementation of spectrogram generation with torch.stft. See
    https://github.com/pytorch/audio/blob/main/torchaudio/functional/functional.py
    and
    https://github.com/pytorch/pytorch/blob/master/torch/functional.py

    Args:
        path (str): The path to .wav audio file.
        from_frame (int): Start frame index of part of audio wav sample that is of interest.
                           Default is 0. (Optional)
        to_frame (int): End frame index of part of audio wav sample that the spectrogram shows.
                         Default is last frame. (Optional)
        n_fft (int): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``1024``)

    Returns:
        Tuple[torch.Tensor, int]: The tensor of the frequency power matrix and the frame number
                                  of initial audio file (mainly so plotting is easier).
    """
    waveform = load_from_wav(path, from_frame, to_frame)

    win_length = None
    hop_length = None  # 512
    power = 2.0

    spec_transform = tf.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,  # default: n_fft
        hop_length=hop_length,  # default: win_length // 2
        power=power,
    )

    return spec_transform(waveform), waveform.shape[0]


def compute_cwt(
    path: str,
    wavelet: str,
    scales: np.ndarray,
    from_frame: int = 0,
    to_frame: int = -1,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Compute cwt of audio file given at path.

    Args:
        path (str): The path to .wav audio file.
        wavelet (str): Mother wavelet of the cwt. Must be from pywt.families(), additionally with
                        own center frequency and bandwidth.
        scales (np.ndarray): Scales under which the mother wavelet is to be dilated in the cwt.
        from_frame (int): Start frame index of part of audio wav sample that is of interest.
                           Default is 0. (Optional)
        to_frame (int): End frame index of part of audio wav sample that the spectrogram shows.
                         Default is last frame. (Optional)

    Returns:
        Tuple[torch.Tensor, np.ndarray]: The tensor of the scales power matrix and the frequencies
                                            that correspond to the scales.
    """
    signal = get_np_signal(path, from_frame, to_frame)

    sampling_period = 1.0 / SAMPLE_RATE
    cwt_out = ptwt.cwt(
        torch.from_numpy(signal), scales, wavelet, sampling_period=sampling_period
    )

    return cwt_out


def plot_spectrogram(
    spec: torch.Tensor,
    max_frame: int,
    start_frame: int = 0,
    end_frame: int = -1,
    title="Spektrogramm",
    fig_name="sample",
    in_khz: bool = True,
    cmap: Union[str, matplotlib.colors.Colormap] = "plasma",
    aspect: Union[str, float] = "auto",
    rect_plot: bool = False,
) -> None:
    """
    Plot spectrogram to given spectral matrix according to [FS21].

    Different plotting options possible. Saving via tikz to .tex.

    Args:
        spec (torch.Tensor): Input tensor containing frequency powers corresponding to frequency bins
                             of shape [window_length // 2, freqency bins]
        max_frame (int): The initial number of frames before transforming with stft, for plotting time.
        start_frame (int): Start frame index of part of audio wav sample that the spectrogram shows.
                           Default is 0. (Optional)
        end_frame (int): End frame index of part of audio wav sample that the spectrogram shows.
                         Default is last frame. (Optional)
        title (str): Title on top of the plot. (Optinal)
        fig_name (str): Title prefix of the file that is generated from the plot. It will be saved
                        under plots/{fig_name}-spectrogram.tex. (Optional)
        in_khz (bool): True if y-axis should be in kHz. False results in Hz. Default is kHz. (Optional)
        cmap (str or `matplotlib.colors.Colormap`): The Colormap instance or registered colormap name
                        used to map scalar data to colors. This parameter is ignored for RGB(A) data.
        aspect ({'equal', 'auto'} or float): The aspect ratio of the Axes.  This parameter is
                particularly relevant for images since it determines whether data pixels are square.
        rect_plot (bool): If True a rectangular plot is given, otherwise it will be square. (Optional)
    """
    fig, axes = plt.subplots(1, 1)
    fig.set_dpi(100)
    axes.set_title(title or "Spektrogram (db)")
    axes.set_xlabel("Zeit (sek)")

    # frequency bins to frequency in Hz
    bin_to_freq = np.fft.fftfreq((spec.shape[0] - 1) * 2, 1 / SAMPLE_RATE)[
        : spec.shape[0] - 1
    ]
    if in_khz:
        ylabel = "Frequenz (kHz)"
        bin_to_freq /= 1000
    else:
        ylabel = "Frequenz (Hz)"

    if end_frame == -1:
        end_frame = max_frame - 1

    extent = [
        start_frame / SAMPLE_RATE,
        end_frame / SAMPLE_RATE,
        bin_to_freq[0],
        bin_to_freq[-1],
    ]
    axes.set_ylabel(ylabel)
    vmin = (
        -50.0
    )  # have to be the same in all plots for comparability -> used same as [FS21]
    vmax = 50.0

    spec_np = spec.numpy()

    # TODO: Rather use: torchaudio.transforms.AmplitudeToDB
    # if preferred: approx. same colormap as [FS21]
    # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#066b7f", "white", "#aa3a03"])
    # cmap = "RdYlBu_r"     # another nice colormap
    cmap = "turbo"
    im = axes.imshow(
        librosa.power_to_db(spec_np),
        extent=extent,
        cmap=cmap,
        origin="lower",
        aspect=aspect,
        vmin=vmin,
        vmax=vmax,
    )

    cb = fig.colorbar(im, ax=axes, label="dB")

    print(f"saving {fig_name}-spectrogram.tex")
    Path(f"{BASE_PATH}/stft/gfx/tikz").mkdir(parents=True, exist_ok=True)

    # for rectangular plots
    if rect_plot:
        fig_width, fig_height = "5in", "2.5in"
        fig.set_size_inches(10, 4, forward=True)
    else:
        fig_width, fig_height = None, None

    save_path = f"{BASE_PATH}/plots/stft/{fig_name}-spectrogram-small.tex"
    tikz_path = "gfx/tikz"

    tikz.save(
        save_path,
        encoding="utf-8",
        standalone=True,
        axis_width=fig_width,
        axis_height=fig_height,
        tex_relative_path_to_data=tikz_path,
        override_externals=True,
    )

    axes.set_title("")
    # axes.set_axis_off()
    cb.remove()  # remove colorbar
    plt.savefig(
        f"{BASE_PATH}/plots/stft/gfx/tikz/{fig_name}-spectrogram-small-000.png",
        bbox_inches="tight",
        pad_inches=0,
    )
    # TODO: introduce parameter vmin, vmax, plot_show


def plot_scalogram(
    scal,
    start_frame: int = 0,
    end_frame: int = -1,
    title: str = "Skalogramm",
    fig_name: str = "sample",
    rect_plot: bool = True,
) -> None:
    """
    Plot scaleogram to given scale-time matrix of a time-dependent signal.

    Different plotting options possible. Saving via tikz to .tex.

    Args:
        scal (Tuple[torch.Tensor, np.ndarray]): Input tensor containing frequency powers corresponding to
                                                scales in cwt. Corresponding frequencies in Hz in np.ndarray.
        start_frame (int): Start frame index of part of audio wav sample that the scaleogram shows.
                           Default is 0. (Optional)
        end_frame (int): End frame index of part of audio wav sample that the scaleogram shows.
                         Default is last frame. (Optional)
        title (str): Title on top of the plot. (Optinal)
        fig_name (str): Title prefix of the file that is generated from the plot. It will be saved
                        under plots/{fig_name}-scaleogram.tex. (Optional)
        rect_plot (bool): If True a rectangular plot is given, otherwise it will be square. (Optional)
    """
    coeff_pt, freqs = scal
    coeff = coeff_pt.numpy()

    freqs /= 1000  # because plot is in kHz not in Hz

    fig, axs = plt.subplots(1, 1)
    cmap = "turbo"
    coeff_db = librosa.power_to_db(np.abs(coeff) ** 2)

    vmin = -20
    vmax = -80
    extent = [start_frame / SAMPLE_RATE, end_frame / SAMPLE_RATE, freqs[-1], freqs[0]]

    im = axs.imshow(
        coeff_db,
        cmap=cmap,
        aspect="auto",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )

    # for rectangular plots
    if rect_plot:
        fig_width, fig_height = "5in", "2.5in"
        fig.set_size_inches(10, 4, forward=True)
    else:
        fig_width, fig_height = None, None

    axs.set_title(title)
    axs.set_xlabel("Zeit (sek)")
    axs.set_ylabel("Frequenz (kHz)")
    cb = fig.colorbar(im, ax=axs, label="dB")
    axs.set_yscale("asinh")

    print(f"saving {fig_name}-scalogram.tex")
    Path("plots/cwt/gfx/tikz").mkdir(parents=True, exist_ok=True)
    tikz.save(
        f"{BASE_PATH}/plots/cwt/{fig_name}-scalogram.tex",
        encoding="utf-8",
        standalone=True,
        axis_width=fig_width,
        axis_height=fig_height,
        tex_relative_path_to_data="gfx/tikz",
        override_externals=True,
    )

    # workaround for smaller images
    axs.set_title("")
    # axs.set_axis_off()
    cb.remove()  # remove colorbar
    plt.savefig(
        f"{BASE_PATH}/plots/cwt/gfx/tikz/{fig_name}-scalogram-000.png",
        bbox_inches="tight",
        pad_inches=0,
    )
    # TODO: introduce parameter vmin, vmax, cmap, yscale, plot_show


def get_np_signal(path: str, start_frame: int, to_frame: int) -> np.ndarray:
    """
    Get normalized signal from wav file at path as Numpy-Array. Amplitude in [-1.0, 1.0]. This is just some utility.

    Args:
        path (str): The path to .wav audio file.
        start_frame (int): Start frame index of part of audio wav sample that is of interest.
                           Default is 0. (Optional)
        to_frame (int): End frame index of part of audio wav sample that the spectrogram shows.
                         Default is last frame. (Optional)

    Returns:
        np.array: Array like signal in [-1., 1.].
    """
    sig: torch.Tensor = load_from_wav(path, start_frame, to_frame, normalize=True)
    sig_np: np.ndarray = sig.numpy()
    # t: np.ndarray = np.linspace(0, sig.shape[0] / SAMPLE_RATE, 20, False)
    return sig_np
