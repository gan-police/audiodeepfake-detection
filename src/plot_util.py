"""Audio File and Spectrogramm utility for plotting.

Means to reproduce plots of Frank, Schönherr (2021): WaveFake: A Data Set to Facilitate
Audio Deepfake Detection [FS21].

Src that reproduces spectrograms in the paper that show apperent differences between
original audio files and the corresponding audio samples that [FS21] generated with different
Deep Learning methods and architectures that already exist, e.g. MelGAN, Waveglow, Hifi-GAN.
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

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# LJSpeech-1.1 specific audio file format
SAMPLE_RATE = 22050
NUM_CHANNELS = 1
BITS_PER_SAMPLE = 16
ENCODING = "PCM_S"


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

                paths = find_wav_files(directory)  # type: ignore
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

    waveform, _sample_rate = torchaudio.load(
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
    path: str,
    from_frame: int = 0,
    to_frame: int = -1,
    n_fft: int = 1023,
    win_length=1024,
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

    power = 2.0

    spec_transform = tf.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=1,
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
    Path("./plots/stft/gfx/tikz").mkdir(parents=True, exist_ok=True)

    # for rectangular plots
    if rect_plot:
        fig_width, fig_height = "5in", "2.5in"
        fig.set_size_inches(10, 4, forward=True)
    else:
        fig_width, fig_height = None, None

    save_path = f"./plots/stft/{fig_name}-spectrogram-small.tex"
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
    axes.set_axis_off()
    cb.remove()  # remove colorbar
    plt.savefig(
        f"./plots/stft/gfx/tikz/{fig_name}-spectrogram-small-000.png",
        bbox_inches="tight",
        pad_inches=0,
    )


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
    # axs.set_yscale("asinh")

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
    axs.set_axis_off()
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

"""Caculate integrated gradients of trained models."""
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import tikzplotlib as tikz
from matplotlib import colors

class Mean:
    """Compute running mean."""

    def __init__(self) -> None:
        """Create a meaner."""
        self.init: Optional[bool] = None

    def update(self, batch_vals: torch.Tensor) -> None:
        """Update the running estimation.

        Args:
            batch_vals (torch.Tensor): The current batch element.
        """
        if self.init is None:
            self.init = True
            self.count = 0
            self.mean = torch.zeros(
                batch_vals.shape, device=batch_vals.device, dtype=torch.float32
            )
            self.std = torch.zeros(
                batch_vals.shape, device=batch_vals.device, dtype=torch.float32
            )
            self.m2 = torch.zeros(
                batch_vals.shape, device=batch_vals.device, dtype=torch.float32
            )
        self.count += 1
        self.mean += batch_vals.detach()

    def finalize(self) -> torch.Tensor:
        """Finish the estimation and return the computed mean and std.

        Returns:
            torch.Tensor: Estimated mean.
        """
        return torch.mean(self.mean, dim=0).squeeze() / self.count

def bar_plot(data, x_ticks, x_labels, path):
    """Plot histogram of model attribution."""
    _fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)

    axs.set_xticks(x_ticks)
    axs.set_xticklabels(x_labels)
    axs.set_xlabel("frequency [kHz]")

    axs.bar(
        x=list(range(data.shape[0])),
        height=data,
        color="crimson",
    )
    save_plot(path)


def im_plot(
    data,
    path,
    cmap,
    x_ticks,
    x_labels,
    y_ticks,
    y_labels,
    vmin=None,
    vmax=None,
    norm=None,
):
    """Plot image of model attribution."""
    fig, axs = plt.subplots(1, 1)
    im = axs.imshow(np.flipud(data), aspect="auto", norm=norm, cmap=cmap)
    axs.set_xlabel("time [sec]")
    axs.set_ylabel("frequency [kHz]")
    axs.set_xticks(x_ticks)
    axs.set_xticklabels(x_labels)
    axs.set_yticks(y_ticks)
    axs.set_yticklabels(y_labels)
    fig.colorbar(im, ax=axs)
    fig.set_dpi(200)
    axs.invert_yaxis()

    save_plot(path)


def save_plot(path):
    """Save plt as standalone latex document."""
    #plt.savefig(path)
    tikz.save(
        f"{path}.tex",
        encoding="utf-8",
        standalone=True,
        tex_relative_path_to_data="images",
        override_externals=True,
    )

def interpolate_images(baseline, image, alphas):
    alphas_x = alphas.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    baseline_x = torch.unsqueeze(baseline, 0)
    input_x = torch.unsqueeze(image, 0)
    delta = input_x - baseline_x
    images = baseline_x +  alphas_x * delta
    return images

def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / 2.0
    integrated_gradients = torch.mean(grads, dim=0)
    return integrated_gradients


def plot_img_attributions(
    image,
    attribution_mask,
    x_ticks,
    y_ticks,
    x_labels,
    y_labels,
    cmap=None,
    overlay_alpha=0.4,
    norm=None
):
    fig, axs = plt.subplots(nrows=1, ncols=3, squeeze=False, figsize=(8, 8))

    axs[0, 0].set_title('Original image')
    axs[0, 0].imshow(np.flipud(image), aspect="auto")
    # axs[0, 0].axis('off')

    axs[0, 1].set_title('Integrated Gradients')
    axs[0, 1].imshow(np.flipud(attribution_mask), aspect="auto",  cmap=cmap)
    axs[0, 2].set_title('Overlay')
    axs[0, 2].imshow(np.flipud(attribution_mask), aspect="auto",  cmap=cmap)
    axs[0, 2].imshow(np.flipud(image), aspect="auto", alpha=overlay_alpha)

    axs[0,0].set_xlabel("time [sec]")
    axs[0,1].set_xlabel("time [sec]")
    axs[0,2].set_xlabel("time [sec]")
    axs[0,0].set_ylabel("frequency [kHz]")

    axs[0,0].set_xticks(x_ticks)
    axs[0,1].set_xticks(x_ticks)
    axs[0,2].set_xticks(x_ticks)
    axs[0,0].set_xticklabels(x_labels)
    axs[0,1].set_xticklabels(x_labels)
    axs[0,2].set_xticklabels(x_labels)
    axs[0,0].set_yticks(y_ticks)
    axs[0,1].set_yticks(y_ticks)
    axs[0,2].set_yticks(y_ticks)
    axs[0,0].set_yticklabels(y_labels)
    axs[0,1].set_yticklabels(y_labels)
    axs[0,2].set_yticklabels(y_labels)
    axs[0,0].invert_yaxis()
    axs[0,1].invert_yaxis()
    axs[0,2].invert_yaxis()

    plt.tight_layout()
    return fig


def plot_attribution_targets(seconds, sample_rate, num_of_scales, path, ig_0, ig_1, ig_01, norm=None):
    t = np.linspace(0, seconds, int(seconds // (1 / sample_rate)))
    bins = np.int64(num_of_scales)
    n = list(range(int(bins)))
    freqs = (sample_rate / 2) * (n / bins)  # type: ignore

    x_ticks = list(range(ig_0.shape[-1]))[:: ig_0.shape[-1] // 4]
    x_labels = np.around(np.linspace(min(t), max(t), ig_0.shape[-1]), 2)[
                                :: ig_0.shape[-1] // 4
                            ]

    y_ticks = n[:: freqs.shape[0] // 6]
    y_labels = np.around(freqs[:: freqs.shape[0] // 6] / 1000, 1)
                    

    cmap=plt.cm.inferno   
    fig, axs = plt.subplots(nrows=1, ncols=3, squeeze=False, figsize=(8, 8))

    axs[0, 0].set_title('Attribution on Real Neuron')
    def sign_log_norm(data):
        # Shift the data to make it positive and then apply a logarithmic scale
        data_shifted = data - np.min(data) + 1e-10  # Add a small value to avoid log(0)
        mean = data_shifted.mean() 
        orig_data = np.log(data_shifted / abs(mean))
        return data * 3
    
    v_min = -ig_1.max()
    v_max = ig_1.max()
    im = axs[0, 0].imshow(np.flipud(sign_log_norm(ig_0)), aspect="auto", cmap=cmap, vmin=v_min, vmax=v_max)
    

    axs[0, 1].set_title('Attribution on Fake Neuron')
    im = axs[0, 1].imshow(np.flipud(sign_log_norm(ig_1)), aspect="auto", cmap=cmap, vmin=v_min, vmax=v_max)

    axs[0, 2].set_title('Attribution Real and Fake')
    v_min = -ig_1.max()
    v_max = ig_1.max()
    im = axs[0, 2].imshow(np.flipud(sign_log_norm(ig_01)), aspect="auto", cmap=cmap, vmin=v_min, vmax=v_max)

    fig.colorbar(im, ax=axs)
    axs[0,0].set_xlabel("time [sec]")
    axs[0,1].set_xlabel("time [sec]")
    axs[0,2].set_xlabel("time [sec]")
    axs[0,0].set_ylabel("frequency [kHz]")

    axs[0,0].set_xticks(x_ticks)
    axs[0,1].set_xticks(x_ticks)
    axs[0,2].set_xticks(x_ticks)
    axs[0,0].set_xticklabels(x_labels)
    axs[0,1].set_xticklabels(x_labels)
    axs[0,2].set_xticklabels(x_labels)
    axs[0,0].set_yticks(y_ticks)
    axs[0,1].set_yticks(y_ticks)
    axs[0,2].set_yticks(y_ticks)
    axs[0,0].set_yticklabels(y_labels)
    axs[0,1].set_yticklabels(y_labels)
    axs[0,2].set_yticklabels(y_labels)
    axs[0,0].invert_yaxis()
    axs[0,1].invert_yaxis()
    axs[0,2].invert_yaxis()

    save_plot(path + "_integrated_gradients")
    plt.show()

if __name__ == "__main__":
    transformations = ["packets", "stft"]
    wavelets = ["sym8"]
    cross_sources = [
        "melgan-lmelgan-mbmelgan-pwg-waveglow-avocodo-hifigan-conformer-jsutmbmelgan-jsutpwg-lbigvgan-bigvgan",
    ]
    targets = ["0", "1", "01"]

    norm = [
    colors.Normalize(0.84-0.02*2, 0.84+2*0.02),
    colors.Normalize(0.82-0.016*2, 0.82+2*0.016)
    ]
    i = 0

    seconds = 1
    sample_rate = 22050
    num_of_scales = 256
    for transformation in transformations:
        for wavelet in wavelets:
            for cross_source in cross_sources:
                path = f"/home/kons/work/Plots/files/plots/{transformation}_{sample_rate}_{seconds}_0_fbmelgan_{wavelet}_2.0_False_ljspeech-{cross_source}x2500_target"
                if os.path.exists(path + "-0_integrated_gradients.npy") and os.path.exists(path + "-1_integrated_gradients.npy") and os.path.exists(path + "-01_integrated_gradients.npy"):
                    ig_0 = np.load(path + "-0_integrated_gradients.npy")
                    ig_1 = np.load(path + "-1_integrated_gradients.npy")
                    ig_01 = np.load(path + "-01_integrated_gradients.npy")
                else:
                    continue
                if os.path.exists(path + "-0_ig_max.npy"):
                    ig_max = np.load(path + "-0_ig_max.npy")
                else:
                    continue
                if os.path.exists(path + "-0_ig_min.npy"):
                    ig_min = np.load(path + "-0_ig_min.npy")
                else:
                    continue
                if os.path.exists(path + "-0_ig_abs.npy"):
                    ig_abs = np.load(path + "-0_ig_abs.npy")
                else:
                    continue
                
                print(np.min(ig_0), np.max(ig_0))
                plot_attribution_targets(seconds, sample_rate, num_of_scales, path, ig_0, ig_1, ig_01, norm[i])

                plt.close()
                i += 1
