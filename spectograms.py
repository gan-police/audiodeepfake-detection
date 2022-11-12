"""Reproduce spectrogram plots in Frank, Schönherr (2021): WaveFake: A Data Set to Facilitate
Audio Deepfake Detection [FS21]

Simple script that reproduces spectrograms in the paper that show apperent differences between
original audio files and the corresponding audio samples that [FS21] reproduced with different
Deep Learning methods and architectures that already exist, e.g. MelGAN, Waveglow, Hifi-GAN.

This script is mainly inspired by:
https://github.com/pytorch/tutorials/blob/master/beginner_source/audio_feature_extractions_tutorial.py

"""

from typing import Tuple, Union

import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as tf

from os import path as pth
import tikzplotlib as tikz
from pathlib import Path


# LJSpeech-1.1 specific audio file format
SAMPLE_RATE = 22050
NUM_CHANNELS = 1
BITS_PER_SAMPLE = 16
ENCODING = "PCM_S"


def load_from_wav(path: str,
                  start_frame: int = 0, end_frame: int = -1,
                  normalize: bool = True):
    """Load signal waveform and meta data from *.wav file. With no normalization
    it does not return float32 as by default in torchaudio.load (see torchaudio.backend).
    For comparable results the audio file is tested, if it has the same format as
    files in LJSpeech-1.1.

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

    meta = torchaudio.info(path)    # contains: sample_rate, num_frames, num_channels, bits_per_sample, encoding

    # Test if audio file is of comparable format as wavs in LJSpeech-1.1
    is_correct_format = (
        meta.sample_rate == SAMPLE_RATE and
        meta.num_channels == NUM_CHANNELS and
        meta.bits_per_sample == BITS_PER_SAMPLE and
        meta.encoding == ENCODING
    )
    if not is_correct_format:
        raise IOError("Audio file is not in the same format as LJSpeech-1.1 Dataset.")

    # framerate_in_sec = meta.num_frames / meta.sample_rate
    # print("Total length in seconds: ", framerate_in_sec)
    # print("Frames: ", meta.num_frames)

    waveform, sample_rate = torchaudio.load(path, normalize=normalize)  # returns torch tensor

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


def compute_spectogram(path: str,
                       from_frame: int = 0, to_frame: int = -1,
                       n_fft: int = 1024) -> Tuple[torch.Tensor, int]:
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
    hop_length = None   # 512
    power = 2.0

    spec_transform = tf.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,  # default: n_fft
        hop_length=hop_length,  # default: win_length // 2
        power=power,
    )

    return spec_transform(waveform), waveform.shape[0]


def plot_spectrogram(spec: torch.Tensor, max_frame: int, start_frame: int = 0, end_frame: int = -1,
                     title="Spektrogramm", fig_name="sample", in_khz: bool = True,
                     cmap: Union[str or ~matplotlib.colors.Colormap] = "plasma",
                     aspect: Union[str or float] = 'auto',
                     rect_plot: bool = False):
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
                        under standalone_plots/{fig_name}-spectrogram.tex. (Optional)
        in_khz (bool): True if y-axis should be in kHz. False results in Hz. Default is kHz. (Optional)
        cmap (str or `~matplotlib.colors.Colormap`): The Colormap instance or registered colormap name
                        used to map scalar data to colors. This parameter is ignored for RGB(A) data.
        aspect ({'equal', 'auto'} or float): The aspect ratio of the Axes.  This parameter is
                particularly relevant for images since it determines whether data pixels are square.
        rect_plot (bool): If True a rectangular plot is given, otherwise it will be square. (Optional)

    Returns:
        None: Plots the corresponding spectrogram and saving it externally. No return value.

    """
    fig, axes = plt.subplots(1, 1)
    fig.set_dpi(100)
    fig.set_size_inches(10, 4, forward=True)
    axes.set_title(title or 'Spektrogram (db)')
    axes.set_xlabel('Zeit (sek)')

    # frequency bins to frequency in Hz
    bin_to_freq = np.fft.fftfreq((spec.shape[0] - 1) * 2, 1 / SAMPLE_RATE)[:spec.shape[0] - 1]
    if in_khz:
        ylabel = "Frequenz (kHz)"
        bin_to_freq /= 1000
    else:
        ylabel = "Frequenz (Hz)"

    if end_frame == -1:
        end_frame = max_frame - 1

    extent = [start_frame/SAMPLE_RATE, end_frame/SAMPLE_RATE, bin_to_freq[0], bin_to_freq[-1]]
    axes.set_ylabel(ylabel)
    vmin = -50  # have to be the same in all plots for comparability -> used same as [FS21]
    vmax = 50

    spec_np = spec.numpy()

    # if preferred: approx. same colormap as [FS21]
    # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#066b7f", "white", "#aa3a03"])
    # cmap = "RdYlBu_r"     # another nice colormap
    im = axes.imshow(librosa.power_to_db(spec_np), extent=extent,
                     cmap=cmap, origin='lower', aspect=aspect, vmin=vmin, vmax=vmax)

    fig.colorbar(im, ax=axes)
    # plt.show(block=False)

    print(f"saving {fig_name}-spectrogram.tex")
    Path("standalone_plots").mkdir(parents=True, exist_ok=True)
    tikz.clean_figure()

    # for rectangular plots
    if rect_plot:
        fig_width, fig_height = "5in", "2.5in"
    else:
        fig_width, fig_height = None, None

    tikz.save(f"standalone_plots/{fig_name}-spectrogram.tex", encoding="utf-8",
              standalone=True, axis_width=fig_width, axis_height=fig_height)


def main():
    real_audio = "../data/LJspeech-1.1/wavs/LJ008-0217.wav"
    fake_audios = [
        "../data/generated_audio/ljspeech_melgan/LJ008-0217_gen.wav",
        "../data/generated_audio/ljspeech_full_band_melgan/LJ008-0217_gen.wav",
        "../data/generated_audio/ljspeech_multi_band_melgan/LJ008-0217_gen.wav",
        "../data/generated_audio/ljspeech_hifiGAN/LJ008-0217_generated.wav",
        "../data/generated_audio/ljspeech_waveglow/LJ008-0217.wav",
        "../data/generated_audio/ljspeech_parallel_wavegan/LJ008-0217_gen.wav",
    ]

    titles_fake = [
        "MelGAN",
        "Full-Band-MelGAN",
        "Multi-Band-MelGAN",
        "Hifi-GAN",
        "Waveglow",
        "Parallel WaveGAN"
    ]

    fig_names = [
        "melgan",
        "fb-melgan",
        "mb-melgan",
        "hifigan",
        "waveglow",
        "parallel-wavegan"
    ]

    from_frame = 0
    to_frame = -1  # -1 for last

    n_fft = 1024    # Frank et al. use 256 in statistics.py...

    rect_plot = True
    # BA specific
    # rect_plot = False

    print("Plotting Spektrograms of LJ008 0217.wav")
    spec, frames = compute_spectogram(real_audio, from_frame, to_frame, n_fft)

    plot_spectrogram(spec, frames, from_frame, to_frame, title='Original', fig_name="original",
                     rect_plot=rect_plot)

    for i in range(len(fake_audios)):
        spec, frames = compute_spectogram(fake_audios[i], from_frame, to_frame, n_fft)
        plot_spectrogram(spec, frames, from_frame, to_frame,
                         title=titles_fake[i], fig_name=fig_names[i],
                         rect_plot=rect_plot)


if __name__ == "__main__":
    main()
