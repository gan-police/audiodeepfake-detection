"""Energy analysis with code from wavefake, Frank et. al 2021 using STFT.

Direct port of: https://github.com/RUB-SysSec/WaveFake/blob/main/statistics.py
"""
import os
import sys
from pathlib import Path

# TODO: put avg_energy.py and this script in one.
import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib as tikz
import torch
from torchaudio.functional import detect_pitch_frequency, spectral_centroid
from torchaudio.transforms import Spectrogram

DEBUG = True
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if DEBUG:
    # Set python path automatically to base directory
    sys.path.append(BASE_PATH)

import src.audiofakedetect.plot_util as util

N_FFT = 300
RES = N_FFT // 2

# Latex font
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def _compute_average_frequency_for_directory(
    directory: str, early_exit=None, compute_stats=True
) -> torch.Tensor:
    dataset = dataset = util.AudioDataset(
        directory,
        sample_rate=util.SAMPLE_RATE,
    )
    """Compute average frequency energy in dB over different frequencies."""
    average_per_file = []
    if compute_stats:
        spectral_centroids = []
        pitches = []
        pitches_std = []

    spec_transform = Spectrogram(n_fft=N_FFT, hop_length=1)

    for i, (clip, fs) in enumerate(dataset):
        specgram = spec_transform(clip).squeeze(0)

        avg = torch.mean(specgram, dim=1)
        avg_db = 10.0 * torch.log(avg + 10e-13)
        average_per_file.append(avg_db)

        if i % 10 == 0:
            print(f"\rProcessed {i:06} files!", end="", flush=True)

        if i == early_exit:
            break

        if compute_stats:
            # compute spectral centroid
            centroid = spectral_centroid(
                waveform=clip,
                sample_rate=fs,
                # same as Spectrogram above
                pad=0,
                window=torch.hann_window(N_FFT),
                n_fft=N_FFT,
                hop_length=N_FFT // 2,
                win_length=N_FFT,
            )
            spectral_centroids.append(torch.mean(centroid))

            # pitch
            pitch = detect_pitch_frequency(clip, fs, freq_low=50, freq_high=500)
            pitches.append(torch.mean(pitch))
            pitches_std.append(torch.std(pitch))

    average_per_file = torch.stack(average_per_file)
    average_per_file = torch.mean(average_per_file, dim=0)

    return average_per_file


def _apply_ax_styling(
    ax,
    title,
    num_freqs,
    y_min=-150.0,
    y_max=40,
    ylabel="Durchschnittliche Energie (dB)",
) -> None:
    """Style axes of bar plots."""
    ax.set_title(title, fontsize=20, fontweight="bold")
    ax.set_ylim(y_min, y_max)

    # convert fftbins to freq.
    freqs = np.fft.fftfreq((num_freqs - 1) * 2, 1 / util.SAMPLE_RATE)[: num_freqs - 1]

    ticks = np.linspace(0, RES, 11)
    # ticks = ax.get_xticks()[1:]
    tiks = np.linspace(freqs[0] / 1000, freqs[-1] / 1000, len(ticks))

    ticklabels = [round(item) for item in tiks]

    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)

    ax.set_xlabel("Frequenz (kHz)", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)


def plot_barplot(data, title, path) -> None:
    """Plot average energy per frequency as bar plot."""
    fig, ax = plt.subplots()

    num_freqs = data.shape[0]

    ax.bar(x=list(range(num_freqs)), height=data, color="#2D5B68")

    _apply_ax_styling(ax, title, num_freqs)

    fig.tight_layout()
    fig.savefig(path)


def plot_difference(
    data, title, ref_data, ref_title, path, absolute: bool = False
) -> None:
    """Plot prosody difference between data and fake data."""
    fig, axis = plt.subplots(1, 1)

    num_freqs = ref_data.shape[0]

    # plot differnce
    ax = axis
    diff = data - ref_data

    if absolute:
        diff = np.abs(diff)

    ax.bar(x=list(range(num_freqs)), height=diff, color="crimson")

    if absolute:
        _apply_ax_styling(
            ax,
            f"Absolute Differenz -- {title}",
            num_freqs,
            y_min=0,
            y_max=10,
        )
    else:
        _apply_ax_styling(
            ax,
            f"Relative Differenz -- {title}",
            num_freqs,
            y_min=-10,
            y_max=10,
        )

    save_path = f"{BASE_PATH}/plots/energy/stft/{path}.tex"

    tikz.save(
        save_path,
        encoding="utf-8",
        standalone=True,
        override_externals=True,
    )


def measure_pitch(path) -> tuple[float, float]:
    """Measure pitch of dataset at given path."""
    y, sr = librosa.load(path)
    f0_curve = librosa.yin(y, fmin=50, fmax=300, sr=sr)
    avg_f0, std_f0 = np.mean(f0_curve), np.std(f0_curve)
    return avg_f0, std_f0


if __name__ == "__main__":
    reference_data = None
    reference_name = None
    stats = False
    Path("./plots/energy/cwt").mkdir(parents=True, exist_ok=True)

    amount = 13100

    # Important: Put corresponding data directories here!
    paths = [
        "../data/real/A_ljspeech/",
        "../data/fake/B_melgan/",
        "../data/fake/C_hifigan/",
        "../data/fake/D_mbmelgan/",
        "../data/fake/E_fbmelgan/",
        "../data/fake/F_waveglow/",
        "../data/fake/G_pwg/",
        "../data/fake/H_lmelgan/",
    ]

    fig_names = [
        "Original",
        "MelGAN",
        "HiFi-GAN",
        "Multi-Band-MelGAN",
        "Full-Band-MelGAN",
        "Waveglow",
        "Parallel-Wavegan",
        "Large-MelGAN",
    ]

    for i in range(len(paths)):
        print("\n======================================")
        print(f"Processing {paths[i]}!", flush=True)
        print("======================================")
        average_freq = _compute_average_frequency_for_directory(paths[i], amount)

        if reference_data is None:
            reference_data = average_freq
            reference_name = fig_names[i]
        else:
            plot_difference(
                average_freq,
                fig_names[i],
                reference_data,
                reference_name,
                f"{fig_names[i].lower().strip()}_difference",
            )
            plot_difference(
                average_freq,
                fig_names[i],
                reference_data,
                reference_name,
                f"{fig_names[i].lower().strip()}_difference_absolute",
                absolute=True,
            )
