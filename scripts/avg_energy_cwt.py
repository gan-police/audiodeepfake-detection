"""Energy analysis of WaveFake dataset of Frank et. al, 2021, using CWT instead of STFT.

Port of: https://github.com/RUB-SysSec/WaveFake/blob/main/statistics.py
"""
# TODO: save results to json to be able to change plotting
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ptwt
import pywt
import tikzplotlib as tikz
import torch

DEBUG = True
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if DEBUG:
    # Set python path automatically to base directory
    sys.path.append(BASE_PATH)

import src.util as util

RES = 129


def _compute_average_frequency_for_directory(
    directory: str, early_exit=None
) -> torch.Tensor:
    dataset = dataset = util.AudioDataset(
        directory,
        sample_rate=util.SAMPLE_RATE,
    )

    average_per_file = []

    sampling_period = 1.0 / util.SAMPLE_RATE
    center_freq = 0.87
    bandwith = 0.001

    wavelet = f"shan{bandwith}-{center_freq}"
    # wavelet = "fbsp3-0.01-1.0"
    nyquist_freq = util.SAMPLE_RATE / 2.0  # maximum frequency that can be analyzed

    # equally spaced normalized frequencies to be analyzed
    freqs = np.linspace(nyquist_freq, 1, RES) / util.SAMPLE_RATE

    scales = pywt.frequency2scale(wavelet, freqs)

    for i, (clip, _fs) in enumerate(dataset):
        sig, freq = ptwt.cwt(clip, scales, wavelet, sampling_period=sampling_period)
        specgram = sig.squeeze(1)

        avg = torch.mean(torch.abs(specgram) ** 2, dim=1)

        avg_db = 10.0 * torch.log(avg + 10e-13)
        average_per_file.append(avg_db)

        if i % 10 == 0:
            print(f"\rProcessed {i:06} files!\n", end="")

        if i == early_exit:
            break

    average_per_file = torch.stack(average_per_file)
    average_per_file = torch.mean(average_per_file, dim=0)

    # flip all -> reverse
    average_per_file = torch.flip(average_per_file, dims=(-1,))
    freq = np.flipud(freq)

    return average_per_file, freq


def _apply_ax_styling(
    ax, title, freqs, y_min=-150.0, y_max=40, ylabel="Durchschnittliche Energie (dB)"
) -> None:
    """Style the bar plot axes accordingly."""
    ax.set_title(title, fontsize=20, fontweight="bold")
    ax.set_ylim(y_min, y_max)

    ticks = np.linspace(0, RES, 11)
    tiks = np.linspace(freqs[0] / 1000, freqs[-1] / 1000, len(ticks))

    ticklabels = [round(item) for item in tiks]

    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)

    ax.set_xlabel("Frequenz (kHz)", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)


def plot_difference(
    data, title, ref_data, ref_title, path, freqs, absolute: bool = False
) -> None:
    """Plot prosody difference between data and reference data (e.g. real vs. fake)."""
    fig, axis = plt.subplots(1, 1)

    num_freqs = freqs.shape[0]

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
            freqs,
            y_min=0,
            y_max=10,
        )
    else:
        _apply_ax_styling(
            ax,
            f"Relative Differenz -- {title}",
            freqs,
            y_min=-10,
            y_max=10,
        )

    save_path = f"{BASE_PATH}/plots/energy/cwt/{path}.tex"

    tikz.save(
        save_path,
        encoding="utf-8",
        standalone=True,
        override_externals=True,
    )


if __name__ == "__main__":
    reference_data = None
    reference_name = None
    Path(f"{BASE_PATH}/plots/energy/cwt").mkdir(parents=True, exist_ok=True)

    amount = 5000

    data_base_dir = f"{BASE_PATH}/tests/data"
    paths = [
        "real/",
        "ljspeech_waveglow/",
    ]

    # Important: Put corresponding data directories here!
    data_base_dir = "/home/kons/uni/bachelor_thesis/git/data/"
    paths = ["LJSpeech-1.1/wavs/", "generated_audio/ljspeech_waveglow/"]

    fig_names = ["Original", "WaveGlow"]

    for i in range(len(paths)):
        print("\n======================================")
        print(f"Processing {paths[i]}!")
        print("======================================")
        average_freq, freqs = _compute_average_frequency_for_directory(
            f"{data_base_dir}/{paths[i]}", amount
        )

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
                freqs,
            )
            plot_difference(
                average_freq,
                fig_names[i],
                reference_data,
                reference_name,
                f"{fig_names[i].lower().strip()}_difference_absolute",
                freqs,
                absolute=True,
            )
