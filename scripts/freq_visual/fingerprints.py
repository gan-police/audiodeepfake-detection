import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pywt
import tikzplotlib as tikz
import torch

DEBUG = True
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if DEBUG:
    # Set python path automatically to base directory
    sys.path.append(BASE_PATH)

import src.plot_util as util
from scripts.freq_visual.intro_plot import compute_pytorch_packet_representation

RES = 150
SAMPLE_RATE = 22_050
F_MIN = 1
F_MAX = 11025


def plot_mean_std(steps, mean, std, label="", marker="."):
    """Plot means and standard deviations with shaded areas."""
    plt.plot(steps, mean, label=label, marker=marker)
    plt.fill_between(steps, mean - std, mean + std, alpha=0.2)


def _compute_fingerprint_rfft(
    directory: str,
    gen_name: str = "",
    seconds: int = 1,
    plot_path="./plots/fingerprints/",
) -> torch.Tensor:
    dataset = util.AudioDataset(
        directory,
        sample_rate=SAMPLE_RATE,
    )
    clips = []
    for clip, _fs in dataset:
        if clip.shape[-1] > seconds * SAMPLE_RATE:
            clip = clip[:, : seconds * SAMPLE_RATE]
            clips.append(clip.numpy())
    print(f"Clip no: {len(clips)}")
    clip_array = np.stack(clips)
    del clips
    freq_clips = np.fft.rfft(clip_array, axis=-1)
    freqs = freq_clips.shape[-1]
    use = freqs // 8
    zeros = np.zeros_like(freq_clips)[:, :, :-use]
    freq_clips = freq_clips[:, :, -use:]
    masked_freq = np.concatenate([zeros, freq_clips], -1)
    masked_time = np.fft.irfft(masked_freq)
    masked_time_mean = np.mean(masked_time, 0)[0]

    mean_ln_abs_fft = np.log(np.abs(np.fft.rfft(masked_time_mean)[-use:]))
    # std_ln_abs_fft = np.log(np.abs(np.fft.rfft(masked_time_std)[-use:]))
    freqs = np.fft.rfftfreq(masked_time_mean.shape[-1], 1.0 / SAMPLE_RATE)[-use:]
    # plt.subplot(2, 1, 1)
    # plt.title(f"{gen_name} - time")
    # plt.plot(masked_time_mean)
    # plt.subplot(2, 1, 2)
    plt.title(f"fingerprint - {gen_name} - ln(abs(rfft(x))))")
    plt.plot(freqs, mean_ln_abs_fft, label=gen_name)
    plt.xlabel("frequency [Hz]")
    plt.ylabel("magnitude")
    plt.savefig(f"{plot_path}/rfft_{gen_name}.png")
    tikz.save(f"{plot_path}/rfft_{gen_name}.tex", standalone=True)
    plt.clf()


def _compute_fingerprint_wpt(
    directory: str,
    gen_name: str = "",
    seconds: int = 1,
    wavelet_str: str = "sym8",
    plot_path="./plots/fingerprints/",
) -> torch.Tensor:
    dataset = util.AudioDataset(
        directory,
        sample_rate=SAMPLE_RATE,
    )
    clips = []
    for clip, _fs in dataset:
        if clip.shape[-1] > seconds * SAMPLE_RATE:
            clip = clip[:, : seconds * SAMPLE_RATE]
            clips.append(clip)
    print(f"Clip no: {len(clips)}")
    clip_array = torch.stack(clips).numpy()

    wavelet = pywt.Wavelet(wavelet_str)
    pywt_wp_tree = pywt.WaveletPacket(data=clip_array, wavelet=wavelet, mode="reflect")

    # get the pytorch decomposition
    level = 8
    wp_nodes = pywt_wp_tree.get_level(level, order="freq")
    wp_paths = [n.path for n in wp_nodes]

    use = len(wp_nodes) - len(wp_nodes) // 2

    for pos, path in enumerate(wp_paths):
        if pos < use:
            pywt_wp_tree[path] = np.zeros_like(pywt_wp_tree[path].data)

    filt_rec = pywt_wp_tree.reconstruct()

    mean_filt = np.mean(filt_rec, 0)
    pywt_wp_tree = pywt.WaveletPacket(data=mean_filt, wavelet=wavelet, mode="reflect")
    wp_nodes = pywt_wp_tree.get_level(level, order="freq")
    to_plot = np.stack([n.data for n in wp_nodes[use:]])

    # norm_list = [torch.max(torch.abs(p)) for p in packet_list]
    # if max_norm:
    #     packet_list = [p / pmax for p, pmax in zip(packet_list, norm_list)]

    mean_ln_abs_wpt = np.log(np.abs(np.mean(np.squeeze(to_plot, 1), -1)))
    plt.title(f"{gen_name} - ln(abs(wpt(x)))")
    plt.plot(mean_ln_abs_wpt)
    plt.savefig(f"{plot_path}/wpt_{gen_name}.png")
    plt.clf()


if __name__ == "__main__":
    base_path = (
        "/p/home/jusers/gasenzer1/juwels/project_drive/kgasenzer/audiodeepfakes/"
    )
    plot_path = base_path + "logs/log2/plots/fingerprints"
    Path(plot_path).mkdir(parents=True, exist_ok=True)

    # Important: Put corresponding data directories here!
    paths = [
        "A_ljspeech/",
        "B_melgan/",
        "C_hifigan/",
        "D_mbmelgan/",
        "E_fbmelgan/",
        "F_waveglow/",
        "G_pwg/",
        "H_lmelgan/",
        "I_avocodo/",
        "J_bigvgan/",
        "K_lbigvgan/",
        "L_conformer/",
        "M_jsutmbmelgan/",
        "N_jsutpwg/",
    ]

    for path in paths:
        path = base_path + "data/fake/" + path
        print(f"Processing {path}.", flush=True)
        name = path.split("/")[-2].split("_")[-1]
        # _compute_fingerprint_wpt(directory=path, gen_name=name, plot_path=plot_path)
        _compute_fingerprint_rfft(directory=path, gen_name=name, plot_path=plot_path)
