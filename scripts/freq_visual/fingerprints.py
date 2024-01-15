"""Plot individual GAN fingerprints in audio deepfakes."""
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pywt
import tikzplotlib as tikz
import torch
from scipy.io.wavfile import write
from tqdm import tqdm

DEBUG = True
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if DEBUG:
    # Set python path automatically to base directory
    sys.path.append(BASE_PATH)

import audiofakedetect.plot_util as util

RES = 150
SAMPLE_RATE = 22_050
F_MIN = 1
F_MAX = 11025
AMOUNT = 13100


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
    dataset = util.AudioDataset(directory, sample_rate=SAMPLE_RATE, amount=AMOUNT)
    clips = []
    for clip, _fs in tqdm(dataset, desc="load dataset", total=len(dataset)):
        if clip.shape[-1] > seconds * SAMPLE_RATE:
            clip = clip[:, : seconds * SAMPLE_RATE]
            clips.append(clip.numpy())

    print(f"Clip no: {len(clips)}")
    clip_array = np.stack(clips[:2500])
    del clips
    freq_clips = np.fft.rfft(clip_array, axis=-1)
    freqs = freq_clips.shape[-1]
    use = freqs  # //2

    zeros = np.zeros_like(freq_clips)[:, :, :-use]
    freq_clips = freq_clips[:, :, -use:]
    masked_freq = np.concatenate([zeros, freq_clips], -1)
    masked_time = np.fft.irfft(masked_freq)
    masked_time_mean = np.mean(masked_time, 0)[0]

    mean_ln_abs_fft = np.abs(np.fft.rfft(masked_time_mean)[-use:])
    # std_ln_abs_fft = np.log(np.abs(np.fft.rfft(masked_time_std)[-use:]))
    freqs = np.fft.rfftfreq(masked_time_mean.shape[-1], 1.0 / SAMPLE_RATE)[-use:]
    # plt.subplot(2, 1, 1)
    # plt.title(f"{gen_name} - time")
    # plt.plot(masked_time_mean)
    # plt.subplot(2, 1, 2)

    plt.title(f"{gen_name}")
    plt.semilogy(freqs, mean_ln_abs_fft, label=gen_name)
    plt.xlabel("frequency [Hz]")
    plt.ylabel("mean absolute Fourier coefficient magnitude")
    plt.grid(True)
    if 1:
        tikz.save(f"{plot_path}/rfft_{gen_name}.tex", standalone=True)
        plt.savefig(f"{plot_path}/rfft_{gen_name}.png")

    plt.clf()

    data = np.fft.irfft(masked_time_mean)
    scaled = np.int16(data / np.max(np.abs(data)) * 32767)
    write(f"{plot_path}/wavs/{gen_name}.wav", SAMPLE_RATE, scaled)

    return (freqs, mean_ln_abs_fft, gen_name)


def _compute_fingerprint_wpt(
    directory: str,
    seconds: int = 1,
    wavelet_str: str = "haar",
    gen_name: str = "",
    plot_path: str = "./plots/fingerprints/",
) -> torch.Tensor:
    dataset = util.AudioDataset(directory, sample_rate=SAMPLE_RATE, amount=AMOUNT)
    clips = []
    for clip, _fs in dataset:
        if clip.shape[-1] > seconds * SAMPLE_RATE:
            clip = clip[:, : seconds * SAMPLE_RATE]
            clips.append(clip)
    print(f"Clip no: {len(clips)}")
    clip_array = torch.stack(clips[:2500]).numpy()

    wavelet = pywt.Wavelet(wavelet_str)
    pywt_wp_tree = pywt.WaveletPacket(data=clip_array, wavelet=wavelet, mode="reflect")

    # get the wavelet decomposition
    level = 14
    wp_nodes = pywt_wp_tree.get_level(level, order="freq")
    wp_paths = [n.path for n in wp_nodes]

    packet_list = []
    for path in wp_paths:
        packet_list.append(pywt_wp_tree[path].data)

    packets = np.stack(packet_list, -1)
    freqs = np.linspace(0, SAMPLE_RATE // 2, len(wp_paths))
    mean_packets = np.mean(np.abs(packets), (0, 1, 2))
    plt.title(gen_name)
    plt.semilogy(freqs, mean_packets, label=gen_name)
    plt.xlabel("frequency [Hz]")
    plt.ylabel("mean wavelet packet magnitude")
    if 1:
        tikz.save(f"{plot_path}/wpt_{gen_name}.tex", standalone=True)
        plt.savefig(f"{plot_path}/wpt_{gen_name}.png")

    plt.clf()
    return freqs, mean_packets


if __name__ == "__main__":
    base_path = "./"
    plot_path = base_path + "logs/log5/plots/fingerprints"
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

    plot_tuples = []
    wp_means = []

    for path in paths:
        path = base_path + "data/fake/" + path
        print(f"Processing {path}.", flush=True)
        name = path.split("/")[-2]
        wp_mean = (
            _compute_fingerprint_wpt(path, gen_name=name, plot_path=plot_path),
            name,
        )
        plot_tuple = _compute_fingerprint_rfft(path, name, plot_path=plot_path)
        wp_means.append(wp_mean)
        plot_tuples.append(plot_tuple)

    # for pos, plot_tuple in enumerate(plot_tuples):
    #     plt.subplot(2, 4, pos+1)
    #     plt.title(plot_tuple[2])
    #     plt.plot(plot_tuple[0], plot_tuple[1])
    # tikz.save(f'{plot_path}/groupplot.tex', standalone=True)
    # [0], [-2]
    # [plt.semilogy(wps[0][0], wps[0][1], label=wps[1]) for wps in wp_means]
    # plt.legend()
    # plt.show()
    sum = np.zeros_like(wp_means[0][0][1])
    for wps in wp_means[1:]:
        plot_name = f"{wp_means[0][1]} - {wps[1]}"
        sum += wps[0][1]
        plt.title(plot_name)
        plt.plot(
            wp_means[0][0][0],
            np.log(np.abs(wp_means[0][0][1])) - np.log(np.abs(wps[0][1])),
            label=plot_name,
        )
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Difference of log-scaled absolute wpt-coefficients")
        tikz.save(f"{plot_path}/wpt_diff_{plot_name}.tex", standalone=True)
        plt.savefig(f"{plot_path}/wpt_diff_{plot_name}.png")
        plt.clf()

    sum /= len(wp_means) - 1
    plt.title("all generators")
    plt.semilogy(wp_means[0][0][0], sum, label="all generators")
    plt.xlabel("frequency [Hz]")
    plt.ylabel("mean wavelet packet magnitude")
    tikz.save(f"{plot_path}/wpt_all_generators.tex", standalone=True)
    plt.savefig(f"{plot_path}/wpt_all_generators.png")

    plt.clf()
    plot_name = "ljspeech - all generators"
    plt.title(plot_name)
    plt.plot(
        wp_means[0][0][0],
        np.log(np.abs(wp_means[0][0][1])) - np.log(np.abs(sum)),
        label=plot_name,
    )
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Difference of log-scaled absolute wpt-coefficients")
    tikz.save(f"{plot_path}/wpt_diff_{plot_name}.tex", standalone=True)
    plt.savefig(f"{plot_path}/wpt_diff_{plot_name}.png")
    plt.clf()

    sum = np.zeros_like(plot_tuples[0][1])
    for ffts in plot_tuples[1:]:
        plot_name = f"{plot_tuples[0][2]} - {ffts[2]}"
        sum += ffts[1]
        plt.title(plot_name)
        plt.plot(
            plot_tuples[0][0],
            np.log(np.abs(plot_tuples[0][1])) - np.log(np.abs(ffts[1])),
            label=plot_name,
        )
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Difference of log-scaled absolute Fourier-coefficients")
        tikz.save(f"{plot_path}/fft_diff_{plot_name}.tex", standalone=True)
        plt.savefig(f"{plot_path}/fft_diff_{plot_name}.png")
        plt.clf()

    sum /= len(plot_tuples) - 1

    gen_name = "all generators"
    plt.title(f"{gen_name}")
    plt.semilogy(plot_tuples[0][0], sum, label=gen_name)
    plt.xlabel("frequency [Hz]")
    plt.ylabel("mean absolute Fourier coefficient magnitude")
    plt.grid(True)
    tikz.save(f"{plot_path}/rfft_{gen_name}.tex", standalone=True)
    plt.savefig(f"{plot_path}/rfft_{gen_name}.png")

    plt.clf()

    plot_name = f"{plot_tuples[0][2]} - all generators"
    plt.title(plot_name)
    plt.plot(
        plot_tuples[0][0],
        np.log(np.abs(plot_tuples[0][1])) - np.log(np.abs(ffts[1])),
        label=plot_name,
    )
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Difference of log-scaled absolute Fourier-coefficients")
    tikz.save(f"{plot_path}/fft_diff_{plot_name}.tex", standalone=True)
    plt.savefig(f"{plot_path}/fft_diff_{plot_name}.png")
    plt.clf()
