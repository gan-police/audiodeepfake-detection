import os
import sys
from pathlib import Path

import matplotlib
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

import src.plot_util as util

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
    for clip, _fs in tqdm(dataset, desc="load dataset", total=len(dataset)):
        if clip.shape[-1] > seconds * SAMPLE_RATE:
            clip = clip[:, : seconds * SAMPLE_RATE]
            clips.append(clip.numpy())

    print(f"Clip no: {len(clips)}")
    clip_array = np.stack(clips[:2500])
    del clips
    freq_clips = np.fft.rfft(clip_array, axis=-1)
    freqs = freq_clips.shape[-1]
    use = freqs # //2

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
    plt.xlabel('frequency [Hz]')
    plt.ylabel('mean absolute Fourier coefficient magnitude')
    plt.grid(True)
    if 1:
        tikz.save(f'{plot_path}/rfft_{gen_name}.tex', standalone=True)
        plt.savefig(f'{plot_path}/rfft_{gen_name}.png')

    plt.clf()

    data = np.fft.irfft(masked_time_mean);
    scaled = np.int16(data / np.max(np.abs(data)) * 32767)
    write(f'{plot_path}/wavs/{gen_name}.wav', SAMPLE_RATE, scaled)

    return (freqs, mean_ln_abs_fft, gen_name)


def _compute_fingerprint_wpt(
    directory: str, seconds: int = 1,
    wavelet_str: str = 'haar', gen_name: str = "",
    plot_path: str = "./plots/fingerprints/"
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
    clip_array = torch.stack(clips[:2500]).numpy()

    wavelet = pywt.Wavelet(wavelet_str)
    pywt_wp_tree = pywt.WaveletPacket(data=clip_array, wavelet=wavelet, mode="reflect")

    # get the pytorch decomposition

    level = 10 # 22
    level = 12 # 6
    level = 14
    wp_nodes = pywt_wp_tree.get_level(level, order='freq')
    wp_paths = [n.path for n in wp_nodes]

    packet_list = []
    for path in wp_paths:
       packet_list.append(pywt_wp_tree[path].data)
        
    packets = np.stack(packet_list, -1)
    freqs = np.linspace(0, SAMPLE_RATE//2, len(wp_paths))
    mean_packets = np.mean(np.abs(packets), (0, 1, 2))
    plt.title(gen_name)
    plt.semilogy(freqs, mean_packets, label=gen_name)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('mean wavelet packet magnitude')
    if 1:
        tikz.save(f'{plot_path}/wpt_{gen_name}.tex', standalone=True)
        plt.savefig(f'{plot_path}/wpt_{gen_name}.png')

    plt.clf()
    return freqs, mean_packets

if __name__ == "__main__":
    base_path = (
        "/p/home/jusers/gasenzer1/juwels/project_drive/kgasenzer/audiodeepfakes/"
    )
    plot_path = base_path + "logs/log3/plots/fingerprints"
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
        # "I_avocodo/",
        # "J_bigvgan/",
        # "K_lbigvgan/",
        "L_conformer/",
        "M_jsutmbmelgan/",
        "N_jsutpwg/",
    ]

    plot_tuples = []
    wp_means = []

    for path in paths:
        path = base_path + "data/fake/" + path
        print(f"Processing {path}.", flush=True)
        name = path.split('/')[-2]
        # _compute_fingerprint_wpt(path, name)
        wp_means.append((_compute_fingerprint_wpt(path, gen_name=name, plot_path=plot_path)))
        plot_tuples.append(_compute_fingerprint_rfft(path, name, plot_path=plot_path))
    
        #import pdb; pdb.set_trace()

    for pos, plot_tuple in enumerate(plot_tuples):
        plt.subplot(2, 4, pos+1)
        plt.title(plot_tuple[2])
        plt.plot(plot_tuple[0], plot_tuple[1])
    tikz.save(f'{plot_path}/groupplot.tex', standalone=True)
    # [0], [-2]
    [plt.semilogy(wps[0][0], wps[0][1], label=wps[1]) for wps in wp_means]
    plt.legend()
    plt.show()
    pass

