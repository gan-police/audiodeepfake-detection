import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib as tikz
import torch
import pywt
import ptwt

import src.plot_util as util
from intro_plot import compute_pytorch_packet_representation

DEBUG = True
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if DEBUG:
    # Set python path automatically to base directory
    sys.path.append(BASE_PATH)

RES = 150
SAMPLE_RATE = 22_050
F_MIN = 1
F_MAX = 11025


def plot_mean_std(steps, mean, std, label="", marker="."):
    """Plot means and standard deviations with shaded areas."""
    plt.plot(steps, mean, label=label, marker=marker)
    plt.fill_between(steps, mean - std, mean + std, alpha=0.2)


def _compute_fingerprint_rfft(
    directory: str, gen_name: str='', seconds: int = 3
) -> torch.Tensor:
    dataset = util.AudioDataset(
        directory,
        sample_rate=SAMPLE_RATE,
    )
    clips = []
    for clip, _fs in dataset:
        if clip.shape[-1] > seconds*SAMPLE_RATE:
            clip = clip[:, :seconds*SAMPLE_RATE]
            clips.append(clip)
    print(f"Clip no: {len(clips)}")
    clip_array = torch.stack(clips)
    freq_clips = torch.fft.rfft(clip_array, axis=-1)
    freqs = freq_clips.shape[-1]
    use = freqs//8
    zeros = torch.zeros_like(freq_clips)[:, :, :-use]
    freq_clips = freq_clips[:, :, -use:]
    masked_freq = torch.cat([zeros, freq_clips], -1)
    masked_time = torch.fft.irfft(masked_freq)
    masked_time_mean = torch.mean(masked_time, 0)[0]
    # masked_time_std = torch.std(masked_time, 0)[0]

    mean_ln_abs_fft = torch.log(torch.abs(torch.fft.rfft(masked_time_mean)[-use:]))
    # std_ln_abs_fft = torch.log(torch.abs(torch.fft.rfft(masked_time_std)[-use:]))

    # plt.subplot(2,1,1)
    # plt.title(f"{gen_name} - time")
    # plt.plot(masked_time_mean)
    # plt.subplot(2,1,2)
    plt.title(f"{gen_name} - ln(abs(rfft(x))))")
    plt.plot(mean_ln_abs_fft.cpu().numpy(), label=gen_name)
    plt.savefig(f'./plots/fingerprints/rfft_{gen_name}.png')
    plt.clf()


def _compute_fingerprint_wpt(
    directory: str, gen_name: str='', seconds: int = 1,
    wavelet_str: str = 'sym8'
) -> torch.Tensor:
    dataset = util.AudioDataset(
        directory,
        sample_rate=SAMPLE_RATE,
    )
    clips = []
    for clip, _fs in dataset:
        if clip.shape[-1] > seconds*SAMPLE_RATE:
            clip = clip[:, :seconds*SAMPLE_RATE]
            clips.append(clip)
    print(f"Clip no: {len(clips)}")
    clip_array = torch.stack(clips).numpy()

    wavelet = pywt.Wavelet(wavelet_str)
    pywt_wp_tree = pywt.WaveletPacket(data=clip_array, wavelet=wavelet, mode="reflect")

    # get the pytorch decomposition
    level = 8
    wp_nodes = pywt_wp_tree.get_level(level, order='freq')
    wp_paths = [n.path for n in wp_nodes]

    use = len(wp_nodes) - len(wp_nodes)//2

    for pos, path in enumerate(wp_paths):
        if pos < use:
            pywt_wp_tree[path] = np.zeros_like(pywt_wp_tree[path].data)
        
    filt_rec = pywt_wp_tree.reconstruct()

    mean_filt = np.mean(filt_rec, 0)
    pywt_wp_tree = pywt.WaveletPacket(data=mean_filt, wavelet=wavelet, mode='reflect')
    wp_nodes = pywt_wp_tree.get_level(level, order='freq')
    to_plot = np.stack([n.data for n in wp_nodes[use:]])

    pass
    # norm_list = [torch.max(torch.abs(p)) for p in packet_list]
    # if max_norm:
    #     packet_list = [p / pmax for p, pmax in zip(packet_list, norm_list)]

    mean_ln_abs_wpt = np.log(np.abs(np.mean(np.squeeze(to_plot, 1), -1)))
    plt.title(f"{gen_name} - ln(abs(wpt(x)))")
    plt.plot(mean_ln_abs_wpt)
    plt.savefig(f'./plots/fingerprints/wpt_{gen_name}.png')
    plt.clf()



if __name__ == "__main__":
    Path("./plots/fingerprints").mkdir(parents=True, exist_ok=True)

    # Important: Put corresponding data directories here!
    paths = [
        "../data/ljspeech/A_wavs/",
        "../data/ljspeech/B_ljspeech_melgan/",
        "../data/ljspeech/C_ljspeech_hifiGAN/",
        "../data/ljspeech/D_ljspeech_melgan_large/",
        "../data/ljspeech/E_ljspeech_multi_band_melgan/",
        "../data/ljspeech/F_ljspeech_parallel_wavegan/",
        "../data/ljspeech/G_ljspeech_waveglow/",
        "../data/ljspeech/H_ljspeech_full_band_melgan/",
    ]


    for path in paths:
        print(f"Processing {path}.", flush=True)
        name = path.split('/')[-2]
        _compute_fingerprint_wpt(path, name)
        _compute_fingerprint_rfft(path, name)

