import matplotlib.pyplot as plt
import numpy as np
import ptwt
import pywt
import tikzplotlib as tikz
import torch
from scipy.io import wavfile


def compute_pytorch_packet_representation(
    pt_data: torch.Tensor,
    wavelet_str: str = "sym8",
    max_lev: int = 8,
    log_scale=False,
    max_norm=False,
    norm_list=None,
):
    """Create a packet image to plot."""
    wavelet = pywt.Wavelet(wavelet_str)
    ptwt_wp_tree = ptwt.WaveletPacket(data=pt_data, wavelet=wavelet, mode="reflect")

    # get the pytorch decomposition
    wp_keys = ptwt_wp_tree.get_level(max_lev)
    packet_list = []

    for node in wp_keys:
        node_wp = ptwt_wp_tree[node]
        packet_list.append(node_wp)

    if norm_list is None:
        pass
        norm_list = [torch.max(torch.abs(p)) for p in packet_list]

    if max_norm:
        packet_list = [p / pmax for p, pmax in zip(packet_list, norm_list)]

    wp_pt = torch.stack(packet_list, dim=-1)

    if log_scale:
        wp_pt_log = torch.log(torch.abs(wp_pt) + 1e-12)
        sign_pattern = ((wp_pt < 0).type(torch.float32) * (-1) + 0.5) * 2
        wp_pt = torch.stack([wp_pt_log, sign_pattern], 1)

    return wp_pt, norm_list


if __name__ == "__main__":
    samplerate1, real = wavfile.read(
        "/home/wolter/uni/audiofake/data/ljspeech/A_wavs/LJ001-0002.wav"
    )
    samplerate2, fake = wavfile.read(
        "/home/wolter/uni/audiofake/data/ljspeech/H_ljspeech_full_band_melgan/LJ001-0002_gen.wav"
    )
    level = 8
    fs = samplerate1
    assert samplerate1 == samplerate2

    real = torch.from_numpy(real).unsqueeze(0).type(torch.float32)
    fake = torch.from_numpy(fake).unsqueeze(0).type(torch.float32)

    real_packets, max_list = compute_pytorch_packet_representation(
        real,
        wavelet_str="sym8",
        max_lev=level,
        norm_list=None,
        max_norm=True,
        log_scale=False,
    )
    fake_packets, _ = compute_pytorch_packet_representation(
        fake,
        wavelet_str="sym8",
        max_lev=level,
        norm_list=max_list,
        max_norm=True,
        log_scale=False,
    )

    real_packets = real_packets[0].numpy()
    fake_packets = fake_packets[0].numpy()

    abs_real_packets = np.abs(real_packets)
    abs_fake_packets = np.abs(fake_packets)
    log_real = abs_real_packets  # np.log(abs_real_packets)
    log_fake = abs_fake_packets  # np.log(abs_fake_packets)
    diff = np.abs(real_packets - fake_packets)
    vmin = np.min(log_real)
    vmax = np.max(log_real)

    # y ticks
    n = list(range(int(np.power(2, level))))
    freqs = np.round((fs / 2) * (n / (np.power(2, level))) / 1000.0, 2)[::24]
    n = n[::24]

    # x ticks
    xticks = list(range(real_packets.shape[0]))[::24]
    xlabels = np.round(np.linspace(0, 1, real_packets.shape[0]), 2)[::24]

    fig, axs = plt.subplots(1, 3)

    axs[0].imshow(real_packets.T, vmin=vmin, vmax=vmax)
    axs[0].invert_yaxis()
    axs[0].set_ylabel("frequency [kHz]")
    axs[0].set_xlabel("Time [s]")
    axs[0].set_xticks(xticks)
    axs[0].set_xticklabels(xlabels)
    axs[0].set_yticks(n)
    axs[0].set_yticklabels(freqs)
    axs[0].set_title("Real audio ljspeech")
    axs[1].imshow(fake_packets.T, vmin=vmin, vmax=vmax)
    axs[1].invert_yaxis()
    # axs[1].set_ylabel("frequency [kHz]")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_xticks(xticks)
    axs[1].set_xticklabels(xlabels)
    axs[1].set_yticks(n)
    axs[1].set_yticklabels(freqs)
    axs[1].set_title("Full band melgan")
    axs[2].imshow(np.abs(real_packets - fake_packets).T, vmin=vmin, vmax=vmax)
    axs[2].invert_yaxis()
    # axs[2].set_ylabel("frequency [kHz]")
    axs[2].set_xlabel("Time [s]")
    axs[2].set_xticks(xticks)
    axs[2].set_xticklabels(xlabels)
    axs[2].set_yticks(n)
    axs[2].set_yticklabels(freqs)
    axs[2].set_title("Absolute difference")
    tikz.save("packet_intro.tex", standalone=True)
    plt.show()
    pass
