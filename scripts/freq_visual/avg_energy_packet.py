"""Compute wavelet packet representations of the input data.

In vscode use:
"env": {
    "PYTHONPATH": "${workspaceFolder}"
}
"""

import os
import sys
from itertools import product
from tqdm import tqdm
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ptwt
import pywt
import tikzplotlib as tikz
import torch

import src.plot_util as util
from src.data_loader import LearnWavefakeDataset


def compute_pytorch_packet_representation(
    pt_data: torch.Tensor, wavelet_str: str = "sym8", max_lev: int = 6
):
    """Create a packet image to plot."""
    wavelet = pywt.Wavelet(wavelet_str)
    ptwt_wp_tree = ptwt.WaveletPacket(data=pt_data, wavelet=wavelet, mode="reflect")

    # get the pytorch decomposition
    wp_keys = ptwt_wp_tree.get_level(max_lev)
    packet_list = []
    for node in wp_keys:
        packet_list.append(ptwt_wp_tree[node])

    wp_pt = torch.stack(packet_list, dim=-1)
    return wp_pt

def compute_cwt():
    pass

def compute_stft():
    pass


label_dict = {0: 'ljspeech', 1: 'melgan', 2: 'hifigan', 3: 'melgan_large', 4: 'mb_melgan', 5: 'p_wavegan',
              6: 'waveglow', 7: 'fb_melgan', 8: 'conformer'}

if __name__ == "__main__":    

    dataset = LearnWavefakeDataset(data_dir='/home/wolter/uni/audiofake/data/ljspeech_22050_44100_0.7_train')
    dataset = torch.utils.data.DataLoader(dataset, batch_size=5000)
    mean_dict = {}
    for it, batch in enumerate(
        tqdm(
            iter(dataset),
            desc="computing wps",
            total=len(dataset)
        )
    ):
        batch_audios = batch['audio'].cuda(non_blocking=True)
        packets = compute_pytorch_packet_representation(batch_audios, wavelet_str="sym8", max_lev=8)
        # plt.imshow(torch.log(torch.abs(packets.cpu())).T)
        # plt.show()
        labels = batch['label']
        max_packets = torch.log(torch.max(torch.abs(packets), 1)[0])
        for l, p in zip(labels, max_packets):
            l = int(l)
            if l in mean_dict.keys():
                mean_dict[l].append(p)
            else:
                mean_dict[l] = [p]
    
    for key in mean_dict.keys():
        plt.plot(torch.mean(torch.stack(mean_dict[key], 0), 0).cpu().numpy(),
                 label=label_dict[key])
    plt.legend()
    plt.show()
    
    pass