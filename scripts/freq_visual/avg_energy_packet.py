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
    pt_data: torch.Tensor, wavelet_str: str = "db5", max_lev: int = 6
):
    """Create a packet image to plot."""
    wavelet = pywt.Wavelet(wavelet_str)
    ptwt_wp_tree = ptwt.WaveletPacket(data=pt_data, wavelet=wavelet, mode="reflect")

    # get the pytorch decomposition
    wp_keys = ptwt_wp_tree.get_level(max_lev)
    packet_list = []
    for node in wp_keys:
        packet_list.append(torch.squeeze(ptwt_wp_tree[node], 0))

    wp_pt = torch.stack(packet_list, dim=-1)
    return wp_pt



if __name__ == "__main__":    

    dataset = LearnWavefakeDataset(data_dir='/home/wolter/uni/audiofake/data/ljspeech_22050_11025_0.7_train')
    mean_dict = {}
    for it, batch in enumerate(
        tqdm(
            iter(dataset),
            desc="Training",
            total=len(dataset)
        )
    ):
        batch_audios = batch['audio'].cuda(non_blocking=True)
        packets = compute_pytorch_packet_representation(batch_audios)
        # plt.imshow(torch.log(torch.abs(packets.cpu())).T)
        # plt.show()
        mean_packets = torch.mean(torch.log(torch.abs(packets)), 0)
        label = int(batch['label'])
        if label in mean_dict.keys():
            mean_dict[label].append(mean_packets)
        else:
            mean_dict[label] = [mean_packets]
    
    for key in mean_dict.keys():
        plt.plot(torch.mean(torch.stack(mean_dict[key], 0), 0).cpu().numpy(),
                 label=key)
    plt.legend()
    plt.show()
    
    pass