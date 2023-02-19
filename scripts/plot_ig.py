import argparse
import pickle
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib as tikz
import torch
from src.integrated_gradients import im_plot, bar_plot


def main():
    path = "_22050_learndeepnet_1_x5056.pkl"
    window_size = 11025
    sample_rate = 22050
    f_min = 1000
    f_max = 9500
    wavelet_name = "cmor3.3-4.17"
    num_of_scales = 150
    model_name = "learndeepnet"
    adapt_wav = False
    target = 1
    times = 79
    batch_size = 64
    plot_path = "./plots/attribution"
    gans = [
        "melgan",
        "lmelgan",
        "mbmelgan",
        "fbmelgan",
        "hifigan",
        "waveglow",
        "pwg",
        "all",
    ]

    if not os.path.exists(f"{plot_path}/gfx/tikz/"):
        os.makedirs(f"{plot_path}/gfx/tikz/")

    for gan in gans:
        postfix = f"{gan}_{wavelet_name}_{window_size}_{num_of_scales}_{int(f_min)}"
        postfix += f"_{int(f_max)}_{sample_rate}_{model_name}_{adapt_wav}"
        postfix += f"_{target}_x{times*batch_size}"
        with open(f"{plot_path}/{gan}{path}", "rb") as file:
            res = pickle.load(file)[0]
        attr_ig = res['mean_ig'].cpu().detach().numpy()
        attr_sal = res['mean_sal'].cpu().detach().numpy()
        ig_max = res['mean_ig_max'].cpu().detach().numpy()
        ig_min = res['mean_ig_min'].cpu().detach().numpy()
        ig_abs = np.abs(ig_max - np.abs(ig_min))

        extent = [0, window_size / sample_rate, f_min / 1000, f_max / 1000]
        im_plot(attr_ig, f"{plot_path}/attr_ig_{postfix}", cmap="PRGn", extent=extent)
        im_plot(
            attr_sal,
            f"{plot_path}/attr_sal_{postfix}",
            cmap="hot",
            extent=extent,
            vmax=np.max(attr_sal).item(),
            vmin=np.min(attr_sal).item(),
        )

        n_bins = ig_max.shape[0]
        bar_plot(ig_max, f_min, f_max, n_bins, f"{plot_path}/attr_max_{postfix}")
        bar_plot(ig_max, f_min, f_max, n_bins, f"{plot_path}/attr_min_{postfix}")
        bar_plot(ig_abs, f_min, f_max, n_bins, f"{plot_path}/attr_abs_{postfix}")
        plt.close()



if __name__ == "__main__":
    main()