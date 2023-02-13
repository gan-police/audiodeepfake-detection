"""Caculate integrated gradients of trained models."""
import pickle
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import IntegratedGradients, Saliency
from torch.utils.data import DataLoader

from .learn_direct_train_classifier import (
    create_data_loaders,
    create_data_loaders_learn,
    get_model,
)
from .ptwt_continuous_transform import get_diff_wavelet


class WelfordEstimator:
    """Compute running mean and standard deviations.

    The Welford approach greatly reduces memory consumption.
    Port of: https://github.com/gan-police/frequency-forensics/blob/main/src/freqdect/prepare_dataset.py
    """

    def __init__(self) -> None:
        """Create a Welfordestimator."""
        self.init: Optional[bool] = None

    # estimate running mean and std
    # average all axis except the color channel
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    def update(self, batch_vals: torch.Tensor) -> None:
        """Update the running estimation.

        Args:
            batch_vals (torch.Tensor): The current batch element.
        """
        if self.init is None:
            self.init = True
            self.count = 0
            self.mean = torch.zeros(
                batch_vals.shape, device=batch_vals.device, dtype=torch.float32
            )
            self.std = torch.zeros(
                batch_vals.shape, device=batch_vals.device, dtype=torch.float32
            )
            self.m2 = torch.zeros(
                batch_vals.shape, device=batch_vals.device, dtype=torch.float32
            )
        self.count += 1
        self.mean += batch_vals.detach()

    def finalize(self) -> torch.Tensor:
        """Finish the estimation and return the computed mean and std.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Estimated mean and variance.
        """
        return torch.mean(self.mean, dim=0).squeeze(0) / self.count


def main() -> None:
    """Calculate integradet gradients."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    home = "/home/s6kogase"
    plot_path = f"{home}/code/plots/attribution"
    num_workers = 0
    gans = [
        "all",
        "fbmelgan",
        "hifigan",
        "mbmelgan",
        "waveglow",
        "pwg",
        "lmelgan",
        "melgan",
    ]
    seeds = [0, 1]
    wavelets = ["cmor3.3-4.17", "cmor4.6-0.87", "shan0.01-0.4"]
    cu_wv = wavelets[0]
    window_size = 11025
    sample_rate = 16000
    model_name = "learnnet"
    times = 30
    target = 1

    f_min = 1000.0
    f_max = 9500.0
    num_of_scales = 150
    nclasses = 2
    batch_size = 64
    make_binary_labels = nclasses == 2
    flattend_size = 17408
    adapt_wav = False

    for gan in gans:
        print(f"Evaluating {gan}...", flush=True)

        welford_ig = WelfordEstimator()
        welford_sal = WelfordEstimator()

        test_data_dir = [
            f"{home}/data/fake_cmor4.6-0.87_{sample_rate}_8000_{window_size}_224_80-4000_1_0.7_{gan}"
        ]

        wavelet = get_diff_wavelet(cu_wv)
        if (
            model_name == "learndeepnet"
            or model_name == "learnnet"
            or model_name == "onednet"
        ):
            _, _, test_data_set = create_data_loaders_learn(
                test_data_dir,
                batch_size,
                False,
                num_workers,
            )
        else:
            _, _, test_data_set = create_data_loaders(
                test_data_dir,
                batch_size,
                False,
                num_workers,
                wavelet,
                sample_rate,
                num_of_scales,
                f_min,
                f_max,
            )

        test_data_loader = DataLoader(
            test_data_set,
            batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        postfix = (
            f"{gan}_{sample_rate}_{model_name}_{adapt_wav}_{target}_x{times*batch_size}"
        )

        for seed in seeds:
            index = 0
            print(f"seed: {seed}")
            torch.manual_seed(seed)

            model_path = f"{home}/code/log/"
            model_path += (
                f"fake_{cu_wv}_{sample_rate}_{window_size}_150_1000-9500_0.7_{gan}"
            )
            model_path += f"_0.0001_128_2_10e_{model_name}_{adapt_wav}_{seed}.pt"
            print(model_path)
            model = get_model(
                wavelet,
                model_name,
                nclasses,
                batch_size,
                f_min,
                f_max,
                sample_rate,
                num_of_scales,
                raw_input=False,
                flattend_size=flattend_size,
            )
            old_state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(old_state_dict)

            model.to(device)

            integrated_gradients = IntegratedGradients(model)
            saliency = Saliency(model)

            model.zero_grad()
            for batch in iter(test_data_loader):
                index += 1
                audio = batch[test_data_loader.dataset.key].cuda(  # type: ignore
                    non_blocking=True
                )
                label = batch["label"].cuda(non_blocking=True)

                if make_binary_labels:
                    label[label > 0] = 1

                if target not in label:
                    index -= 1
                    if index == times:
                        break
                    continue

                audio_cwt = model.cwt(audio)  # type: ignore

                attributions_ig = integrated_gradients.attribute(
                    audio_cwt,
                    target=label,
                    n_steps=200,
                    internal_batch_size=batch_size,
                ).squeeze(0)
                welford_ig.update(attributions_ig)
                del attributions_ig

                attributions_sals = saliency.attribute(audio_cwt, target=label).squeeze(
                    0
                )
                welford_sal.update(attributions_sals)
                del attributions_sals

                torch.cuda.empty_cache()
                if index % 5 == 0:
                    print("processed ", index * batch_size, flush=True)
                if index == times:
                    break

        mean_ig = welford_ig.finalize()
        mean_sal = welford_sal.finalize()
        mean_ig_nt_max = torch.max(mean_ig, dim=1)[0]
        mean_ig_nt_min = torch.min(mean_ig, dim=1)[0]
        audio_cwt = torch.mean(audio_cwt, dim=0)

        inital = np.transpose(audio_cwt.cpu().detach().numpy(), (1, 2, 0))
        attr_ig = mean_ig.cpu().detach().numpy()
        attr_sal = mean_sal.cpu().detach().numpy()

        extent = [0, window_size, 1000, 9500]
        cmap = "PRGn"
        fig, axes = plt.subplots(1, 1)
        im = axes.imshow(inital.squeeze(2), cmap="turbo", extent=extent)
        fig.set_size_inches(40, 20, forward=True)

        fig.colorbar(im, ax=axes)
        fig.set_dpi(200)
        plt.savefig(f"{plot_path}/init_{postfix}.png")

        fig, axes = plt.subplots(1, 1)
        im = axes.imshow(attr_ig, cmap=cmap, extent=extent)
        fig.set_size_inches(40, 20, forward=True)
        fig.colorbar(im, ax=axes)
        fig.set_dpi(200)
        plt.savefig(f"{plot_path}/attr_ig_{postfix}.png")

        fig, axes = plt.subplots(1, 1)
        im = axes.imshow(
            attr_sal,
            cmap="hot",
            extent=extent,
            vmax=np.max(attr_sal).item(),
            vmin=np.min(attr_sal).item(),
        )
        fig.set_size_inches(40, 20, forward=True)
        fig.colorbar(im, ax=axes)
        fig.set_dpi(200)
        plt.savefig(f"{plot_path}/attr_sal_{postfix}.png")

        n_bins = mean_ig_nt_max.shape[0]

        fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)

        ig_max = mean_ig_nt_max.cpu().detach().numpy()
        axs.bar(
            x=list(range(n_bins)),
            height=np.flipud(ig_max),
            color="crimson",
        )

        ticks = np.linspace(0, n_bins, 10)
        tiks = np.linspace(1000, 9500, len(ticks))

        ticklabels = [round(item) for item in tiks]

        axs.set_xticks(ticks)
        axs.set_xticklabels(ticklabels)

        axs.set_xlabel("Frequenz (kHz)")
        axs.set_ylabel("Intensität")
        plt.savefig(f"{plot_path}/attr_timeless_max_{postfix}.png")

        fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
        ig_min = mean_ig_nt_min.cpu().detach().numpy()
        axs.bar(
            x=list(range(n_bins)),
            height=np.flipud(ig_min),
            color="crimson",
        )
        ticks = np.linspace(0, n_bins, 10)
        tiks = np.linspace(1000, 9500, len(ticks))

        ticklabels = [round(item) for item in tiks]

        axs.set_xticks(ticks)
        axs.set_xticklabels(ticklabels)

        axs.set_xlabel("Frequenz (kHz)")
        axs.set_ylabel("Intensität")
        plt.savefig(f"{plot_path}/attr_timeless_min_{postfix}.png")

        fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
        ticks = np.linspace(0, n_bins, 10)
        tiks = np.linspace(1000, 9500, len(ticks))

        ticklabels = [round(item) for item in tiks]

        axs.set_xticks(ticks)
        axs.set_xticklabels(ticklabels)

        axs.set_xlabel("Frequenz (kHz)")
        axs.set_ylabel("Intensität")

        ig_abs = np.abs(ig_max - np.abs(ig_min))
        axs.bar(
            x=list(range(n_bins)),
            height=np.flipud(ig_abs),
            color="crimson",
        )

        plt.savefig(f"{plot_path}/attr_timeless_abs_{postfix}.png")

        # save results
        stats_file = plot_path + "/" + postfix + ".pkl"
        try:
            res = pickle.load(open(stats_file, "rb"))
        except OSError as e:
            res = []
            print(
                e,
                "stats.pickle does not exist, \
                creating a new file.",
            )
        res.append(
            {
                "mean_ig": mean_ig,
                "mean_sal": mean_sal,
                "mean_ig_max": mean_ig_nt_max,
                "mean_ig_min": mean_ig_nt_min,
                "mean_ig_abs": ig_abs,
            }
        )
        pickle.dump(res, open(stats_file, "wb"))
        plt.close()


if __name__ == "__main__":
    main()
