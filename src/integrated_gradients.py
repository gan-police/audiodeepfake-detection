"""Caculate integrated gradients of trained models."""
import argparse
import pickle
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib as tikz
import torch
from captum.attr import IntegratedGradients, Saliency
from torch.utils.data import DataLoader

from .ptwt_continuous_transform import get_diff_wavelet
from .train_classifier import create_data_loaders, get_model


class Mean:
    """Compute running mean."""

    def __init__(self) -> None:
        """Create a meaner."""
        self.init: Optional[bool] = None

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
            torch.Tensor: Estimated mean.
        """
        return torch.mean(self.mean, dim=0).squeeze(0) / self.count


def main() -> None:
    """Calculate integradet gradients."""
    args = _parse_args()
    print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    wavelet_name = args.wavelet
    wavelet = get_diff_wavelet(wavelet_name)
    if args.stft:
        wavelet_name = "stft"
    plot_path = args.plot_path
    num_workers = args.num_workers
    gans = args.gans
    seeds = args.seeds
    sample_rate = args.sample_rate
    window_size = args.window_size
    model_name = args.model
    batch_size = args.batch_size
    flattend_size = args.flattend_size
    adapt_wav = args.adapt_wavelet
    nclasses = args.nclasses
    f_min = args.f_min
    f_max = args.f_max
    num_of_scales = args.num_of_scales
    data_prefix = args.data_prefix

    Path(f"{plot_path}/gfx/tikz").mkdir(parents=True, exist_ok=True)

    if args.times < batch_size:
        times = 1
        batch_size = args.times
    else:
        times = args.times // batch_size

    target = args.target_label

    make_binary_labels = nclasses == 2

    for gan in gans:
        print(f"Evaluating {gan}...", flush=True)

        welford_ig = Mean()
        welford_sal = Mean()

        test_data_dir = f"{data_prefix}_{gan}"

        _, _, test_data_set = create_data_loaders(
            test_data_dir,
            batch_size,
            False,
            num_workers,
        )

        test_data_loader = DataLoader(
            test_data_set,
            batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        postfix = f"{gan}_{wavelet_name}_{window_size}_{num_of_scales}_{int(f_min)}"
        postfix += f"_{int(f_max)}_{sample_rate}_{model_name}_{adapt_wav}"
        postfix += f"_{target}_x{times*batch_size}"

        for seed in seeds:
            index = 0
            print(f"seed: {seed}")
            torch.manual_seed(seed)

            model_path = f"./log/fake_{wavelet_name}_{sample_rate}_{window_size}"
            model_path += (
                f"_{num_of_scales}_{int(f_min)}-{int(f_max)}_0.7_{gan}_0.0001_128"
            )
            model_path += f"_{nclasses}_10e_{model_name}_{adapt_wav}_{seed}.pt"
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

                audio_cwt = model.transform(audio)  # type: ignore

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
        mean_ig_max = torch.max(mean_ig, dim=1)[0]
        mean_ig_min = torch.min(mean_ig, dim=1)[0]
        audio_cwt = torch.mean(audio_cwt, dim=0)

        inital = np.transpose(audio_cwt.cpu().detach().numpy(), (1, 2, 0))
        attr_ig = mean_ig.cpu().detach().numpy()
        attr_sal = mean_sal.cpu().detach().numpy()
        ig_max = mean_ig_max.cpu().detach().numpy()
        ig_min = mean_ig_min.cpu().detach().numpy()
        ig_abs = np.abs(ig_max + np.abs(ig_min))

        # save results
        stats_file = plot_path + "/" + postfix + ".pkl"
        try:
            res = pickle.load(open(stats_file, "rb"))
        except OSError as e:
            res = []
            print(
                e,
                "results.pickle does not exist, creating a new file.",
            )
        res.append(
            {
                "attr_ig": attr_ig,
                "attr_sal": attr_sal,
                "ig_max": ig_max,
                "ig_min": ig_min,
                "mean_ig_abs": ig_abs,
            }
        )
        pickle.dump(res, open(stats_file, "wb"))

        extent = [0, window_size / sample_rate, f_min / 1000, f_max / 1000]
        im_plot(
            inital.squeeze(2),
            f"{plot_path}/cwt_{postfix}",
            cmap="turbo",
            extent=extent,
            cb_label="dB",
        )
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


def bar_plot(ig, f_min, f_max, n_bins, path):
    """Plot histogram of model attribution."""
    _fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    ticks = np.linspace(0, n_bins, 10)
    tiks = np.linspace(f_min / 1000, f_max / 1000, len(ticks))

    ticklabels = [round(item, 1) for item in tiks]

    axs.set_xticks(ticks)
    axs.set_xticklabels(ticklabels)
    axs.set_xlabel("Frequenz (kHz)")
    axs.set_ylabel("Intensität")

    axs.bar(
        x=list(range(n_bins)),
        height=np.flipud(ig),
        color="crimson",
    )

    save_plot(path)


def im_plot(ig, path, cmap, extent, vmin=None, vmax=None, cb_label=None):
    """Plot image of model attribution."""
    fig, axs = plt.subplots(1, 1)
    im = axs.imshow(ig, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
    axs.set_xlabel("Zeit (sek)")
    axs.set_ylabel("Frequenz (kHz)")
    fig.colorbar(im, ax=axs, label=cb_label)
    fig.set_dpi(200)

    save_plot(path)


def save_plot(path):
    """Save plt as standalone latex document."""
    tikz.save(
        f"{path}.tex",
        encoding="utf-8",
        standalone=True,
        tex_relative_path_to_data="gfx/tikz",
        override_externals=True,
    )


def _parse_args():
    """Parse cmd line args for attributing audio classification models with integrated gradients."""
    parser = argparse.ArgumentParser(description="Eval models.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="input batch size for attribution (default: 64)",
    )
    parser.add_argument(
        "--target-label",
        type=int,
        default=1,
        help="label of target (default: 1)",
    )
    parser.add_argument(
        "--times",
        type=int,
        default=5056,
        help="number of testing samples for attribution (default: 5056)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=11025,
        help="window size of samples in dataset (default: 11025)",
    )
    parser.add_argument(
        "--num-of-scales",
        type=int,
        default=150,
        help="number of scales used in training set. (default: 150)",
    )
    parser.add_argument(
        "--wavelet",
        type=str,
        default="cmor3.3-4.17",
        help="Wavelet to use in cwt. (default: cmor3.3-4.17)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Sample rate of audio. (default: 22050)",
    )
    parser.add_argument(
        "--f-min",
        type=float,
        default=1000,
        help="Minimum frequency analyzed in Hz. (default: 1000)",
    )
    parser.add_argument(
        "--f-max",
        type=float,
        default=9500,
        help="Maximum frequency analyzed in Hz. (default: 9500)",
    )
    parser.add_argument(
        "--data-prefix",
        type=str,
        help="shared prefix of the data paths",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default="./plots/attribution",
        help="path for the attribution plots and results (default: ./plots/attribution)",
    )
    parser.add_argument(
        "--nclasses", type=int, default=2, help="number of classes (default: 2)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="the random seeds that are attributed.",
    )
    parser.add_argument(
        "--gans",
        type=str,
        nargs="+",
        default=[
            "melgan",
            "lmelgan",
            "mbmelgan",
            "fbmelgan",
            "hifigan",
            "waveglow",
            "pwg",
            "all",
        ],
        help="model postfix specifying the gan in the training set.",
    )
    parser.add_argument(
        "--flattend-size",
        type=int,
        default=21888,
        help="dense layer input size (default: 21888)",
    )
    parser.add_argument(
        "--model",
        choices=[
            "onednet",
            "learndeepnet",
            "learnnet",
        ],
        default="learndeepnet",
        help="The model type. Default: learndeepnet.",
    )
    parser.add_argument(
        "--adapt-wavelet",
        action="store_true",
        help="If differentiable wavelets shall be used.",
    )
    parser.add_argument(
        "--stft",
        action="store_true",
        help="If differentiable wavelets shall be used.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of worker processes started by the test and validation data loaders (default: 2)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
