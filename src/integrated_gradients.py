"""Caculate integrated gradients of trained models."""
import argparse
import pickle
from pathlib import Path
from typing import Optional

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib as tikz
import torch
from captum.attr import IntegratedGradients, Saliency
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data_loader import LearnWavefakeDataset
from .ptwt_continuous_transform import get_diff_wavelet
from .train_classifier import get_model
from .utils import set_seed
from .wavelet_math import get_transforms


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

    model_path = (args.model_path_prefix).split("_")

    wavelet = get_diff_wavelet(args.wavelet)
    num_workers = args.num_workers
    plot_path = args.plot_path
    gans = args.gans
    seeds = args.seeds
    sample_rate = args.sample_rate
    model_name = args.model
    batch_size = args.batch_size
    flattend_size = args.flattend_size
    nclasses = args.nclasses
    f_min = args.f_min
    f_max = args.f_max
    num_of_scales = args.num_of_scales
    data_prefix = args.data_prefix
    features = args.features
    hop_length = args.hop_length

    if "doubledelta" in features:
        channels = 60
    elif "delta" in features:
        channels = 40
    elif "lfcc" in features:
        channels = 20
    else:
        channels = int(args.num_of_scales)

    Path(f"{plot_path}/images").mkdir(parents=True, exist_ok=True)

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

        transforms, normalize = get_transforms(
            args,
            f"{data_prefix}_{gan}",
            features,
            device,
            wavelet,
            normalization=args.mean == 0.0,
            pbar=args.pbar,
        )
        set_name = "_test"
        test_data_dir = "/home/s6kogase/data/run5/fake_22050_22050_0.7_all"
        test_data_set = LearnWavefakeDataset(test_data_dir + set_name)

        test_data_loader = DataLoader(
            test_data_set,
            batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        postfix = f'{"_".join(model_path)}_{gan}'
        postfix += f"_target{target}_x{times*batch_size}"
        postfix = postfix.split("/")[-1]
        bar = tqdm(
            iter(test_data_loader),
            desc="attribute",
            total=len(test_data_loader),
            disable=not args.pbar,
        )
        for seed in seeds:
            index = 0
            print(f"seed: {seed}")
            set_seed(seed)

            model_dir = f'{"_".join(model_path)}_{gan}_{str(seed)}.pt'
            print(model_dir)
            model = get_model(
                wavelet=wavelet,
                model_name=model_name,
                nclasses=nclasses,
                batch_size=batch_size,
                f_min=f_min,
                f_max=f_max,
                sample_rate=sample_rate,
                num_of_scales=num_of_scales,
                flattend_size=flattend_size,
                stft=args.transform == "stft",
                features=features,
                hop_length=hop_length,
                in_channels=2 if args.loss_less else 1,
                channels=channels,
            )
            old_state_dict = torch.load(model_dir, map_location=device)
            model.load_state_dict(old_state_dict)

            model.to(device)

            integrated_gradients = IntegratedGradients(model)
            saliency = Saliency(model)

            model.zero_grad()
            for batch in bar:
                index += 1
                label = (batch["label"].cuda() != 0).type(torch.long)

                if make_binary_labels:
                    label[label > 0] = 1

                if target not in label:
                    index -= 1
                    if index == times:
                        break
                    continue

                freq_time_dt = transforms(batch["audio"].cuda())
                freq_time_dt_norm = normalize(freq_time_dt)

                attributions_ig = integrated_gradients.attribute(
                    freq_time_dt_norm,
                    target=label,
                    n_steps=200,
                    internal_batch_size=batch_size,
                ).squeeze(0)
                welford_ig.update(attributions_ig)
                del attributions_ig

                attributions_sals = saliency.attribute(
                    freq_time_dt_norm, target=label
                ).squeeze(0)
                welford_sal.update(attributions_sals)
                del attributions_sals

                torch.cuda.empty_cache()
                if index == times:
                    break

        mean_ig = welford_ig.finalize()
        mean_sal = welford_sal.finalize()
        mean_ig_max = torch.max(mean_ig, dim=1)[0]
        mean_ig_min = torch.min(mean_ig, dim=1)[0]
        audio_packets = torch.mean(freq_time_dt, dim=0)

        inital = np.transpose(audio_packets.cpu().detach().numpy(), (1, 2, 0)).squeeze(
            -1
        )
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

        seconds = args.window_size / sample_rate
        t = np.linspace(0, seconds, int(seconds // (1 / sample_rate)))
        bins = np.int64(num_of_scales)
        n = list(range(int(bins)))
        freqs = (sample_rate / 2) * (n / bins)  # type: ignore

        x_ticks = list(range(inital.shape[-1]))[:: inital.shape[-1] // 10]
        x_labels = np.around(np.linspace(min(t), max(t), inital.shape[-1]), 2)[
            :: inital.shape[-1] // 10
        ]

        y_ticks = n[:: freqs.shape[0] // 10]
        y_labels = np.around(freqs[:: freqs.shape[0] // 10] / 1000, 1)

        im_plot(
            freq_time_dt[0].squeeze(0).cpu().detach().numpy(),
            f"{plot_path}/raw_{postfix}",
            cmap="turbo",
            x_ticks=x_ticks,
            x_labels=x_labels,
            y_ticks=y_ticks,
            y_labels=y_labels,
            vmax=np.max(attr_ig).item(),
            vmin=np.min(attr_ig).item(),
        )
        im_plot(
            attr_ig,
            f"{plot_path}/attr_ig_{postfix}",
            cmap="viridis_r",
            x_ticks=x_ticks,
            x_labels=x_labels,
            y_ticks=y_ticks,
            y_labels=y_labels,
            vmax=np.max(attr_ig).item(),
            vmin=np.min(attr_ig).item(),
        )
        im_plot(
            attr_sal,
            f"{plot_path}/attr_sal_{postfix}",
            cmap="plasma",
            vmax=np.max(attr_sal).item(),
            vmin=np.min(attr_sal).item(),
            x_ticks=x_ticks,
            x_labels=x_labels,
            y_ticks=y_ticks,
            y_labels=y_labels,
            norm=colors.SymLogNorm(linthresh=0.01),
        )

        # bar_plot(ig_max, x_ticks, x_ticklabels, f"{plot_path}/attr_max_{postfix}")
        # bar_plot(ig_min, x_ticks, x_ticklabels, f"{plot_path}/attr_min_{postfix}")
        bar_plot(ig_abs, y_ticks, y_labels, f"{plot_path}/attr_abs_{postfix}")
        plt.close()


def bar_plot(data, x_ticks, x_labels, path):
    """Plot histogram of model attribution."""
    _fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)

    axs.set_xticks(x_ticks)
    axs.set_xticklabels(x_labels)
    axs.set_xlabel("frequency [kHz]")

    axs.bar(
        x=list(range(data.shape[0])),
        height=np.flipud(data),
        color="crimson",
    )
    save_plot(path)


def im_plot(
    data,
    path,
    cmap,
    x_ticks,
    x_labels,
    y_ticks,
    y_labels,
    vmin=None,
    vmax=None,
    norm=None,
):
    """Plot image of model attribution."""
    fig, axs = plt.subplots(1, 1)
    im = axs.imshow(np.flipud(data), aspect="auto", norm=norm, cmap=cmap)
    axs.set_xlabel("time [sec]")
    axs.set_ylabel("frequency [kHz]")
    axs.set_xticks(x_ticks)
    axs.set_xticklabels(x_labels)
    axs.set_yticks(y_ticks)
    axs.set_yticklabels(y_labels)
    fig.colorbar(im, ax=axs)
    fig.set_dpi(200)
    axs.invert_yaxis()

    save_plot(path)


def save_plot(path):
    """Save plt as standalone latex document."""
    tikz.save(
        f"{path}.tex",
        encoding="utf-8",
        standalone=True,
        tex_relative_path_to_data="images",
        override_externals=True,
    )


def _parse_args():
    """Parse cmd line args for attributing audio classification models with integrated gradients."""
    parser = argparse.ArgumentParser(description="Eval models.")
    parser.add_argument(
        "--target-label",
        type=int,
        default=1,
        help="label of target (default: 1)",
    )
    parser.add_argument(
        "--times",
        type=int,
        default=10000,
        help="number of testing samples for attribution (default: 5056)",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default="./plots/attribution",
        help="path for the attribution plots and results (default: ./plots/attribution)",
    )
    parser.add_argument(
        "--model-path-prefix",
        type=str,
        help="Prefix of model path (without '_seed.pt').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="input batch size for testing (default: 128)",
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
        help="Minimum frequency to analyze in Hz. (default: 1000)",
    )
    parser.add_argument(
        "--f-max",
        type=float,
        default=9500,
        help="Maximum frequency to analyze in Hz. (default: 9500)",
    )
    parser.add_argument(
        "--data-prefix",
        type=str,
        help="shared prefix of the data paths",
    )
    parser.add_argument(
        "--nclasses", type=int, default=2, help="number of classes (default: 2)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="the random seeds that are evaluated.",
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
            "bigvgan",
            "bigvganl",
            "avocodo",
            "all",
        ],
        help="model postfix specifying the gan in the trainingset.",
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
            "lcnn",
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
        "--pbar",
        action="store_true",
        help="enables progress bars",
    )
    parser.add_argument(
        "--calc-normalization",
        action="store_true",
        help="calculate normalization for debugging purposes.",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="If differentiable wavelets shall be used.",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=2.0,
        help="Calculate power spectrum of given factor (for stft and packets) (default: 2.0).",
    )
    parser.add_argument(
        "--loss-less",
        action="store_true",
        help="if sign pattern is to be used as second channel.",
    )
    parser.add_argument(
        "--mean",
        type=float,
        default=0,
        help="Pre calculated mean. (default: 0)",
    )
    parser.add_argument(
        "--std",
        type=float,
        default=1,
        help="Pre calculated std. (default: 1)",
    )
    parser.add_argument(
        "--transform",
        choices=[
            "stft",
            "cwt",
            "packets",
        ],
        default="stft",
        help="Use different transformations.",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=1,
        help="Hop length in transformation. (default: 1)",
    )
    parser.add_argument(
        "--features",
        choices=["lfcc", "delta", "doubledelta", "all", "none"],
        default="none",
        help="Use features like lfcc, its first and second derivatives or all of them. \
              Delta and Dooubledelta include lfcc computing. Default: none.",
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
