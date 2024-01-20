"""Caculate integrated gradients of trained models."""
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib as tikz
import torch
from matplotlib.figure import Figure


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
        return torch.mean(self.mean, dim=0).squeeze() / self.count


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
    plt.savefig(path + ".jpg")
    tikz.save(
        f"{path}.tex",
        encoding="utf-8",
        standalone=True,
        tex_relative_path_to_data="images",
        override_externals=True,
    )


def interpolate_images(
    baseline: torch.Tensor,
    image: torch.Tensor,
    alphas: torch.Tensor,
) -> torch.Tensor:
    """Interpolate images with baseline and different alphas.

    Args:
        baseline (torch.Tensor): Black baseline of image.
        image (torch.Tensor): Current image.
        alphas (torch.Tensor): Alpha overlays to use.

    Returns:
        torch.Tensor: Interpolated image using the given alphas.
    """
    alphas_x = alphas.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    baseline_x = torch.unsqueeze(baseline, 0)
    input_x = torch.unsqueeze(image, 0)
    delta = input_x - baseline_x
    images = baseline_x + alphas_x * delta
    return images


def integral_approximation(gradients: torch.Tensor) -> torch.Tensor:
    """Approximate integral using the riemann trapez method.

    Args:
        gradients (torch.Tensor): Gradients to approximate.

    Returns:
        torch.Tensor: Riemann integral approximation.
    """
    grads = (gradients[:-1] + gradients[1:]) / 2.0
    integrated_gradients = torch.mean(grads, dim=0)
    return integrated_gradients


def plot_img_attributions(
    image: np.ndarray,
    attribution_mask: np.ndarray,
    cmap=None,
    overlay_alpha: float = 0.4,
) -> Figure:
    """Plot image attributions using an overlay.

    Args:
        image (np.ndarray): Example image in background.
        attribution_mask (np.ndarray): Attribution calculated on many images.
        cmap (Any): Matplotlib colormap. Defaults to None.
        overlay_alpha (float): Alpha value that specifies how much the attribution
                               mask overlays the image. Defaults to 0.4.

    Returns:
        Figure: Matplotlib figure.
    """
    fig, axs = plt.subplots(nrows=1, ncols=3, squeeze=False, figsize=(8, 8))

    axs[0, 0].set_title("Original image")
    axs[0, 0].imshow(image, aspect="auto")
    axs[0, 0].axis("off")

    axs[0, 1].set_title("Attribution mask")
    axs[0, 1].imshow(attribution_mask, aspect="auto", cmap=cmap)
    axs[0, 1].axis("off")
    axs[0, 2].set_title("Overlay")
    axs[0, 2].imshow(attribution_mask, aspect="auto", cmap=cmap)
    axs[0, 2].imshow(image, aspect="auto", alpha=overlay_alpha)
    axs[0, 2].axis("off")

    plt.tight_layout()
    return fig


def plot_attribution_targets(
    seconds: int,
    sample_rate: int,
    num_of_scales: int,
    path: str,
    ig_0: np.ndarray,
    ig_1: np.ndarray,
    ig_01: np.ndarray,
) -> None:
    """Plot attribution of real (ig_0) against fake (ig_1) neuron from model output.

    Args:
        seconds (int): Number of seconds shown in one image.
        sample_rate (int): Sample rate of used audio files.
        num_of_scales (int): Number of scales used in WPT/STFT.
        path (str): Path to the folder where the output will be saved.
        ig_0 (np.ndarray): Integrated gradients for the real neuron.
        ig_1 (np.ndarray): Integrated gradients for the fake neuron.
        ig_01 (np.ndarray): Integrated gradients for both real and fake neuron.
    """
    t = np.linspace(0, seconds, int(seconds // (1 / sample_rate)))
    bins = np.int64(num_of_scales)
    n = list(range(int(bins)))
    freqs = (sample_rate / 2) * (n / bins)  # type: ignore

    x_ticks = list(range(ig_0.shape[-1]))[:: ig_0.shape[-1] // 4]
    x_labels = np.around(np.linspace(min(t), max(t), ig_0.shape[-1]), 2)[
        :: ig_0.shape[-1] // 4
    ]

    y_ticks = n[:: freqs.shape[0] // 6]
    y_labels = np.around(freqs[:: freqs.shape[0] // 6] / 1000, 1)

    cmap = plt.get_cmap("inferno")
    fig, axs = plt.subplots(nrows=1, ncols=3, squeeze=False, figsize=(8, 8))

    axs[0, 0].set_title("Attribution on Real Neuron")

    def sign_log_norm(data):
        # # Shift the data to make it positive and then apply a logarithmic scale
        # data_shifted = data - np.min(data) + 1e-10  # Add a small value to avoid log(0)
        # mean = data_shifted.mean()
        return data * 3

    v_min = -ig_1.max()
    v_max = ig_1.max()
    im = axs[0, 0].imshow(
        np.flipud(sign_log_norm(ig_0)), aspect="auto", cmap=cmap, vmin=v_min, vmax=v_max
    )

    axs[0, 1].set_title("Attribution on Fake Neuron")
    im = axs[0, 1].imshow(
        np.flipud(sign_log_norm(ig_1)), aspect="auto", cmap=cmap, vmin=v_min, vmax=v_max
    )

    axs[0, 2].set_title("Attribution Real and Fake")
    v_min = -ig_1.max()
    v_max = ig_1.max()
    im = axs[0, 2].imshow(
        np.flipud(sign_log_norm(ig_01)),
        aspect="auto",
        cmap=cmap,
        vmin=v_min,
        vmax=v_max,
    )

    fig.colorbar(im, ax=axs)
    axs[0, 0].set_xlabel("time [sec]")
    axs[0, 1].set_xlabel("time [sec]")
    axs[0, 2].set_xlabel("time [sec]")
    axs[0, 0].set_ylabel("frequency [kHz]")

    axs[0, 0].set_xticks(x_ticks)
    axs[0, 1].set_xticks(x_ticks)
    axs[0, 2].set_xticks(x_ticks)
    axs[0, 0].set_xticklabels(x_labels)
    axs[0, 1].set_xticklabels(x_labels)
    axs[0, 2].set_xticklabels(x_labels)
    axs[0, 0].set_yticks(y_ticks)
    axs[0, 1].set_yticks(y_ticks)
    axs[0, 2].set_yticks(y_ticks)
    axs[0, 0].set_yticklabels(y_labels)
    axs[0, 1].set_yticklabels(y_labels)
    axs[0, 2].set_yticklabels(y_labels)
    axs[0, 0].invert_yaxis()
    axs[0, 1].invert_yaxis()
    axs[0, 2].invert_yaxis()

    save_plot(path + "_integrated_gradients")
    plt.show()


def plot_attribution(
    transformations: list,
    wavelets: list,
    cross_sources: list,
    plot_path: str,
    seconds: int = 1,
    sample_rate: int = 22050,
    num_of_scales: int = 256,
) -> None:
    """Plot attribution for saved attribution scores."""
    for transformation in transformations:
        for wavelet in wavelets:
            for cross_source in cross_sources:
                path = (
                    f"{plot_path}/{transformation}_{sample_rate}"
                    + f"_{seconds}_0_fbmelgan_{wavelet}_2.0_False_ljspeech-{cross_source}x2500_target"
                )
                if (
                    os.path.exists(path + "-0_integrated_gradients.npy")
                    and os.path.exists(path + "-1_integrated_gradients.npy")
                    and os.path.exists(path + "-01_integrated_gradients.npy")
                ):
                    ig_0 = np.load(path + "-0_integrated_gradients.npy")
                    ig_1 = np.load(path + "-1_integrated_gradients.npy")
                    ig_01 = np.load(path + "-01_integrated_gradients.npy")
                else:
                    continue

                if not os.path.exists(f"{plot_path}/images"):
                    os.mkdir(f"{plot_path}/images")

                plot_attribution_targets(
                    seconds,
                    sample_rate,
                    num_of_scales,
                    path,
                    ig_0,
                    ig_1,
                    ig_01,
                )

                plt.close()
