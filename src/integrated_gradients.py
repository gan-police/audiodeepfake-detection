"""Caculate integrated gradients of trained models."""
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


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
    plt.savefig(path)
    """tikz.save(
        f"{path}.tex",
        encoding="utf-8",
        standalone=True,
        tex_relative_path_to_data="images",
        override_externals=True,
    )"""


def interpolate_images(baseline, image, alphas):
    alphas_x = alphas.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    baseline_x = torch.unsqueeze(baseline, 0)
    input_x = torch.unsqueeze(image, 0)
    delta = input_x - baseline_x
    images = baseline_x + alphas_x * delta
    return images


def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / 2.0
    integrated_gradients = torch.mean(grads, dim=0)
    return integrated_gradients


def plot_img_attributions(
    baseline, image, attribution_mask, cmap=None, overlay_alpha=0.4
):
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
