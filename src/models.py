"""Models for classification of audio deepfakes."""
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torchaudio.transforms import ComputeDeltas

from .wavelet_math import LFCC, CWTLayer, STFTLayer


def compute_parameter_total(net: torch.nn.Module) -> int:
    """Compute the parameter total of the input net.

    Args:
        net (torch.nn.Module): The model containing the
            parameters to count.

    Returns:
        int: The parameter total.
    """
    total = 0
    for p_name, p in net.named_parameters():
        if p.requires_grad:
            print(p_name)
            print(p.shape)
            total += int(np.prod(p.shape))  # type: ignore
    return total


class LearnDeepTestNet(nn.Module):
    """Deep CNN with 2D convolutions for detecting audio deepfakes."""

    def __init__(
        self,
        wavelet,
        classes: int = 2,
        f_min: float = 2000,
        f_max: float = 4000,
        num_of_scales: int = 150,
        sample_rate: int = 8000,
        batch_size: int = 256,
        flattend_size: int = 21888,
        stft: bool = False,
        features: str = "none",
        hop_length: int = 64,
        raw_input: Optional[bool] = True,
    ) -> None:
        """Define network sturcture."""
        super(LearnDeepTestNet, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        freqs = torch.linspace(f_max, f_min, num_of_scales, device=device) / sample_rate

        self.flattend_size = flattend_size
        self.raw_input = raw_input
        self.features = features

        # only set log scale if raw cwt or stft are used
        if "lfcc" in features or "delta" in features or "all" in features:
            self.pre_log = False
            if "all" in features:
                channels = 60
            else:
                channels = 20
        else:
            self.pre_log = True
            channels = 32

        if stft:
            self.transform = STFTLayer(  # type: ignore
                n_fft=num_of_scales * 2 - 1,
                hop_length=hop_length,
                log_scale=self.pre_log,
            )
        else:
            self.transform = CWTLayer(  # type: ignore
                wavelet=wavelet,
                freqs=freqs,
                hop_length=hop_length,
                batch_size=batch_size,
                log_scale=self.pre_log,
            )

        self.lfcc = LFCC(
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
            num_of_scales=num_of_scales,
        )
        self.delta = ComputeDeltas()

        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, stride=2),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, stride=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattend_size, classes),
        )
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x) -> torch.Tensor:
        """Forward pass."""
        if self.raw_input:
            x = self.transform(x)

        if not self.pre_log:
            lfcc = self.lfcc(x)
            x = lfcc

        if "delta" in self.features or "all" in self.features:
            delta = self.delta(lfcc)
            x = delta

        if "double" in self.features or "all" in self.features:
            doubledelta = self.delta(delta)
            x = doubledelta

        if "all" in self.features:
            x = torch.hstack(
                (
                    lfcc.squeeze(),
                    delta.squeeze(),
                    doubledelta.squeeze(),
                )
            ).unsqueeze(1)

        if self.pre_log:
            x = self.cnn1(x)
        else:
            x = self.cnn2(x)

        x = self.fc(x)
        return self.logsoftmax(x)

    def get_name(self) -> str:
        """Return custom string identifier."""
        return "LearnDeepTestNet"


class LearnNet(nn.Module):
    """Deep CNN with 2D convolutions for detecting audio deepfakes.

    A little less params than LearnDeepTestNet.
    """

    def __init__(
        self,
        wavelet,
        classes: int = 2,
        f_min: float = 2000,
        f_max: float = 4000,
        num_of_scales: int = 150,
        sample_rate: int = 8000,
        batch_size: int = 256,
        flattend_size: int = 39168,
        stft: bool = False,
        hop_length: int = 1,
        raw_input: Optional[bool] = True,
    ) -> None:
        """Define network sturcture."""
        super(LearnNet, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        freqs = torch.linspace(f_max, f_min, num_of_scales, device=device) / sample_rate
        self.flattend_size = flattend_size
        self.raw_input = raw_input

        if stft:
            self.transform = STFTLayer(
                n_fft=num_of_scales * 2 - 1,
                hop_length=hop_length,
            )  # type: ignore
        else:
            self.transform = CWTLayer(  # type: ignore
                wavelet=wavelet,
                freqs=freqs,
                batch_size=batch_size,
                hop_length=hop_length,
            )

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(
                32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(
                32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=3),
            nn.BatchNorm2d(
                64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=3),
            nn.BatchNorm2d(
                128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattend_size, classes, bias=True),
        )
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x) -> torch.Tensor:
        """Forward pass."""
        if self.raw_input:
            x = self.transform(x)
        x = self.cnn(x)
        x = self.fc(x)
        return self.logsoftmax(x)

    def get_name(self) -> str:
        """Return custom string identifier."""
        return "LearnNet"


class OneDNet(nn.Module):
    """Deep CNN with 1D convolutions for detecting audio deepfakes."""

    def __init__(
        self,
        wavelet,
        classes: int = 2,
        f_min: float = 1000,
        f_max: float = 9500,
        num_of_scales: int = 150,
        sample_rate: int = 22050,
        batch_size: int = 128,
        stride: int = 2,
        flattend_size: int = 5440,
        stft: bool = False,
        hop_length: int = 1,
        raw_input: Optional[bool] = True,
    ) -> None:
        """Define network structure."""
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        freqs = torch.linspace(f_max, f_min, num_of_scales, device=device) / sample_rate

        self.flattend_size = flattend_size
        self.raw_input = raw_input

        if stft:
            self.transform = STFTLayer(
                n_fft=num_of_scales * 2 - 1,
                hop_length=hop_length,
            )  # type: ignore
        else:
            self.transform = CWTLayer(  # type: ignore
                wavelet=wavelet,
                freqs=freqs,
                batch_size=batch_size,
                hop_length=hop_length,
            )

        self.cnn = nn.Sequential(
            nn.Conv1d(num_of_scales, 32, kernel_size=53, stride=stride),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=stride),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=stride),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=stride),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=stride),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=stride),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AvgPool1d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattend_size, classes),  # 1184, 128, 4992, 9984
        )
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x) -> torch.Tensor:
        """Forward pass."""
        if self.raw_input:
            x = self.transform(x)
        x = x.squeeze(1)
        x = self.cnn(x)
        x = self.fc(x)
        return self.logsoftmax(x)

    def get_name(self) -> str:
        """Return custom string identifier."""
        return "OneDNet"


def save_model(model: torch.nn.Module, path) -> None:
    """Save the state dict of the model to the specified path.

    Args:
        model (torch.nn.Module): model to store
        path: file path of the storage file
    """
    torch.save(model.state_dict(), path)


def initialize_model(model: torch.nn.Module, path) -> torch.nn.Module:
    """Initialize the given model from a stored state dict file.

    Args:
        model (torch.nn.Module): model to initialize
        path: file path of the storage file
    """
    return model.load_state_dict(torch.load(path))
