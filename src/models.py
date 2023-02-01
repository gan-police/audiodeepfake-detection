"""Models for classification of audio deepfakes."""
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .wavelet_math import CWTLayer


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
        raw_input: Optional[bool] = True,
    ) -> None:
        """Define network sturcture."""
        super(LearnDeepTestNet, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        freqs = torch.linspace(f_max, f_min, num_of_scales, device=device) / sample_rate
        self.flattend_size = flattend_size
        self.raw_input = raw_input
        self.cwt = CWTLayer(wavelet=wavelet, freqs=freqs, batch_size=batch_size)

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
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(
                32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(
                64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(
                128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(
                128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattend_size, classes, bias=True),
        )

    def forward(self, x) -> torch.Tensor:
        """Forward pass."""
        if self.raw_input:
            x = self.cwt(x)
        x = self.cnn(x)
        x = self.fc(x)
        return x

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
        raw_input: Optional[bool] = True,
    ) -> None:
        """Define network sturcture."""
        super(LearnNet, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        freqs = torch.linspace(f_max, f_min, num_of_scales, device=device) / sample_rate

        self.flattend_size = flattend_size
        self.raw_input = raw_input
        self.cwt = CWTLayer(wavelet=wavelet, freqs=freqs, batch_size=batch_size)

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

    def forward(self, x) -> torch.Tensor:
        """Forward pass."""
        if self.raw_input:
            x = self.cwt(x)
        x = self.cnn(x)
        x = self.fc(x)
        return x

    def get_name(self) -> str:
        """Return custom string identifier."""
        return "LearnNet"


class OneDNet(nn.Module):
    """Deep CNN with 1D convolutions for detecting audio deepfakes."""

    def __init__(
        self,
        wavelet,
        classes=2,
        f_min=2000,
        f_max=4000,
        num_of_scales=224,
        sample_rate=22050,
        batch_size=256,
        stride=2,
        flattend_size=5440,
    ) -> None:
        """Define network structure."""
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        freqs = torch.linspace(f_max, f_min, num_of_scales, device=device) / sample_rate

        self.flattend_size = flattend_size
        self.cwt = CWTLayer(wavelet=wavelet, freqs=freqs, batch_size=batch_size)

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

    def forward(self, x) -> torch.Tensor:
        """Forward pass."""
        x = self.cwt(x)
        x = x.squeeze(1)
        x = self.cnn(x)
        x = self.fc(x)
        return x

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


class Regression(torch.nn.Module):
    """A linear-regression model."""

    def __init__(self, classes: int, flt_size: int) -> None:
        """Create the regression model.

        Args:
            classes (int): The number of classes or sources to classify.
            flt_size (int): The number of input pixels.
        """
        super().__init__()
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flt_size, 1024),
            # nn.ReLU(),
            nn.Linear(1024, 256),
            # nn.ReLU(),
            nn.Linear(256, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the regression forward pass.

        Args:
            x (torch.Tensor): An input tensor of shape [batch_size, ...].

        Returns:
            torch.Tensor: An output of shape [batch_size, classes].
        """
        out = self.linear(x)
        return out

    def get_name(self) -> str:
        """Return custom string identifier."""
        return "Regression"


class TestNet(nn.Module):
    """Simple CNN for Testing."""

    def __init__(self, classes=2, batch_size=64) -> None:
        """Define network sturcture."""
        super(TestNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(
                32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(
                32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(
                64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(46656, classes, bias=True),  # 64: 10752, 80: 224: 46656
        )

    def forward(self, input) -> torch.Tensor:
        """Forward pass."""
        x = self.cnn(input)
        x = self.fc(x)
        return x

    def get_name(self) -> str:
        """Return custom string identifier."""
        return "TestNet"


class DeepTestNet(nn.Module):
    """Deep CNN with 2D convolutions for detecting audio deepfakes."""

    def __init__(self, classes=2, batch_size=64) -> None:
        """Define network sturcture."""
        super(DeepTestNet, self).__init__()
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
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(
                64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(
                32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(
                64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(
                128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(
                128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                16896, classes, bias=True
            ),  # 64: 10752, 80: 224: 46656, 512; 16896:4368, 17152
        )

    def forward(self, input) -> torch.Tensor:
        """Forward pass."""
        x = self.cnn(input)
        x = self.fc(x)
        return x

    def get_name(self) -> str:
        """Return custom string identifier."""
        return "DeepTestNet"
