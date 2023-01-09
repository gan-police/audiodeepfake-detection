"""Models for classification of audio deepfakes."""
import torch
import torch.nn as nn


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
            nn.Linear(16896, classes, bias=True),  # 64: 10752, 80: 224: 46656, 512
        )

    def forward(self, input) -> torch.Tensor:
        """Forward pass."""
        x = self.cnn(input)
        x = self.fc(x)
        return x

    def get_name(self) -> str:
        """Return custom string identifier."""
        return "DeepTestNet"


class DeepTestNetOld(nn.Module):
    """Old Deep CNN with 2D convolutions for detecting audio deepfakes."""

    def __init__(self, classes=2, batch_size=64) -> None:
        """Define network sturcture."""
        super(DeepTestNetOld, self).__init__()
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
            nn.Conv2d(64, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(
                32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
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
            nn.Linear(128, classes, bias=True),  # 64: 10752, 80: 224: 46656, 512
        )

    def forward(self, input) -> torch.Tensor:
        """Forward pass."""
        x = self.cnn(input)
        x = self.fc(x)
        return x

    def get_name(self) -> str:
        """Return custom string identifier."""
        return "DeepTestNetOld"


class OneDNet(nn.Module):
    """Deep CNN with 1D convolutions for detecting audio deepfakes."""

    def __init__(self, n_input=1, n_output=2, stride=2, n_channel=64) -> None:
        """Define network structure."""
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_input, n_channel, kernel_size=50, stride=stride),
            nn.BatchNorm1d(n_channel),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(n_channel, n_channel, kernel_size=3),
            nn.BatchNorm1d(n_channel),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(n_channel, n_channel, kernel_size=3),
            nn.BatchNorm1d(n_channel),
            nn.ReLU(),
            nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3),
            nn.BatchNorm1d(2 * n_channel),
            nn.ReLU(),
            nn.MaxPool1d(3),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9984, n_output),  # 1184, 128, 4992, 9984
        )

    def forward(self, x) -> torch.Tensor:
        """Forward pass."""
        x = x.squeeze(1)
        x = self.cnn(x)
        # x = torch.nn.functional.avg_pool1d(x, x.shape[-1])
        # x = x.squeeze(2)
        # x = x.permute(0, 2, 1)
        x = self.fc(x)
        return x

    def get_name(self) -> str:
        """Return custom string identifier."""
        return "OneDNet"


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
