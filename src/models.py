"""Models for classification of audio deepfakes."""
import torch
import torch.nn as nn


class TestNet(nn.Module):
    """Simple CNN for Testing."""

    def __init__(self, classes=2, batch_size=64) -> None:
        """Define network sturcture."""
        super(TestNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2),
            # nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            # nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            # nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            # nn.MaxPool2d(2),
            nn.ReLU(),
            # nn.Conv2d(64, 128, kernel_size=3),
            # nn.MaxPool2d(2),
            # nn.ReLU(),
            # nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(8 * 8 * batch_size, classes, bias=False),
            nn.Linear(4224, classes, bias=False),
        )
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, input) -> torch.Tensor:
        """Forward pass."""
        x = self.cnn(input)
        x = self.fc(x)
        return x

    def get_name(self) -> str:
        """Return custom string identifier."""
        return "TestNet"


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
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, classes),
        )

        # self.activation = torch.nn.Sigmoid()
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

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


def initialize_model(model: torch.nn.Module, path) -> None:
    """Initialize the given model from a stored state dict file.

    Args:
        model (torch.nn.Module): model to initialize
        path: file path of the storage file
    """
    model.load_state_dict(torch.load(path))
