"""Models for classification of audio deepfakes."""
import torch
import torch.nn as nn


class CNN(nn.Module):
    """CNN for classifying deepfakes."""

    def __init__(self, n_input=1, n_output=2) -> None:
        """Define network sturcture."""
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(238080, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_output, bias=False),
        )

    def forward(self, input) -> torch.Tensor:
        """Forward pass."""
        x = self.cnn(input)
        x = self.fc(x)
        return x

    def get_name(self) -> str:
        """Return custom string identifier."""
        return "CNN"


class Net(nn.Module):
    """Another CNN for classifying deepfakes."""

    def __init__(self, n_classes=2) -> None:
        """Define network sturcture."""
        super(Net, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.AdaptiveAvgPool2d((8, 8)),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16384, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x) -> torch.Tensor:
        """Forward Pass."""
        x = self.cnn(input)
        x = self.fc(x)
        return x

    def get_name(self) -> str:
        """Return custom string identifier."""
        return "Net"


class TestNet(nn.Module):
    """Simple CNN for Testing."""

    def __init__(self, n_classes=2) -> None:
        """Define network sturcture."""
        super(TestNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, n_classes, bias=False),
        )

    def forward(self, input) -> torch.Tensor:
        """Forward pass."""
        x = self.cnn(input)
        x = self.fc(x)
        return x

    def get_name(self) -> str:
        """Return custom string identifier."""
        return "TestNet"
