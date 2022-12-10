"""Models for classification of audio deepfakes."""
import torch
import torch.nn as nn


class CNN(nn.Module):
    """For classifying deepfakes."""

    def __init__(self, n_input=1, n_output=2) -> None:
        """Define network sturcture."""
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input, 64, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(572544, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_output, bias=False),
        )

    def forward(self, input) -> torch.Tensor:
        """Forward pass."""
        x = self.cnn(input)
        x = self.fc(x)
        return x
