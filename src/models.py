"""Models for classification of audio deepfakes."""
import torch
import torch.nn as nn
import torch.nn.functional as func


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
            nn.Linear(137984, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_output, bias=False),
        )

    def forward(self, input) -> torch.Tensor:
        """Forward pass."""
        x = self.cnn(input)
        x = self.fc(x)
        return x


class Net(nn.Module):
    """Another CNN for classifying deepfakes."""

    def __init__(self, n_classes=2):
        """Define network sturcture."""
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.pooling = nn.AdaptiveAvgPool2d((8, 8))  # extended
        self.fc1 = nn.Linear(16384, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        """Forward Pass."""
        x = self.conv1(x)
        x = func.relu(x)
        x = self.conv2(x)
        x = func.relu(x)
        x = func.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = func.relu(x)
        x = self.conv4(x)
        x = func.relu(x)
        x = func.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = self.pooling(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = func.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
