"""Models for classification of audio deepfakes."""
import torch
import torch.nn as nn

# TODO: Save model, load model


class CNN(nn.Module):
    """CNN for classifying deepfakes."""

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
            # nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(169344, 256),
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
            nn.Conv2d(1, 32, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8 * 128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes),
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

    def __init__(self, classes=2, batch_size=64) -> None:
        """Define network sturcture."""
        super(TestNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8 * batch_size, classes, bias=False),
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


class TestNet2(nn.Module):
    """Simple CNN for Testing."""

    def __init__(self, classes=2) -> None:
        """Define network sturcture."""
        super(TestNet2, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8192, classes, bias=False),
        )

    def forward(self, input) -> torch.Tensor:
        """Forward pass."""
        x = self.cnn(input)
        x = self.fc(x)
        return x

    def get_name(self) -> str:
        """Return custom string identifier."""
        return "TestNet2"


class Regression(torch.nn.Module):
    """A shallow linear-regression model."""

    def __init__(self, classes: int) -> None:
        """Create the regression model.

        Args:
            classes (int): The number of classes or sources to classify.
        """
        super().__init__()
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12800, 256),
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
