"""Return configuration for grid search."""
import torch
from src.models import BLSTMLayer

def get_config() -> dict:
    """Return config dictonary for grid search.

    Note: The keys that adhere to the initial command line args will be
    overridden by this configuration. Any other key is also welcome and
    can be used in "modules"-architecure for example.
    """
    config = {
        "learning_rate": [0.0005],
        "weight_decay": [0.001],
        "wavelet": ["sym8"],
        "dropout_cnn": [0.7],
        "dropout_lstm": [0.1],
        "num_of_scales": [256],
        "epochs": [20],
        "block_norm": [True],
        "aug_contrast": [False],
        "model": ["modules"],
        "module": [TestNet]
    }

    return config


class TestNet(torch.nn.Module):
    """Deep CNN."""

    def __init__(
        self,
        args,
    ) -> None:
        """Define network sturcture."""
        super(TestNet, self).__init__()

        channels_in = 2 if args.loss_less == "True" else 1

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(channels_in, 64, 5, 1, padding=2),
            torch.nn.ReLU6(),
            torch.nn.SyncBatchNorm(64),
            torch.nn.Conv2d(64, 64, 3, 1, padding=1),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(0.6),
        )

        size = int(args.num_of_scales * 64 / 2)

        self.lstm = torch.nn.Sequential(
            BLSTMLayer(size, size)
            #torch.nn.Dropout(0.1)
        )

        self.fc = torch.nn.Linear(size, 2)

    def forward(self, x) -> torch.Tensor:
        """Forward pass."""
        #import pdb; pdb.set_trace()
        x = self.cnn(x.permute(0, 1, 3, 2))
        x = x.permute(0, 2, 1, 3).contiguous()
        shape = x.shape
        x = self.lstm(x.view(shape[0], shape[1], -1))
        x = self.fc(x).mean(1)

        return x
    

class Regression(torch.nn.Module):
    """A shallow linear-regression model."""

    def __init__(self, args):
        """Create the regression model.

        Args:
            classes (int): The number of classes or sources to classify.
        """
        super().__init__()
        self.linear = torch.nn.Linear(args.num_of_scales * 101, 2)

        # self.activation = torch.nn.Sigmoid()
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the regression forward pass.

        Args:
            x (torch.Tensor): An input tensor of shape
                [batch_size, ...]

        Returns:
            torch.Tensor: A logsoftmax scaled output of shape
                [batch_size, classes].
        """
        #import pdb; pdb.set_trace()
        x_flat = torch.reshape(x, [x.shape[0], -1])
        return self.logsoftmax(self.linear(x_flat))