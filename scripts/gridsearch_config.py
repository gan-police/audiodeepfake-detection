"""Return configuration for grid search."""
import torch
from src.models import BLSTMLayer

def get_config() -> dict:
    """Return config dictonary for grid search.

    Note: The keys must adhere to the args keys.
    """
    config = {
        "learning_rate": [0.0005],
        "weight_decay": [0.001],
        "wavelet": ["sym8"],
        "dropout_cnn": [0.7],
        "dropout_lstm": [0.1],
        "num_of_scales": [256],
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
            torch.nn.Conv2d(64, 32, 3, 1, padding=1),
            torch.nn.Dropout(0.5),
        )

        self.lstm = torch.nn.Sequential(
            BLSTMLayer(8192, 8192)
        )

        self.fc = torch.nn.Linear(8192, 2)

    def forward(self, x) -> torch.Tensor:
        """Forward pass."""
        #import pdb; pdb.set_trace()
        x = self.cnn(x.permute(0, 1, 3, 2))
        x = x.permute(0, 2, 1, 3).contiguous()
        shape = x.shape
        x = self.lstm(x.view(shape[0], shape[1], -1))
        x = self.fc(x).mean(1)

        return x