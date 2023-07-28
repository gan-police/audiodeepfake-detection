"""Return configuration for grid search."""
import torch
from torch import nn
import torchvision
from functools import partial
from copy import copy
from src.models import BLSTMLayer, MaxFeatureMap2D, parse_model_str

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
        "epochs": [10],
        "validation_interval": [10],
        "block_norm": [False],
        "batch_size": [128],
        "aug_contrast": [False],
        "model": ["gridmodel"],
        "model_data": [[
            {
                "layers": [
                    [torchvision.ops, "Permute 0,1,3,2"],
                    "Conv2d 1 [64,32,128] 2 1 2",
                    "MaxFeatureMap2D",
                    "MaxPool2d 2 2",
                    "Conv2d [32,16,64] 64 1 1 0",
                    "MaxFeatureMap2D",
                    "SyncBatchNorm 32",
                    "Conv2d 32 96 3 1 1",
                    "MaxFeatureMap2D",
                    "MaxPool2d 2 2",
                    "SyncBatchNorm 48",
                    "Conv2d 48 96 1 1 0",
                    "MaxFeatureMap2D",
                    "SyncBatchNorm 48",
                    "Conv2d 48 128 3 1 1",
                    "MaxFeatureMap2D",
                    "MaxPool2d 2 2",
                    "Conv2d 64 128 1 1 0",
                    "MaxFeatureMap2D",
                    "SyncBatchNorm 64",
                    "Conv2d 64 64 3 1 1",
                    "MaxFeatureMap2D",
                    "SyncBatchNorm 32",
                    "Conv2d 32 64 1 1 0",
                    "MaxFeatureMap2D",
                    "SyncBatchNorm 32",
                    "Conv2d 32 64 3 1 1",
                    "MaxFeatureMap2D",
                    "MaxPool2d 2 2",
                    "Dropout 0.7",
                ],
                "input_shape": (1, 256, 101),
                "transforms": [partial(transf)]
            },
            {
                "layers": [
                    "BLSTMLayer 512 512",
                    "BLSTMLayer 512 512",
                    "Dropout 0.1",
                    "Linear 512 2",
                ],
                "input_shape": (1, 512),
                "transforms": [partial(torch.Tensor.mean, dim=1)]
            }
        ]],
        "module": [LCNN]
    }
    if "model_data" in config.keys():
        for i in range(len(config["model_data"])):
            new_els = []
            for j in range(len(config["model_data"][i])):
                trials = parse_model_str(config["model_data"][i][j]["layers"])
                config["model_data"][i][j]["layers"] = trials[0]
                if len(trials) > 1:
                    for k in range(1, len(trials)):
                        if len(new_els) < len(trials) - 1:
                            config_copy = [copy(config_part) for config_part in config["model_data"][i]]
                            config_copy[j]["layers"] = trials[k]
                            new_els.append(config_copy)
                        elif len(new_els) == len(trials) - 1:
                            new_els[k-1][j]["layers"] = trials[k]
                        else:
                            raise RuntimeError
                elif len(new_els) > 0:
                    for k in range(0, len(new_els)):
                        new_els[k][j]["layers"] = trials[0]
            config["model_data"].extend(new_els)

    return config


def transf(x):
    x = x.permute(0, 2, 1, 3)
    x = x.contiguous()
    return x.view(x.shape[0], x.shape[1], -1)


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
    

class LCNN(nn.Module):
    """Deep CNN with 2D convolutions for detecting audio deepfakes.

    Fork of ASVSpoof Challenge 2021 LA Baseline.
    """

    def __init__(
        self,
        args,
        classes: int = 2,
    ) -> None:
        """Define network sturcture."""
        super(LCNN, self).__init__()

        # LCNN from AVSpoofChallenge 2021
        self.lcnn = nn.Sequential(
            nn.Conv2d(8, 64, 5, 1, padding=2),
            MaxFeatureMap2D(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 1, 1, padding=0),
            MaxFeatureMap2D(),
            nn.SyncBatchNorm(32, affine=False),
            nn.Conv2d(32, 96, 3, 1, padding=1),
            MaxFeatureMap2D(),
            nn.MaxPool2d(2, 2),
            nn.SyncBatchNorm(48, affine=False),
            nn.Conv2d(48, 96, 1, 1, padding=0),
            MaxFeatureMap2D(),
            nn.SyncBatchNorm(48, affine=False),
            nn.Conv2d(48, 128, 3, 1, padding=1),
            MaxFeatureMap2D(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(args.dropout_cnn),
        )
        size = int(args.num_of_scales * 4)
        self.lstm = nn.Sequential(
            BLSTMLayer(size, size),
            BLSTMLayer(size, size),
            #nn.Dropout(args.dropout_lstm),
        )

        self.fc = nn.Linear(size, classes)
        self.scale1 = torch.nn.Sequential(torch.nn.Conv2d(1, 7, 3, padding=1),torch.nn.ReLU(),)
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x) -> torch.Tensor:
        """Forward pass."""
        x_init = x
        #import pdb; pdb.set_trace()
        x1 = x.permute(0, 1, 3, 2)[:,:,:,:128]
        x2 = x.permute(0, 1, 3, 2)[:,:,:,128:]
        x1 = self.scale1(x2)
        x = torch.cat([x2, x1], 1)
        x = self.lcnn(x)
        x = x.permute(0, 2, 1, 3).contiguous()
        shape = x.shape
        x = self.lstm(x.view(shape[0], shape[1], -1))
        x = self.fc(x).mean(1)

        return self.logsoftmax(x)