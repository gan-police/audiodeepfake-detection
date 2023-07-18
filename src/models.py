"""Models for classification of audio deepfakes."""
import sys

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torchsummary import summary
import torchvision.ops as torchvisionops


def contrast(waveform: torch.Tensor) -> torch.Tensor:
    """Apply contrast effect."""
    enhancement_amount = np.random.uniform(0, 100.0)
    return torchaudio.functional.contrast(waveform, enhancement_amount)


def add_noise(waveform: torch.Tensor) -> torch.Tensor:
    """Add noise to waveform."""
    noise = torch.randn(waveform.shape).to(waveform.device)
    noise_snr = np.random.uniform(30, 40)
    snr = noise_snr * torch.ones(waveform.shape[:-1]).to(waveform.device)
    return torchaudio.functional.add_noise(waveform, noise, snr)


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


class GridModelWrapper(nn.Module):
    """Deep CNN."""

    def __init__(
        self,
        sequentials: list,
        transforms: list,
        in_channels: int = 1,
    ) -> None:
        """Define network sturcture."""
        super(GridModelWrapper, self).__init__()

        self.sequentials = nn.ParameterList(sequentials)
        self.transforms = transforms
        self.len = len(self.sequentials)
        
        if len(self.transforms) != self.len:
            print("Warning: length of transforms and sequentials are not the same.")

    def forward(self, x) -> torch.Tensor:
        """Forward pass."""
        for i in range(self.len):
            x = self.sequentials[i](x)
            if len(self.transforms) > i:
                for j in range(len(self.transforms[i])):
                    x = self.transforms[i][j](x)

        return x

class LCNN(nn.Module):
    """Deep CNN with 2D convolutions for detecting audio deepfakes.

    Fork of ASVSpoof Challenge 2021 LA Baseline.
    """

    def __init__(
        self,
        classes: int = 2,
        in_channels: int = 1,
        lstm_channels: int = 256,
        dropout_cnn: float = 0.6,
        dropout_lstm: float = 0.3,
    ) -> None:
        """Define network sturcture."""
        super(LCNN, self).__init__()

        # LCNN from AVSpoofChallenge 2021
        self.lcnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, 5, 1, padding=2),
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
            nn.Conv2d(64, 128, 1, 1, padding=0),
            MaxFeatureMap2D(),
            nn.SyncBatchNorm(64, affine=False),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            MaxFeatureMap2D(),
            nn.SyncBatchNorm(32, affine=False),
            nn.Conv2d(32, 64, 1, 1, padding=0),
            MaxFeatureMap2D(),
            nn.SyncBatchNorm(32, affine=False),
            nn.Conv2d(32, 64, 3, 1, padding=1),
            MaxFeatureMap2D(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout_cnn),
        )

        self.lstm = nn.Sequential(
            BLSTMLayer((lstm_channels // 16) * 32, (lstm_channels // 16) * 32),
            BLSTMLayer((lstm_channels // 16) * 32, (lstm_channels // 16) * 32),
            nn.Dropout(dropout_lstm),
        )

        self.fc = nn.Linear((lstm_channels // 16) * 32, classes)

    def forward(self, x) -> torch.Tensor:
        """Forward pass."""
        x = self.lcnn(x.permute(0, 1, 3, 2))
        x = x.permute(0, 2, 1, 3).contiguous()
        shape = x.shape
        x = self.lstm(x.view(shape[0], shape[1], -1))
        x = self.fc(x).mean(1)

        return x

    def get_name(self) -> str:
        """Return custom string identifier."""
        return "LCNN"


class LearnDeepNet(nn.Module):
    """Deep CNN with 2D convolutions for detecting audio deepfakes."""

    def __init__(
        self,
        classes: int = 2,
        flattend_size: int = 21888,
        in_channels: int = 32,
    ) -> None:
        """Define network sturcture."""
        super(LearnDeepNet, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=3, stride=1),
            nn.SyncBatchNorm(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1),
            nn.SyncBatchNorm(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.SyncBatchNorm(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.SyncBatchNorm(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2),
            nn.SyncBatchNorm(128),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Dropout(0.7),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattend_size, classes),
        )

        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x) -> torch.Tensor:
        """Forward pass."""
        x = self.cnn(x)
        x = self.fc(x)

        return self.logsoftmax(x)

    def get_name(self) -> str:
        """Return custom string identifier."""
        return "LearnDeepNet"


class OneDNet(nn.Module):
    """Deep CNN with 1D convolutions for detecting audio deepfakes."""

    def __init__(
        self,
        classes: int = 2,
        flattend_size: int = 21888,
        in_channels: int = 32,
        num_of_scales: int = 256,
        stride: int = 1,
    ) -> None:
        """Define network structure."""
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(num_of_scales, in_channels, kernel_size=20, stride=stride),
            nn.SyncBatchNorm(in_channels),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=stride),
            nn.SyncBatchNorm(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=stride),
            nn.SyncBatchNorm(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=stride),
            nn.SyncBatchNorm(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=stride),
            nn.SyncBatchNorm(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=stride),
            nn.SyncBatchNorm(64),
            nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Dropout(0.7),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattend_size, classes),
        )
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x) -> torch.Tensor:
        """Forward pass."""
        x = self.cnn(x.squeeze())
        x = self.fc(x)
        return self.logsoftmax(x)

    def get_name(self) -> str:
        """Return custom string identifier."""
        return "OneDNet"


class MaxFeatureMap2D(nn.Module):
    """Max feature map (along 2D).

    MaxFeatureMap2D(max_dim=1) from AVSPoofChallenge 2021 LA Baseline.

    l_conv2d = MaxFeatureMap2D(1)
    data_in = torch.rand([1, 4, 5, 5])
    data_out = l_conv2d(data_in)

    Input:
    ------
    data_in: tensor of shape (batch, channel, ...)

    Output:
    -------
    data_out: tensor of shape (batch, channel//2, ...)

    Note
    ----
    By default, Max-feature-map is on channel dimension,
    and maxout is used on (channel ...)
    """

    def __init__(self, max_dim=1) -> None:
        """Initialize with max feature dimension."""
        super().__init__()
        self.max_dim = max_dim

    def forward(self, inputs) -> torch.Tensor:
        """Forward max feature map."""
        # suppose inputs (batchsize, channel, length, dim)

        shape = list(inputs.size())

        if self.max_dim >= len(shape):
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But input has %d dimensions" % (len(shape)))
            sys.exit(1)
        if shape[self.max_dim] // 2 * 2 != shape[self.max_dim]:
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But this dimension has an odd number of data")
            sys.exit(1)
        shape[self.max_dim] = shape[self.max_dim] // 2
        shape.insert(self.max_dim, 2)

        # view to (batchsize, 2, channel//2, ...)
        # maximize on the 2nd dim
        m, _ = inputs.reshape(*shape).max(self.max_dim)
        return m


class BLSTMLayer(nn.Module):
    """Wrapper over dilated conv1D.

    From AVSPoofChallenge 2021 LA Baseline.

    Input tensor:  (batchsize=1, length, dim_in)
    Output tensor: (batchsize=1, length, dim_out)
    We want to keep the length the same.
    """

    def __init__(self, input_dim, output_dim) -> None:
        """Initialize class."""
        super(BLSTMLayer, self).__init__()
        if output_dim % 2 != 0:
            print("Output_dim of BLSTMLayer is {:d}".format(output_dim))
            print("BLSTMLayer expects a layer size of even number")
            sys.exit(1)
        # bi-directional LSTM
        self.l_blstm = nn.LSTM(input_dim, output_dim // 2, bidirectional=True)

    def forward(self, x):
        """Forward lstm input."""
        # permute to (length, batchsize=1, dim)
        blstm_data, _ = self.l_blstm(x.permute(1, 0, 2))
        # permute it backt to (batchsize=1, length, dim)
        return blstm_data.permute(1, 0, 2)


def get_model(
    args,
    model_name: str,
    nclasses: int = 2,
    num_of_scales: int = 256,
    flattend_size: int = 21888,
    in_channels: int = 1,
    channels: int = 32,
    dropout_cnn: float = 0.6,
    dropout_lstm: float = 0.3,
) -> LearnDeepNet | OneDNet | LCNN:
    """Get torch module model with given parameters."""
    if model_name == "learndeepnet":
        model = LearnDeepNet(
            classes=nclasses,
            flattend_size=flattend_size,
            in_channels=in_channels,
        )  # type: ignore
    elif model_name == "lcnn":
        model = LCNN(
            classes=nclasses,
            in_channels=in_channels,
            lstm_channels=channels,
            dropout_cnn=dropout_cnn,
            dropout_lstm=dropout_lstm,
        )  # type: ignore
    elif model_name == "onednet":
        model = OneDNet(
            classes=nclasses,
            num_of_scales=num_of_scales,
            flattend_size=flattend_size,
        )  # type: ignore
    elif model_name == "gridmodel":
        input_shape = (1, 256, 101)

        model_data = [
            "Conv2d 1 32 3 2",
            "Conv2d 32 64 3 2",
            "Flatten",
            "Linear 96768 2",
            "ReLU",
            "Softmax 1",
        ]
        import torchvision
        from functools import partial
        model_seq = []
        model_data = [
            [torchvision.ops, "Permute 0,1,3,2"],
            "Conv2d 1 64 5 1 2",
            "MaxFeatureMap2D",
            "MaxPool2d 2 2",
            "Conv2d 32 64 1 1 0",
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
        ]
        model_data2 = [
            "BLSTMLayer 512 512",
            "BLSTMLayer 512 512",
            "Dropout 0.1",
            "Linear 512 2",
        ]

        model_seq.append(parse_sequential(model_data, input_shape))
        model_seq.append(parse_sequential(model_data2, (1, 512)))

        def transform(x):
            x = x.permute(0, 2, 1, 3)
            x = x.contiguous()
            return x.view(x.shape[0], x.shape[1], -1)

        transforms = [
            [partial(transform)],
            [partial(torch.Tensor.mean, dim=1)]
        ]
        
        if False not in model_seq:
            model = GridModelWrapper(
                sequentials=model_seq,
                transforms=transforms,
            )
        else:
            raise RuntimeError("Model not valid.")
    
    elif model_name == "modules":
        model = args.module(args)
    return model


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


def parse_sequential(model_str, input_shape) -> nn.Sequential | bool:
    """Parse given model into torch.nn.Module."""
    layers = []

    for layer_str in model_str:
        if type(layer_str) is list:
            module = layer_str[0]
            layer_parts = layer_str[1].split(" ")  
        else:
            module = nn     # default
            layer_parts = layer_str.split(" ")
        try:
            layer_type = getattr(module, layer_parts[0])
        except AttributeError:
            if layer_parts[0] == "MaxFeatureMap2D":
                layer_type = MaxFeatureMap2D
            elif layer_parts[0] == "BLSTMLayer":
                layer_type = BLSTMLayer
            else:
                print(f"Warning: given layer type {layer_parts[0]} not found.")
                return False

        layer_args = []
        for arg in layer_parts[1:]:
            if "," in arg:  # assume list
                arg_list = arg.split(",")
                if "." in arg_list[0]:   # assume float list
                    layer_args.append([float(a) for a in arg_list])
                else:   # assume int list
                    layer_args.append([int(a) for a in arg_list])
            elif "." in arg:    # assume single float
                layer_args.append(float(arg))
            else:   # assume single int
                layer_args.append(int(arg))
        layer = layer_type(*layer_args)
        layers.append(layer)

    model = nn.Sequential(*layers)
    dim_check = check_dimensions(model, input_shape)

    if not dim_check:
        return False
    else:
        return model


def check_dimensions(
    model,
    input_shape,
    verbose: bool = True,
) -> bool:
    """Check if model is valid for given dimensions."""
    try:
        summary(model, input_shape, verbose=0)
    except RuntimeError as e:
        if verbose:
            print(f"Error: {e}")
        return False

    return True


if __name__ == "__main__":
    input_shape = (1, 256, 101)

    model_data = [
        "Conv2d 1 32 3 2",
        "Conv2d 32 64 3 1",
        "Flatten",
        "Linear 96768 2",
        "ReLU",
        "Softmax 1",
    ]

    parse_sequential(model_data, input_shape)
