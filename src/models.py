"""Models for classification of audio deepfakes."""
import ast
import sys
from copy import copy

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torchsummary import summary


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
            # nn.Dropout(dropout_lstm),
        )

        self.fc = nn.Linear((lstm_channels // 16) * 32, classes)

    def forward(self, x) -> torch.Tensor:
        """Forward pass."""
        x = self.lcnn(x.permute(0, 1, 3, 2))
        x = x.permute(0, 2, 1, 3).contiguous()
        shape = x.shape
        # import pdb; pdb.set_trace()
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
        args,
    ) -> None:
        """Define network sturcture."""
        super(LearnDeepNet, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, padding=2),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.SyncBatchNorm(64, affine=False),
            nn.Conv2d(64, 32, 1, 1, padding=0),
            nn.PReLU(),
            nn.SyncBatchNorm(32, affine=False),
            nn.Conv2d(32, 96, 3, 1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.SyncBatchNorm(96, affine=False),
            nn.Conv2d(96, 128, 3, 1, padding=1),
            nn.PReLU(),
            nn.SyncBatchNorm(128, affine=False),
            nn.Conv2d(128, 32, 3, 1, padding=1),
            nn.PReLU(),
            nn.SyncBatchNorm(32, affine=False),
            nn.Conv2d(32, 64, 3, 1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.6),
        )
        time_dim = (args.input_dim[-1] + 2) // 8
        self.dil_cnn = nn.Sequential(
            nn.SyncBatchNorm(time_dim, affine=True),
            nn.Conv2d(time_dim, time_dim, 3, 1, padding=1, dilation=1),
            nn.PReLU(),
            nn.SyncBatchNorm(time_dim, affine=True),
            nn.Conv2d(time_dim, time_dim, 5, 1, padding=2, dilation=2),
            nn.PReLU(),
            nn.SyncBatchNorm(time_dim, affine=True),
            nn.Conv2d(time_dim, time_dim, 7, 1, padding=2, dilation=4),
            nn.PReLU(),
            nn.Dropout(0.2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(2),
            nn.Linear(args.flattend_size, 2),
        )
        self.single_gpu = not args.ddp

    def forward(self, x) -> torch.Tensor:
        """Forward pass."""
        # [batch, channels, packets, time]
        x = self.cnn(x.permute(0, 1, 3, 2))

        # [batch, channels, time, packets]
        x = x.permute(0, 2, 1, 3).contiguous()

        # "[batch, time, channels, packets]"
        x = self.dil_cnn(x)
        x = self.fc(x).mean(1)

        return x


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
        model_seq = []
        transforms = []
        for model_layer in args.model_data:
            if "input_shape" in model_layer.keys():
                input_shape = model_layer["input_shape"]
            else:
                input_shape = None

            if "transforms" in model_layer.keys():
                transform = model_layer["transforms"]
            else:
                transform = []

            model_seq.append(
                parse_sequential(
                    model_list=model_layer["layers"], input_shape=input_shape
                )
            )
            transforms.append(transform)

        if False not in model_seq:
            model = GridModelWrapper(
                sequentials=model_seq,
                transforms=transforms,
            )
        else:
            raise RuntimeError("Model not valid.")

    elif model_name == "modules":
        if check_dimensions(args.module(args), args.input_dim[1:]):
            model = args.module(args)
        else:
            raise RuntimeError("Model not valid.")
    else:
        raise RuntimeError(f"Model with model string '{model_name}' does not exist.")
    return model


def parse_model_str(model_str):
    parsed_output = []
    for element in model_str:
        new_elements = []
        output_els = 1
        postfix = None
        if isinstance(element, list):
            postfix = element[0]
            element = element[-1]  # bcause element = [module, layer]
        if isinstance(element, str):
            split = element.split()
            element_parts = [
                ast.literal_eval(part) for part in split[1:]
            ]  # assuming pattern: "Class args"
            element_parts.insert(0, split[0])
        else:
            raise RuntimeError(f"Model string invalid at {element}.")

        # get number of combinations. Must be the same for each layer element.
        for part in element_parts:
            if isinstance(part, list):
                if output_els == 1:
                    output_els = len(part)
                    break

        for i in range(output_els):
            output_list = []

            for part in element_parts:
                if isinstance(part, list):
                    if output_els != len(part):
                        raise RuntimeError(
                            f"Model layers must contain the same amount of elements. Expected {output_els}, but got {len(part)}."
                        )
                    part = part[i]
                output_list.append(str(part).replace(" ", ""))
            if postfix is not None:
                output_list = [postfix, output_list]
            new_elements.append(output_list)

        if len(parsed_output) > 0:
            last_layer = copy(parsed_output[-1])
        else:
            last_layer = None

        for i in range(len(new_elements)):
            if len(parsed_output) == 0:
                parsed_output = [[new_elements[i]]]
            elif len(parsed_output) < i + 1:
                if last_layer is not None:
                    layer = copy(last_layer)
                    layer.append(new_elements[i])
                else:
                    layer = [new_elements[i]]
                parsed_output.append(layer)
            else:
                if len(new_elements) == 1:
                    for part in parsed_output:
                        part.append(new_elements[i])
                else:
                    parsed_output[i].append(new_elements[i])

    return parsed_output


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


def parse_sequential(model_list, input_shape=None) -> nn.Sequential | bool:
    """Parse given model into torch.nn.Module."""
    layers = []

    for layer in model_list:
        if not isinstance(layer[0], str):
            module = layer[0]
            layer_parts = layer[1]
        else:
            module = nn  # default
            layer_parts = layer
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

        import ast

        layer_args = [ast.literal_eval(part) for part in layer_parts[1:]]
        layer = layer_type(*layer_args)
        layers.append(layer)

    model = nn.Sequential(*layers)

    # only perform dim check if input_shape is given
    if input_shape is not None:
        dim_check = check_dimensions(model, input_shape)
        if not dim_check:
            return False

    return model


def check_dimensions(
    model,
    input_shape,
    verbose: bool = True,
) -> bool:
    """Check if model is valid for given dimensions."""
    try:
        summary(model, input_shape, verbose=1)
    except RuntimeError as e:
        if verbose:
            print(f"Error: {e}")
        return False
    #import pdb; pdb.set_trace()
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
