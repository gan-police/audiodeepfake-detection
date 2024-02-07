"""Models for classification of audio deepfakes."""

import ast
import sys
from copy import copy
from typing import Any, Union

import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary

from .utils import DotDict


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
            nn.Dropout(0.7),
        )

        self.lstm = nn.Sequential(
            BLSTMLayer((lstm_channels // 16) * 32, (lstm_channels // 16) * 32),
            BLSTMLayer((lstm_channels // 16) * 32, (lstm_channels // 16) * 32),
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


class Regression(torch.nn.Module):
    """A shallow linear-regression model."""

    def __init__(self, args: DotDict) -> None:
        """Create the regression model.

        Args:
            args (DotDict): The configuration dictionary.
        """
        super().__init__()
        self.linear = torch.nn.Linear(args.num_of_scales * 101, 2)
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the regression forward pass.

        Args:
            x (torch.Tensor): An input tensor of shape [batch_size, ...].

        Returns:
            torch.Tensor: A logsoftmax scaled output of shape
                [batch_size, classes].
        """
        x_flat = torch.reshape(x, [x.shape[0], -1])
        return self.logsoftmax(self.linear(x_flat))


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

    def __init__(self, max_dim: int = 1) -> None:
        """Initialize with max feature dimension."""
        super().__init__()
        self.max_dim = max_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
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


class DCNN(torch.nn.Module):
    """Deep CNN with dilated convolutions."""

    def __init__(
        self,
        args: DotDict,
    ) -> None:
        """Define network sturcture.

        Args:
            args (DotDict): The configuration dictionary.
        """
        super(DCNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(
                args.input_dim[1], args.ochannels1, args.kernel1, stride=1, padding=2
            ),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.SyncBatchNorm(args.ochannels1, affine=False),
            nn.Conv2d(args.ochannels1, args.ochannels2, 1, 1, padding=0),
            nn.PReLU(),
            nn.SyncBatchNorm(args.ochannels2, affine=False),
            nn.Conv2d(args.ochannels2, args.ochannels3, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.SyncBatchNorm(args.ochannels3, affine=False),
            nn.Conv2d(args.ochannels3, args.ochannels4, 3, 1, padding=1),
            nn.PReLU(),
            nn.SyncBatchNorm(args.ochannels4, affine=False),
            nn.Conv2d(args.ochannels4, args.ochannels5, 3, 1, padding=1),
            nn.PReLU(),
            nn.SyncBatchNorm(args.ochannels5, affine=False),
            nn.Conv2d(args.ochannels5, 64, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(args.dropout_cnn),
        )

        time_dim = ((args.input_dim[-1])) // 8 + args.time_dim_add

        self.dil_conv = nn.Sequential(
            nn.SyncBatchNorm(time_dim, affine=True),
            nn.Conv2d(time_dim, time_dim, 3, 1, padding=1, dilation=1),
            nn.PReLU(),
            nn.SyncBatchNorm(time_dim, affine=True),
            nn.Conv2d(time_dim, time_dim, 5, 1, padding=2, dilation=2),
            nn.PReLU(),
            nn.SyncBatchNorm(time_dim, affine=True),
            nn.Conv2d(time_dim, time_dim, 7, 1, padding=2, dilation=4),
            nn.PReLU(),
            nn.Dropout(args.dropout_lstm),
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
        x = self.dil_conv(x)
        x = self.fc(x).mean(1)

        return x

    def get_name(self) -> str:
        """Get name of model."""
        return "DCNN"


class DCNNxDropout(torch.nn.Module):
    """Deep CNN with dilated convolutions."""

    def __init__(
        self,
        args: DotDict,
    ) -> None:
        """Define network sturcture.

        Args:
            args (DotDict): The configuration dictionary.
        """
        super(DCNNxDropout, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(
                args.input_dim[1], args.ochannels1, args.kernel1, stride=1, padding=2
            ),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.SyncBatchNorm(args.ochannels1, affine=False),
            nn.Conv2d(args.ochannels1, args.ochannels2, 1, 1, padding=0),
            nn.PReLU(),
            nn.SyncBatchNorm(args.ochannels2, affine=False),
            nn.Conv2d(args.ochannels2, args.ochannels3, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.SyncBatchNorm(args.ochannels3, affine=False),
            nn.Conv2d(args.ochannels3, args.ochannels4, 3, 1, padding=1),
            nn.PReLU(),
            nn.SyncBatchNorm(args.ochannels4, affine=False),
            nn.Conv2d(args.ochannels4, args.ochannels5, 3, 1, padding=1),
            nn.PReLU(),
            nn.SyncBatchNorm(args.ochannels5, affine=False),
            nn.Conv2d(args.ochannels5, 64, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
        )

        time_dim = ((args.input_dim[-1])) // 8 + args.time_dim_add

        self.dil_conv = nn.Sequential(
            nn.SyncBatchNorm(time_dim, affine=True),
            nn.Conv2d(time_dim, time_dim, 3, 1, padding=1, dilation=1),
            nn.PReLU(),
            nn.SyncBatchNorm(time_dim, affine=True),
            nn.Conv2d(time_dim, time_dim, 5, 1, padding=2, dilation=2),
            nn.PReLU(),
            nn.SyncBatchNorm(time_dim, affine=True),
            nn.Conv2d(time_dim, time_dim, 7, 1, padding=2, dilation=4),
            nn.PReLU(),
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
        x = self.dil_conv(x)
        x = self.fc(x).mean(1)

        return x

    def get_name(self) -> str:
        """Get name of model."""
        return "DCNNxDropout"


class DCNNxDilation(torch.nn.Module):
    """Deep CNN without dilated convolutions."""

    def __init__(
        self,
        args: DotDict,
    ) -> None:
        """Define network sturcture.

        Args:
            args (DotDict): The configuration dictionary.
        """
        super(DCNNxDilation, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(
                args.input_dim[1], args.ochannels1, args.kernel1, stride=1, padding=2
            ),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.SyncBatchNorm(args.ochannels1, affine=False),
            nn.Conv2d(args.ochannels1, args.ochannels2, 1, 1, padding=0),
            nn.PReLU(),
            nn.SyncBatchNorm(args.ochannels2, affine=False),
            nn.Conv2d(args.ochannels2, args.ochannels3, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.SyncBatchNorm(args.ochannels3, affine=False),
            nn.Conv2d(args.ochannels3, args.ochannels4, 3, 1, padding=1),
            nn.PReLU(),
            nn.SyncBatchNorm(args.ochannels4, affine=False),
            nn.Conv2d(args.ochannels4, args.ochannels5, 3, 1, padding=1),
            nn.PReLU(),
            nn.SyncBatchNorm(args.ochannels5, affine=False),
            nn.Conv2d(args.ochannels5, 64, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(args.dropout_cnn),
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
        x = self.fc(x).mean(1)

        return x

    def get_name(self) -> str:
        """Get name of model."""
        return "DCNNxDilation"


def get_model(
    args: DotDict,
    model_name: str,
    nclasses: int = 2,
    in_channels: int = 1,
    lead: bool = False,
) -> Union[LCNN, GridModelWrapper, Any]:
    """Get torch module model with given parameters.

    Args:
        args (DotDict): Configuration dictionary.
        model_name (str): Model name as str, one of: "lcnn", "gridmodel", "modules".
        nclasses (int): Number of output classes of the network. Defaults to 2.
        in_channels (int): Number of input channels. Defaults to 1.
        lead (bool): True if gpu rank is 0. Defaults to False.

    Raises:
        RuntimeError: If parsed model from model string has faults.
        RuntimeError: If modular model fails in executing torchsummary, e.g. if the input
                      dimensions do not fit into the model layers.
        RuntimeError: If model name is not valid.
        RuntimeError: If model name is gridmodel but model_data is not inside the config.

    Returns:
        Union[LCNN, GridModelWrapper, Any]: The model or model wrapper.
    """
    model: LCNN | GridModelWrapper | Any = None
    if model_name == "lcnn":
        if "doubledelta" in args.features:
            lstm_channels = 60
        elif "delta" in args.features:
            lstm_channels = 40
        elif "lfcc" in args.features:
            lstm_channels = 20
        else:
            lstm_channels = int(args.num_of_scales)
        model = LCNN(
            classes=nclasses,
            in_channels=in_channels,
            lstm_channels=lstm_channels,
        )
    elif model_name == "gridmodel":
        if not hasattr(args, "model_data"):
            raise RuntimeError(
                "Config dict does not contain the key model_data,"
                "which should hold the list like model structure."
            )
        model = get_gridsearch_model(args.model_data)
    elif model_name == "modules":
        if check_dimensions(args.module(args), args.input_dim[1:], verbose=lead):
            model = args.module(args)
        else:
            raise RuntimeError("Model not valid.")
    else:
        raise RuntimeError(f"Model with model string '{model_name}' does not exist.")
    return model


def get_gridsearch_model(model_data: list) -> GridModelWrapper:
    """Generate sequential torch model from list like structure.

    Args:
        model_data (list): A model given in a list like structure.

    Raises:
        RuntimeError: If model_data has incorrect format.

    Returns:
        GridModelWrapper: The wrapper class for the sequential grid model.
    """
    # parse model data if exists
    model_data = parse_model(model_data)

    model_seq = []
    transforms = []
    for model_layer in model_data:
        if "input_shape" in model_layer.keys():
            input_shape = model_layer["input_shape"]
        else:
            input_shape = None

        if "transforms" in model_layer.keys():
            transform = model_layer["transforms"]
        else:
            transform = []

        model_seq.append(
            parse_sequential(model_list=model_layer["layers"], input_shape=input_shape)
        )
        transforms.append(transform)

    if False not in model_seq:
        return GridModelWrapper(
            sequentials=model_seq,
            transforms=transforms,
        )
    else:
        raise RuntimeError("Model not valid.")


def parse_model(model_data: list) -> list:
    """Parse the model data list into a structured list.

    Examples:
        Example model string structure:

        >>> model_data = [[
        ...                 {
        ...                     "layers": [
        ...                         [torchvision.ops, "Permute 0,1,3,2"],
        ...                         "Conv2d 1 [64,32,128] 2 1 2",
        ...                         "MaxFeatureMap2D",
        ...                         "MaxPool2d 2 2",
        ...                         ...
        ...                         "Dropout 0.7",
        ...                     ],
        ...                     "input_shape": (1, 256, 101),
        ...                     "transforms": [partial(transf)],
        ...                 },
        ...                 {
        ...                     "layers": [
        ...                         "BLSTMLayer 512 512",
        ...                         "BLSTMLayer 512 512",
        ...                         "Dropout 0.1",
        ...                         "Linear 512 2",
        ...                     ],
        ...                     "input_shape": (1, 512),
        ...                     "transforms": [partial(torch.Tensor.mean, dim=1)],
        ...                 },
        ...             ]]

    Args:
        model_data (list): The whole model data.

    Raises:
        RuntimeError: If a parsing error occurs.

    Returns:
        list: The parsed list.
    """
    for i in range(len(model_data)):
        new_els: list[Any] = []
        for j in range(len(model_data[i])):
            trials = parse_model_str(model_data[i][j]["layers"])
            model_data[i][j]["layers"] = trials[0]
            if len(trials) > 1:
                for k in range(1, len(trials)):
                    if len(new_els) < len(trials) - 1:
                        config_copy = [
                            copy(config_part) for config_part in model_data[i]
                        ]
                        config_copy[j]["layers"] = trials[k]
                        new_els.append(config_copy)
                    elif len(new_els) == len(trials) - 1:
                        new_els[k - 1][j]["layers"] = trials[k]
                    else:
                        raise RuntimeError("Parsing error")
            elif len(new_els) > 0:
                for k in range(0, len(new_els)):
                    new_els[k][j]["layers"] = trials[0]
        model_data.extend(new_els)

    return model_data


def parse_model_str(model_str: list) -> list:
    """Parse a given model string into a separated list for each layer.

    Examples:
        Example model string structure:

        >>> model_str = [
        ...                 [torchvision.ops, "Permute 0,1,3,2"],
        ...                 "Conv2d 1 [64,32,128] 2 1 2",
        ...                 "MaxPool2d 2 2",
        ...                 "Conv2d [32,16,64] 64 1 1 0",
        ...                 ...
        ...                 "MaxPool2d 2 2",
        ...                 "Dropout 0.7",
        ...             ]

    Args:
        model_str (list): A list matching the above example structure.

    Raises:
        RuntimeError: If an element is not of type string or list.
        RuntimeError: If the model layers do not contain the same amount of elements
                      when using nested lists.

    Returns:
        list: The parsed result.
    """
    # TODO: write a test for this
    parsed_output: list = []
    postfix: Any = None
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
            output_list: list[Any] = []

            for part in element_parts:
                if isinstance(part, list):
                    if output_els != len(part):
                        raise RuntimeError(
                            "Model layers must contain the same amount of elements."
                            + f"Expected {output_els}, but got {len(part)}."
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
        summary(model, input_shape, verbose=1 if verbose else 0)
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
