"""Models for classification of audio deepfakes."""
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from pywt import ContinuousWavelet
from torchaudio.transforms import ComputeDeltas

from .wavelet_math import LFCC, CWTLayer, STFTLayer


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


class LCNN(nn.Module):
    """Deep CNN with 2D convolutions for detecting audio deepfakes.

    Fork of ASVSpoof Challenge 2021 LA Baseline.
    """

    def __init__(
        self,
        classes: int = 2,
        in_channels: int = 1,
        lstm_channels: int = 32,
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
            nn.BatchNorm2d(32, affine=False),
            nn.Conv2d(32, 96, 3, 1, padding=1),
            MaxFeatureMap2D(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(48, affine=False),
            nn.Conv2d(48, 96, 1, 1, padding=0),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(48, affine=False),
            nn.Conv2d(48, 128, 3, 1, padding=1),
            MaxFeatureMap2D(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 1, 1, padding=0),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(64, affine=False),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(32, affine=False),
            nn.Conv2d(32, 64, 1, 1, padding=0),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(32, affine=False),
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

    def get_name(self) -> str:
        """Return custom string identifier."""
        return "LCNN"


class LearnDeepTestNet(nn.Module):
    """Deep CNN with 2D convolutions for detecting audio deepfakes."""

    def __init__(
        self,
        wavelet,
        classes: int = 2,
        f_min: float = 2000,
        f_max: float = 4000,
        num_of_scales: int = 150,
        sample_rate: int = 8000,
        batch_size: int = 256,
        flattend_size: int = 21888,
        stft: bool = False,
        features: str = "none",
        hop_length: int = 64,
        raw_input: Optional[bool] = True,
        adapt_wavelet: bool = False,
    ) -> None:
        """Define network sturcture."""
        super(LearnDeepTestNet, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        freqs = torch.linspace(f_max, f_min, num_of_scales, device=device) / sample_rate
        self.flattend_size = flattend_size
        self.raw_input = raw_input
        self.features = features

        # only set log scale if raw cwt or stft are used
        if "lfcc" in features or "delta" in features or "all" in features:
            self.pre_log = False
            if "all" in features:
                channels = 60
            else:
                channels = 20
        else:
            self.pre_log = True
            channels = 32

        if stft:
            self.transform = STFTLayer(  # type: ignore
                n_fft=num_of_scales * 2 - 1,
                hop_length=hop_length,
                log_scale=self.pre_log,
            )
        else:
            self.transform = CWTLayer(  # type: ignore
                wavelet=wavelet,
                freqs=freqs,
                hop_length=hop_length,
                log_scale=self.pre_log,
                adapt_wavelet=adapt_wavelet,
            )

        self.lfcc = LFCC(
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
            num_of_scales=num_of_scales,
        )
        self.delta = ComputeDeltas()

        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, stride=2),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=5, stride=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattend_size, classes),
        )

        # LCNN from AVSpoofChallenge 2021
        self.lcnn = nn.Sequential(
            nn.Conv2d(1, 64, 5, 1, padding=2),
            MaxFeatureMap2D(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 1, 1, padding=0),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(32, affine=False),
            nn.Conv2d(32, 96, 3, 1, padding=1),
            MaxFeatureMap2D(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(48, affine=False),
            nn.Conv2d(48, 96, 1, 1, padding=0),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(48, affine=False),
            nn.Conv2d(48, 128, 3, 1, padding=1),
            MaxFeatureMap2D(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 1, 1, padding=0),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(64, affine=False),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(32, affine=False),
            nn.Conv2d(32, 64, 1, 1, padding=0),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(32, affine=False),
            nn.Conv2d(32, 64, 3, 1, padding=1),
            MaxFeatureMap2D(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.7),
        )

        self.lstm = nn.Sequential(
            BLSTMLayer((channels // 16) * 32, (channels // 16) * 32),
            BLSTMLayer((channels // 16) * 32, (channels // 16) * 32),
        )

        self.out = nn.Linear((channels // 16) * 32, classes)
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x) -> torch.Tensor:
        """Forward pass."""
        if self.raw_input:
            x = self.transform(x)

        if not self.pre_log:
            lfcc = self.lfcc(x)
            x = lfcc

        if "delta" in self.features or "all" in self.features:
            delta = self.delta(lfcc)
            x = delta

        if "double" in self.features or "all" in self.features:
            doubledelta = self.delta(delta)
            x = doubledelta

        if "all" in self.features:
            x = torch.hstack(
                (
                    lfcc.squeeze(),
                    delta.squeeze(),
                    doubledelta.squeeze(),
                )
            ).unsqueeze(1)

        if self.pre_log:
            x = self.cnn1(x)
            x = self.fc(x)
        else:
            x = self.lcnn(x.permute(0, 1, 3, 2))
            x = x.permute(0, 2, 1, 3).contiguous()
            shape = x.shape
            x = self.lstm(x.view(shape[0], shape[1], -1))
            x = self.out(x).mean(1)

        return self.logsoftmax(x)

    def get_name(self) -> str:
        """Return custom string identifier."""
        return "LearnDeepTestNet"


class LearnNet(nn.Module):
    """Deep CNN with 2D convolutions for detecting audio deepfakes.

    A little less params than LearnDeepTestNet.
    """

    def __init__(
        self,
        wavelet,
        classes: int = 2,
        f_min: float = 2000,
        f_max: float = 4000,
        num_of_scales: int = 150,
        sample_rate: int = 8000,
        batch_size: int = 256,
        flattend_size: int = 39168,
        stft: bool = False,
        hop_length: int = 1,
        raw_input: Optional[bool] = True,
    ) -> None:
        """Define network sturcture."""
        super(LearnNet, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        freqs = torch.linspace(f_max, f_min, num_of_scales, device=device) / sample_rate
        self.flattend_size = flattend_size
        self.raw_input = raw_input

        if stft:
            self.transform = STFTLayer(
                n_fft=num_of_scales * 2 - 1,
                hop_length=hop_length,
            )  # type: ignore
        else:
            self.transform = CWTLayer(  # type: ignore
                wavelet=wavelet,
                freqs=freqs,
                hop_length=hop_length,
            )

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(
                32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(
                32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=3),
            nn.BatchNorm2d(
                64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=3),
            nn.BatchNorm2d(
                128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattend_size, classes, bias=True),
        )
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x) -> torch.Tensor:
        """Forward pass."""
        if self.raw_input:
            x = self.transform(x)
        x = self.cnn(x)
        x = self.fc(x)
        return self.logsoftmax(x)

    def get_name(self) -> str:
        """Return custom string identifier."""
        return "LearnNet"


class OneDNet(nn.Module):
    """Deep CNN with 1D convolutions for detecting audio deepfakes."""

    def __init__(
        self,
        wavelet,
        classes: int = 2,
        f_min: float = 1000,
        f_max: float = 9500,
        num_of_scales: int = 150,
        sample_rate: int = 22050,
        batch_size: int = 128,
        stride: int = 2,
        flattend_size: int = 5440,
        stft: bool = False,
        hop_length: int = 1,
        raw_input: Optional[bool] = True,
    ) -> None:
        """Define network structure."""
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        freqs = torch.linspace(f_max, f_min, num_of_scales, device=device) / sample_rate

        self.flattend_size = flattend_size
        self.raw_input = raw_input

        if stft:
            self.transform = STFTLayer(
                n_fft=num_of_scales * 2 - 1,
                hop_length=hop_length,
            )  # type: ignore
        else:
            self.transform = CWTLayer(  # type: ignore
                wavelet=wavelet,
                freqs=freqs,
                hop_length=hop_length,
            )

        self.cnn = nn.Sequential(
            nn.Conv1d(num_of_scales, 32, kernel_size=53, stride=stride),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=stride),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=stride),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=stride),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=stride),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=stride),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AvgPool1d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattend_size, classes),  # 1184, 128, 4992, 9984
        )
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x) -> torch.Tensor:
        """Forward pass."""
        if self.raw_input:
            x = self.transform(x)
        x = x.squeeze(1)
        x = self.cnn(x)
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
    wavelet: ContinuousWavelet,
    model_name: str,
    nclasses: int = 2,
    batch_size: int = 128,
    f_min: float = 1000,
    f_max: float = 9500,
    sample_rate: int = 22050,
    num_of_scales: int = 150,
    flattend_size: int = 21888,
    stft: bool = False,
    features: str = "none",
    hop_length: int = 1,
    raw_input: Optional[bool] = True,
    adapt_wavelet: bool = False,
    in_channels: int = 1,
    channels: int = 32,
) -> LearnDeepTestNet | OneDNet | LearnNet:
    """Get torch module model with given parameters."""
    if model_name == "learndeepnet":
        model = LearnDeepTestNet(
            classes=nclasses,
            wavelet=wavelet,
            f_min=f_min,
            f_max=f_max,
            sample_rate=sample_rate,
            num_of_scales=num_of_scales,
            batch_size=batch_size,
            raw_input=raw_input,
            flattend_size=flattend_size,
            stft=stft,
            features=features,
            hop_length=hop_length,
            adapt_wavelet=adapt_wavelet,
        )  # type: ignore
    elif model_name == "lcnn":
        model = LCNN(
            classes=nclasses,
            in_channels=in_channels,
            lstm_channels=channels,
        )  # type: ignore
    elif model_name == "learnnet":
        model = LearnNet(
            classes=nclasses,
            wavelet=wavelet,
            f_min=f_min,
            f_max=f_max,
            sample_rate=sample_rate,
            num_of_scales=num_of_scales,
            batch_size=batch_size,
            raw_input=raw_input,
            flattend_size=flattend_size,
            stft=stft,
            hop_length=hop_length,
        )  # type: ignore
    elif model_name == "onednet":
        model = OneDNet(
            classes=nclasses,
            wavelet=wavelet,
            f_min=f_min,
            f_max=f_max,
            sample_rate=sample_rate,
            num_of_scales=num_of_scales,
            batch_size=batch_size,
            raw_input=raw_input,
            flattend_size=flattend_size,
            stft=stft,
            hop_length=hop_length,
        )  # type: ignore
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
