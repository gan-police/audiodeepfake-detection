"""Module implementing wavelet related math functions.

The idea is to provide functionality to make the cwt useful
for audio analysis and gan-content recognition.
"""
import os
import pickle
from math import log
from typing import Optional

import numpy as np
import ptwt
import pywt
import torch
import torchvision
from torchaudio import functional
from torchaudio.transforms import AmplitudeToDB, ComputeDeltas, Spectrogram
from tqdm import tqdm

from .data_loader import LearnWavefakeDataset, WelfordEstimator


class STFTLayer(torch.nn.Module):
    """A base class for STFT transformation."""

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 1,
        log_offset: float = 1e-12,
        log_scale: bool = False,
        power: float = 2.0,
    ):
        """Initialize config.

        Args:
            n_fft (int): Size of FFT, creates n_fft // 2 + 1 bins. (Default: 512)
            hop_length (int): Length of hop between STFT windows. (Default: 1)
            log_scale (bool): Sets wether transformed audios are log scaled.
                              Default: True.
            log_offset (float): Offset for log scaling. (Default: 1e-12)
        """
        super().__init__()
        self.transform = Spectrogram(
            n_fft=n_fft, hop_length=hop_length, power=power
        ).cuda()
        self.log_scale = log_scale
        self.log_offset = log_offset

    def forward(self, input) -> torch.Tensor:
        """Transform input into frequency-time-representation.

        Returns:
            torch.Tensor: freq-time transformed input tensor with dimensions
                (batch_size, channels, number of frequencies (n_fft//2 + 1), time)
        """
        specgram = self.transform(input)

        if self.log_scale:
            specgram = torch.log(specgram + 1e-12)

        return specgram


class LFCC(torch.nn.Module):
    """Create the linear-frequency cepstral coefﬁcients (LFCC features) from an audio signal.

    By default, this calculates the LFCC features on the DB-scaled linear scaled spectrogram
    to be consistent with the MFCC implementation.

    Args:
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        n_lin (int, optional): Number of linear filterbanks. (Default: ``128``)
        n_lfcc (int, optional): Number of lfc coefficients to retain. (Default: ``40``)
        f_min (float, optional): Minimum frequency. (Default: ``0.``)
        f_max (float or None, optional): Maximum frequency. (Default: ``None``)
        dct_type (int, optional): type of DCT (discrete cosine transform) to use. (Default: ``2``)
        norm (str, optional): norm to use. (Default: ``'ortho'``)
        log_lf (bool, optional): whether to use log lf-spectrograms instead of db-scaled. (Default: ``False``)
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_lin: int = 20,
        n_lfcc: int = 20,
        f_min: float = 0.0,
        f_max: Optional[float] = 11025,
        dct_type: int = 2,
        norm: str = "ortho",
        log_lf: bool = True,
        num_of_scales: int = 150,
    ) -> None:
        """Initialize Lfcc module.

        Raises:
            ValueError: If unsupported DCT type is given.
        """
        super().__init__()

        supported_dct_types = [2]
        if dct_type not in supported_dct_types:
            raise ValueError("DCT type not supported: {}".format(dct_type))

        self.sample_rate = sample_rate
        self.n_lin = n_lin
        self.n_fft = num_of_scales * 2 - 1
        self.n_lfcc = n_lfcc
        self.f_min = f_min
        self.f_max = f_max
        self.dct_type = dct_type
        self.norm = norm
        self.top_db = 80.0
        self.amplitude_to_DB = AmplitudeToDB("power", self.top_db)

        if self.n_lfcc > self.n_lin:
            raise ValueError("Cannot select more LFCC coefficients than # lin bins")

        filter_mat = functional.linear_fbanks(
            n_freqs=num_of_scales,
            f_min=self.f_min,
            f_max=self.f_max,
            n_filter=self.n_lin,
            sample_rate=self.sample_rate,
        )
        self.register_buffer("filter_mat", filter_mat)

        dct_mat = functional.create_dct(n_lfcc, self.n_lin, self.norm)
        self.register_buffer("dct_mat", dct_mat)
        self.log_lf = log_lf

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Calculate LFCC assuming an transformed input.

        Args:
             input (Tensor): Scaleogram or Spectrogram of audio (the cwt,stft transformed signal).

        Returns:
            Tensor: specgram_lf_db of size (..., ``n_lfcc``, time).
        """
        shape = input.size()
        specgram = input.reshape(-1, shape[-2], shape[-1])

        specgram = torch.matmul(specgram.transpose(1, 2), self.filter_mat)  # type: ignore
        specgram = specgram.transpose(1, 2)

        # unpack batch
        specgram = specgram.unsqueeze(1)

        if self.log_lf:
            log_offset = 1e-12
            specgram = torch.log(specgram + log_offset)
        else:
            specgram = self.amplitude_to_DB(specgram)

        lfcc = torch.matmul(specgram.transpose(-2, -1), self.dct_mat)  # type: ignore

        return lfcc.transpose(-2, -1)


def compute_pytorch_packet_representation(
    pt_data: torch.Tensor,
    wavelet: pywt.Wavelet,
    max_lev: int = 8,
    log_scale: bool = False,
    loss_less: bool = False,
    power: float = 2.0,
):
    """Create a packet image."""
    ptwt_wp_tree = ptwt.WaveletPacket(data=pt_data, wavelet=wavelet, mode="reflect")

    # get the pytorch decomposition
    wp_keys = ptwt_wp_tree.get_level(max_lev)
    packet_list = []
    for node in wp_keys:
        packet_list.append(ptwt_wp_tree[node])

    wp_pt = torch.stack(packet_list, dim=-1)

    if log_scale:
        wp_pt_log = torch.log(torch.abs(wp_pt).pow(power) + 1e-12)

        if loss_less:
            sign_pattern = ((wp_pt < 0).type(torch.float32) * (-1) + 0.5) * 2
            wp_pt = torch.stack([wp_pt_log, sign_pattern], 1)
        else:
            wp_pt = wp_pt_log.unsqueeze(1)
    else:
        wp_pt = wp_pt.unsqueeze(1)

    return wp_pt


class Packets(torch.nn.Module):
    """Compute wavelet packet representation as module."""

    def __init__(
        self,
        wavelet_str: str = "sym8",
        max_lev: int = 8,
        log_scale: bool = False,
        loss_less: bool = False,
        power: float = 2.0,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.wavelet = pywt.Wavelet(wavelet_str)
        self.max_lev = max_lev
        self.log_scale = log_scale
        self.loss_less = loss_less
        self.power = power

    def forward(self, pt_data: torch.Tensor) -> torch.Tensor:
        """Forward packet representation."""
        return compute_pytorch_packet_representation(
            pt_data,
            self.wavelet,
            self.max_lev,
            self.log_scale,
            self.loss_less,
            self.power,
        ).permute(0, 1, 3, 2)


def get_transforms(
    args,
    data_prefix,
    features,
    device,
    normalization,
    pbar: bool = False,
) -> tuple[torch.nn.Sequential, torch.nn.Sequential]:
    """Initialize transformations and normalize."""
    if args.transform == "stft":
        transform = STFTLayer(  # type: ignore
            n_fft=args.num_of_scales * 2 - 1,
            hop_length=args.hop_length,
            log_scale=args.features == "none" and args.log_scale,
            power=args.power,
        ).cuda()
    elif args.transform == "packets":
        transform = Packets(  # type: ignore
            wavelet_str=args.wavelet,
            max_lev=int(log(args.num_of_scales, 2)),
            log_scale=args.features == "none" and args.log_scale,
            loss_less=False if args.loss_less == "False" else True,
            power=args.power,
        )

    lfcc = LFCC(
        sample_rate=args.sample_rate,
        f_min=args.f_min,
        f_max=args.f_max,
        num_of_scales=args.num_of_scales,
    )

    transforms = torch.nn.Sequential(transform)

    if "lfcc" in features:
        transforms.append(lfcc)

    if "delta" in features:
        transforms.append(ComputeDeltas())

    if "doubledelta" in features:
        transforms.append(ComputeDeltas())

    norm_dir = (
        args.log_dir
        + "/norms/"
        + args.data_prefix.replace("/", "_")
        + "_"
        + args.transform
        + "_"
        + args.wavelet
        + "_"
        + str(args.num_of_scales)
        + "_"
        + str(args.power)
        + "_"
        + "loss_less"
        if args.loss_less
        else ""
    )

    if os.path.exists(f"{norm_dir}_mean_std.pkl"):
        print("Loading pre calculated mean and std from file.")
        with open(f"{norm_dir}_mean_std.pkl", "rb") as file:
            mean, std = pickle.load(file)
            mean = torch.from_numpy(mean.astype(np.float32)).to(device)
            std = torch.from_numpy(std.astype(np.float32)).to(device)
    elif normalization:
        print("computing mean and std values.", flush=True)
        dataset = LearnWavefakeDataset(data_prefix + "_train")
        norm_dataset_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=8000,
            shuffle=False,
        )
        welford = WelfordEstimator()
        with torch.no_grad():
            for batch in tqdm(
                iter(norm_dataset_loader),
                desc="comp normalization",
                total=len(norm_dataset_loader),
                disable=not pbar,
            ):
                freq_time_dt = transforms(batch["audio"].cuda())
                welford.update(freq_time_dt.permute(0, 3, 2, 1))
            mean, std = welford.finalize()
            with open(f"{norm_dir}_mean_std.pkl", "wb") as f:
                pickle.dump([mean.cpu().numpy(), std.cpu().numpy()], f)
    else:
        print("Using default mean and std.")
        mean = torch.tensor(args.mean, device=device)
        std = torch.tensor(args.std, device=device)
    print("mean", mean, "std:", std)

    normalize = torch.nn.Sequential(
        torchvision.transforms.Normalize(mean, std),
    )

    return transforms, normalize
