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

from .data_loader import WelfordEstimator, get_costum_dataset
from .utils import DotDict


class STFTLayer(torch.nn.Module):
    """A base class for STFT transformation."""

    def __init__(
        self,
        n_fft: int = 511,
        hop_length: int = 220,
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

        self.transform = Spectrogram(n_fft=n_fft, hop_length=hop_length, power=power)

        if torch.cuda.is_available():
            self.transform.cuda()

        self.log_scale = log_scale
        self.log_offset = log_offset
        self.block_norm_dict = None

    def forward(self, input) -> tuple[torch.Tensor, None]:
        """Transform input into frequency-time-representation.

        Returns:
            torch.Tensor: freq-time transformed input tensor with dimensions
                (batch_size, channels, number of frequencies (n_fft//2 + 1), time)
        """
        specgram = self.transform(input)

        if self.log_scale:
            specgram = torch.log(specgram + 1e-12)

        return specgram, None


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
    block_norm: bool = False,
    compute_welford: bool = False,
    block_norm_dict=None,
) -> tuple[torch.Tensor, dict]:
    """Create a packet image.

    Adaption of ptwt (https://github.com/v0lta/PyTorch-Wavelet-Toolbox/tree/main/src/ptwt).
    """
    ptwt_wp_tree = ptwt.WaveletPacket(data=pt_data, wavelet=wavelet, mode="reflect")

    # get the pytorch decomposition
    wp_keys = ptwt_wp_tree.get_level(max_lev)
    packet_list = []

    if block_norm_dict is None:
        block_norm_dict = {}

    for node in wp_keys:
        node_wp = ptwt_wp_tree[node]

        if compute_welford:
            if node in block_norm_dict.keys():
                block_norm_dict[node].update(node_wp.unsqueeze(-1))
            else:
                welfi = WelfordEstimator()
                welfi.update(node_wp.unsqueeze(-1))
                block_norm_dict[node] = welfi

        if block_norm:
            node_wp = node_wp / torch.max(torch.abs(node_wp))
        packet_list.append(node_wp)

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

    return wp_pt, block_norm_dict


class Packets(torch.nn.Module):
    """Compute wavelet packet representation as module."""

    def __init__(
        self,
        wavelet_str: str = "sym8",
        max_lev: int = 8,
        log_scale: bool = False,
        loss_less: bool = False,
        power: float = 2.0,
        block_norm: bool = False,
        compute_welford: bool = False,
        block_norm_dict=None,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.wavelet = pywt.Wavelet(wavelet_str)
        self.max_lev = max_lev
        self.log_scale = log_scale
        self.loss_less = loss_less
        self.power = power

        self.block_norm = block_norm
        self.compute_welford = compute_welford
        self.block_norm_dict = block_norm_dict

    def forward(self, pt_data: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Forward packet representation."""
        packets, block_norm_dict = compute_pytorch_packet_representation(
            pt_data,
            self.wavelet,
            self.max_lev,
            self.log_scale,
            self.loss_less,
            self.power,
            block_norm=self.block_norm,
            compute_welford=self.compute_welford,
            block_norm_dict=self.block_norm_dict,
        )

        return packets.permute(0, 1, 3, 2), block_norm_dict


def get_transforms(
    args: DotDict,
    features: str,
    device: str,
    normalization: bool,
    pbar: bool = False,
    verbose: bool = True,
) -> tuple[torch.nn.Sequential, torch.nn.Sequential]:
    """Initialize transformations and normalize.

    Args:
        args (DotDict): Current configuration.
        features (str): If lfcc or delta or double delta features should be applied.
        device (str): Which device the model uses.
        normalization (bool): If the data should be normalized (only if norm does not exist).
        pbar (bool): If tqdm bars should be used. Defaults to False.
        verbose (bool): If more verbose logging should be enabled. Defaults to True.

    Returns:
        tuple[torch.nn.Sequential, torch.nn.Sequential]: The frequency-space transform module and
                                                         normalization module.
    """
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
            block_norm_dict=None,
            block_norm=False,
            compute_welford=True,
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

    loss_less = "_loss_less" if args.loss_less == "True" else ""

    norm_dir = (
        args.log_dir
        + "/norms/"
        + args.data_path.replace("/", "_")
        + "_"
        + "-".join(args.only_use)
        + "_"
        + args.transform
        + "_"
        + args.wavelet
        + "_"
        + str(args.num_of_scales)
        + "_"
        + str(args.power)
        + loss_less
        + "_"
        + str(args.sample_rate)
        + "_"
        + str(args.seconds)
        + "secs"
    )

    if os.path.exists(f"{norm_dir}_mean_std.pkl") and args.block_norm is False:
        if verbose:
            print("Loading pre calculated mean and std from file.")
        with open(f"{norm_dir}_mean_std.pkl", "rb") as file:
            mean, std = pickle.load(file)
            mean = torch.from_numpy(mean.astype(np.float32)).to(device)
            std = torch.from_numpy(std.astype(np.float32)).to(device)
    elif os.path.exists(f"{norm_dir}_mean_std_bn.pt") and args.block_norm is True:
        if verbose:
            print("Loading pre calculated mean and std from file.")
        welford_dict = torch.load(f"{norm_dir}_mean_std_bn.pt", map_location=device)
        for k, _ in welford_dict.items():
            welford_dict[k]["mean"].cuda(non_blocking=True)
            welford_dict[k]["std"].cuda(non_blocking=True)
    elif normalization:
        if verbose:
            print("computing mean and std values.", flush=True)
        welford_dict, mean, std = calc_normalization(args, pbar, transforms, norm_dir)
    else:
        if verbose:
            print("Using default mean and std.")
        mean = torch.tensor(args.mean, device=device)
        std = torch.tensor(args.std, device=device)

    if args.block_norm:
        mean = 0.0
        std = 1.0
        transforms[0].block_norm_dict = welford_dict
        transforms[0].compute_welford = False
        transforms[0].block_norm = True

    normalize = torch.nn.Sequential(
        torchvision.transforms.Normalize(mean, std),
    )

    return transforms, normalize


def calc_normalization(
    args: DotDict,
    pbar: bool,
    transforms: torch.nn.Sequential,
    norm_dir: str,
) -> tuple:
    """Calculate normalization of training dataset.

    Args:
        args (DotDict): Current configuration.
        pbar (bool): True if tqdm bars should be enabled.
        transforms (torch.nn.Sequential): The transforms to be applied to a dataset sample.
        norm_dir (str): Path to directory where to save the mean and std.

    Returns:
        tuple: The block norm dictionary, the mean and std. blocknorm is None if block norm is disabled.
    """
    dataset = get_costum_dataset(
        data_path=args.data_path,
        ds_type="train",
        only_use=args.only_use,
        save_path=args.save_path,
        limit=args.limit_train[0],
        asvspoof_name=(
            f"{args.asvspoof_name}_T"
            if args.asvspoof_name is not None and "LA" in args.asvspoof_name
            else args.asvspoof_name
        ),
        file_type=args.file_type,
        resample_rate=args.sample_rate,
        seconds=args.seconds,
    )
    norm_dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4000,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    welford = WelfordEstimator()

    with torch.no_grad():
        for batch in tqdm(
            iter(norm_dataset_loader),
            desc="comp normalization",
            total=len(norm_dataset_loader),
            disable=not pbar,
        ):
            freq_time_dt, welford_dict = transforms(
                batch["audio"].cuda(non_blocking=True)
            )
            transforms[0].block_norm_dict = welford_dict
            welford.update(freq_time_dt.permute(0, 3, 2, 1))
        mean, std = welford.finalize()
        if args.block_norm:
            for key in welford_dict.keys():
                mean, std = welford_dict[key].finalize()
                welford_dict[key] = {"mean": mean, "std": std}

            torch.save(welford_dict, f"{norm_dir}_mean_std_bn.pkl")
        else:
            with open(f"{norm_dir}_mean_std.pkl", "wb") as f:
                pickle.dump([mean.cpu().numpy(), std.cpu().numpy()], f)

    return welford_dict, mean, std
