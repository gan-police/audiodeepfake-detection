"""Module implementing wavelet related math functions.

The idea is to provide functionality to make the cwt useful
for audio analysis and gan-content recognition.
"""
from typing import Optional

import ptwt
import pywt
import torch
import torchvision
from torchaudio import functional
from torchaudio.transforms import AmplitudeToDB, ComputeDeltas, Spectrogram
from tqdm import tqdm

from .data_loader import LearnWavefakeDataset, WelfordEstimator
from .ptwt_continuous_transform import cwt


class CWTLayer(torch.nn.Module):
    """A base class for learnable Continuous Wavelets."""

    def __init__(
        self,
        wavelet,
        freqs: torch.Tensor,
        hop_length: int = 1,
        log_scale: bool = True,
        log_offset: float = 1e-12,
        adapt_wavelet: bool = False,
    ):
        """Initialize wavelet config.

        Args:
            wavelet: Wavelet used for continuous wavelet transform.
            freqs (torch.Tensor): Tensor holding desired frequencies to be calculated
                                  in CWT.
            log_scale (bool): Sets wether transformed audios are log scaled to decibel scale.
                              Default: True.
            log_offset (float): Offset for log scaling. (Default: 10e-13)
        """
        super().__init__()
        self.freqs = freqs
        self.log_scale = log_scale
        self.wavelet = wavelet
        self.log_offset = log_offset
        self.hop_length = hop_length
        self.scales = (self.wavelet.center.cpu() / self.freqs).detach()
        # self.scales = torch.linspace(1.0, 32.0, freqs.shape[0]).detach()
        self.adapt_wavelet = adapt_wavelet

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Transform input into scale-time-representation.

        Returns:
            torch.Tensor: Scale-time transformed input tensor with dimensions
                (batch_size, channels, number of scales (freqs.shape[0]), time)
        """
        if self.adapt_wavelet:
            # recompute scales if wavelet changes
            self.scales = (self.wavelet.center.cpu() / self.freqs).detach()

        x = input.squeeze(1)
        sig = cwt(x, self.scales, self.wavelet)
        sig = torch.abs(sig) ** 2

        if self.log_scale:
            sig = 10.0 * torch.log(sig + self.log_offset)

        sig = sig.to(torch.float32)

        sig = sig.permute(1, 0, 2)
        scalgram = torch.unsqueeze(sig, dim=1)

        scalgram = scalgram[:, :, :, :: self.hop_length]

        return scalgram


class STFTLayer(torch.nn.Module):
    """A base class for STFT transformation."""

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 1,
        log_scale: bool = True,
        log_offset: float = 1e-12,
    ):
        """Initialize config.

        Args:
            n_fft (int): Size of FFT, creates n_fft // 2 + 1 bins. (Default: 512)
            hop_length (int): Length of hop between STFT windows. (Default: 1)
            log_scale (bool): Sets wether transformed audios are log scaled to decibel scale.
                              Default: True.
            log_offset (float): Offset for log scaling. (Default: 10e-13)
        """
        super().__init__()
        self.transform = Spectrogram(n_fft=n_fft, hop_length=hop_length).cuda()
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
            specgram = 10.0 * torch.log(specgram + self.log_offset)

        specgram = specgram.to(torch.float32)

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
    block_norm: bool = False,
    log_scale: bool = False,
):
    """Create a packet image."""
    ptwt_wp_tree = ptwt.WaveletPacket(data=pt_data, wavelet=wavelet, mode="reflect")

    # get the pytorch decomposition
    wp_keys = ptwt_wp_tree.get_level(max_lev)
    packet_list = []
    for node in wp_keys:
        packet_list.append(ptwt_wp_tree[node])

    if block_norm:
        packet_list = [wp / torch.max(torch.abs(wp)) for wp in packet_list]

    wp_pt = torch.stack(packet_list, dim=-1)

    if log_scale:
        wp_pt = torch.log(torch.abs(wp_pt) + 1e-12)

    return wp_pt


class Packets(torch.nn.Module):
    """Compute wavelet packet representation as module."""

    def __init__(
        self,
        wavelet_str: str = "sym8",
        max_lev: int = 8,
        block_norm=False,
        log_scale=False,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.wavelet = pywt.Wavelet(wavelet_str)
        self.max_lev = max_lev
        self.block_norm = block_norm
        self.log_scale = log_scale

    def forward(self, pt_data: torch.Tensor) -> torch.Tensor:
        """Forward packet representation."""
        return compute_pytorch_packet_representation(
            pt_data,
            self.wavelet,
            self.max_lev,
            self.block_norm,
            self.log_scale,
        ).unsqueeze(1)


def get_transforms(
    args,
    data_prefix,
    features,
    device,
    wavelet,
    normalization,
    pbar: bool = False,
) -> tuple[torch.nn.Sequential, torch.nn.Sequential]:
    """Initialize transformations and normalize."""
    if args.transform == "stft":
        transform = STFTLayer(  # type: ignore
            n_fft=args.num_of_scales * 2 - 1,
            hop_length=args.hop_length,
            log_scale=args.features == "none",
        ).cuda()
    elif args.transform == "cwt":
        freqs = (
            torch.linspace(args.f_max, args.f_min, args.num_of_scales, device=device)
            / args.sample_rate
        )
        transform = CWTLayer(  # type: ignore
            wavelet=wavelet,
            freqs=freqs,
            hop_length=args.hop_length,
            log_scale=args.features == "none",
        )

    elif args.transform == "packets":
        transform = Packets(  # type: ignore
            wavelet_str="sym8",
            max_lev=7,
            log_scale=args.features == "none",
            block_norm=False,
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

    if normalization:
        print("computing mean and std values.", flush=True)
        dataset = LearnWavefakeDataset(data_prefix + "_train")
        norm_dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=8000)
        welford = WelfordEstimator()
        with torch.no_grad():
            for batch in tqdm(
                iter(norm_dataset_loader),
                desc="comp normalization",
                total=len(norm_dataset_loader),
                disable=not pbar,
            ):
                freq_time_dt = transforms(batch["audio"].cuda())
                welford.update(freq_time_dt.squeeze(1).unsqueeze(-1))
            mean, std = welford.finalize()
    else:
        mean = torch.tensor(-6.717658042907715, device=device)
        std = torch.tensor(2.4886956214904785, device=device)
    print("mean", mean.item(), "std:", std.item())

    normalize = torch.nn.Sequential(
        torchvision.transforms.Normalize(mean, std),
    )

    return transforms, normalize
