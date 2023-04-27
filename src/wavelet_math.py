"""Module implementing wavelet related math functions.

The idea is to provide functionality to make the cwt useful
for audio analysis and gan-content recognition.
"""
from typing import Optional

import torch
from torchaudio import functional
from torchaudio.transforms import AmplitudeToDB, Spectrogram

from .ptwt_continuous_transform import cwt


class CWTLayer(torch.nn.Module):
    """A base class for learnable Continuous Wavelets."""

    def __init__(
        self,
        wavelet,
        freqs: torch.Tensor,
        batch_size: int = 128,
        hop_length: int = 1,
        log_scale: bool = True,
        log_offset: float = 1e-6,
    ):
        """Initialize wavelet config.

        Args:
            wavelet: Wavelet used for continuous wavelet transform.
            freqs (torch.Tensor): Tensor holding desired frequencies to be calculated
                                  in CWT.
            batch_size (int): Internal batch size for CWT.
            log_scale (bool): Sets wether transformed audios are log scaled to decibel scale.
                              Default: True.
            log_offset (float): Offset for log scaling. (Default: 10e-13)
        """
        super().__init__()
        self.freqs = freqs
        self.batch_size = batch_size
        self.log_scale = log_scale
        self.wavelet = wavelet
        self.log_offset = log_offset
        self.hop_length = hop_length

    def forward(self, input) -> torch.Tensor:
        """Transform input into scale-time-representation.

        Returns:
            torch.Tensor: Scale-time transformed input tensor with dimensions
                (batch_size, channels, number of scales (freqs.shape[0]), time)
        """
        scales = (self.wavelet.center / self.freqs).detach()

        x = input.squeeze(1)
        sig = cwt(x, scales, self.wavelet)
        sig = torch.abs(sig) ** 2

        if self.log_scale:
            sig = 10.0 * torch.log(sig + self.log_offset)

        sig = sig.to(torch.float32)

        sig = sig.permute(1, 0, 2)
        scalgram = torch.unsqueeze(sig, dim=1)

        if self.hop_length != 1:
            scalgram = scalgram[:, :, :, :: self.hop_length]

        return scalgram


class STFTLayer(torch.nn.Module):
    """A base class for STFT transformation."""

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 1,
        log_scale: bool = True,
        log_offset: float = 1e-6,
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
        self.transform = Spectrogram(n_fft=n_fft, hop_length=hop_length)
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
            log_offset = 1e-6
            specgram = torch.log(specgram + log_offset)
        else:
            specgram = self.amplitude_to_DB(specgram)

        lfcc = torch.matmul(specgram.transpose(-2, -1), self.dct_mat)  # type: ignore

        return lfcc.transpose(-2, -1)
