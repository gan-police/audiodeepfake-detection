"""Module implementing wavelet related math functions.

The idea is to provide functionality to make the cwt useful
for audio analysis and gan-content recognition.
"""

import pywt
import ptwt
import torch
from torchaudio.transforms import Spectrogram

from .ptwt_continuous_transform import cwt


class CWTLayer(torch.nn.Module):
    """A base class for learnable Continuous Wavelets."""

    def __init__(
        self,
        wavelet,
        freqs: torch.Tensor,
        batch_size: int,
        log_scale: bool = True,
        log_offset: float = 1e-12,
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

    def forward(self, input) -> torch.Tensor:
        """Transform input into scale-time-representation.

        Returns:
            torch.Tensor: Scale-time transformed input tensor with dimensions
                (batch_size, channels, number of scales (freqs.shape[0]), time)
        """
        scales = (self.wavelet.center / self.freqs).detach()

        x = input.squeeze(1)
        sig = cwt(x, scales, self.wavelet)
        if self.log_scale:
            sig = torch.abs(sig) ** 2
            sig = 10.0 * torch.log(sig + self.log_offset)
            sig = sig.to(torch.float32)

        sig = sig.permute(1, 0, 2)
        scalgram = torch.unsqueeze(sig, dim=1)

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
        specgram = 10.0 * torch.log(specgram + self.log_offset)
        specgram = specgram.to(torch.float32)

        return specgram


def compute_pytorch_packet_representation(
    pt_data: torch.Tensor, wavelet_str: str = "sym8", max_lev: int = 8
):
    """Create a packet image to plot."""
    wavelet = pywt.Wavelet(wavelet_str)
    ptwt_wp_tree = ptwt.WaveletPacket(data=pt_data, wavelet=wavelet, mode="reflect")

    # get the pytorch decomposition
    wp_keys = ptwt_wp_tree.get_level(max_lev)
    packet_list = []
    for node in wp_keys:
        packet_list.append(ptwt_wp_tree[node])

    wp_pt = torch.stack(packet_list, dim=-1)
    return wp_pt
