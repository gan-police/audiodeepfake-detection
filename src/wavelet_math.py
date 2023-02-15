"""Module implementing wavelet related math functions.

The idea is to provide functionality to make the cwt useful
for audio analysis and gan-content recognition.
"""
import torch

from .ptwt_continuous_transform import cwt


class CWTLayer(torch.nn.Module):
    """A base class for learnable Continuous Wavelets."""

    def __init__(
        self,
        wavelet,
        freqs: torch.Tensor,
        batch_size: int,
        log_scale: bool = True,
    ):
        """Initialize wavelet config.

        Args:
            wavelet: Wavelet used for continuous wavelet transform.
            freqs (torch.Tensor): Tensor holding desired frequencies to be calculated
                                  in CWT.
            batch_size (int): Internal batch size for CWT.
            log_scale (bool): Sets wether transformed audios are log scaled to decibel scale.
                              Default: True.
        """
        super().__init__()
        self.freqs = freqs
        self.batch_size = batch_size
        self.log_scale = log_scale
        self.wavelet = wavelet

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
            sig = 10.0 * torch.log(sig + 10e-13)
            sig = sig.to(torch.float32)

        sig = sig.permute(1, 0, 2)
        scalgram = torch.unsqueeze(sig, dim=1)

        return scalgram
