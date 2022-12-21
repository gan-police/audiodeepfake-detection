"""A torchaudio conform CWT call."""
import numpy as np
import ptwt
import pywt
import torch
import torchaudio.functional as funct
from torchaudio.transforms import AmplitudeToDB

center_freq = 0.87
bandwith = 0.001


class CWT(torch.nn.Module):
    """Create time-scale/frequency-representation of audio-signal with cwt.

    By default, this calculates the continuous wavelet transform on the DB-scaled scaleogram.

    Port of torchaudio.transforms.LFCC but with CWT instead of STFT.

    Args:
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        n_lin (int, optional): Number of scales to be computed. (Default: ``60``)
        cut (bool, optional): Cut audio signal to equal length (for equal batch sizes). (Default: False)
        max_len (int, optional): If cut is True, the audio file is cut to this length. (Default: 4 sec)
        f_min (float, optional): Minimal frequency being analyzed.
        f_max (float, optional): Maximal frequency being analyzed.
    """

    def __init__(
        self,
        length: int,
        sample_rate: float = 16000.0,
        n_lfcc: int = 64,
        f_min: float = 80.0,
        f_max: float = 2000.0,
        resolution: int = 128,
    ) -> None:
        """Calculate scales for cwt, set object params."""
        super().__init__()

        self.sample_period = 1.0 / sample_rate
        self.wavelet = f"shan{bandwith}-{center_freq}"
        self.transform = AmplitudeToDB(stype="power")

        nyquist_freq = sample_rate / 2.0  # maximum frequency that can be analyzed

        if f_max >= nyquist_freq:
            f_max = nyquist_freq
        # equally spaced normalized frequencies to be analyzed
        freqs = np.linspace(f_max, f_min, resolution) / sample_rate
        self.scales = pywt.frequency2scale(self.wavelet, freqs)
        # self.scales = np.linspace(self.scales[0], self.scales[-1])
        # print("scales:", self.scales)

        n_filter = 128
        n_freqs = length
        filter_mat = funct.linear_fbanks(
            n_freqs=n_freqs,
            f_min=f_min,
            f_max=f_max,
            n_filter=n_filter,
            sample_rate=sample_rate,
        )
        self.filter_mat = filter_mat

        dct_mat = funct.create_dct(n_lfcc, n_filter, "ortho")
        self.dct_mat = dct_mat

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return lfcc using cwt of audio signal of correct dimensions.

        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: scaleogram, time/freq representation of
                    dims: (Channels, Number of Scales, n_filter)
        """
        sig, _freqs = ptwt.cwt(
            waveform, self.scales, self.wavelet, sampling_period=self.sample_period
        )
        sig = torch.abs(sig) ** 2

        sig_tr = sig.squeeze(1)
        sig_tr = torch.unsqueeze(sig_tr, dim=0)
        scale = torch.matmul(sig_tr, self.filter_mat.double())
        scale = self.transform(scale)
        lfcc_cwt = torch.matmul(scale, self.dct_mat.double()).transpose(-1, -2)
        return lfcc_cwt
