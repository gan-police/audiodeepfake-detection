"""A torchaudio conform CWT call."""
import numpy as np
import ptwt
import pywt
import torch
from torchaudio.transforms import AmplitudeToDB


class CWT(torch.nn.Module):
    """Create time-scale/frequency-representation of audio-signal with cwt.

    By default, this calculates the continuous wavelet transform on the DB-scaled scaleogram.

    Args:
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``22050``)
        n_lin (int, optional): Number of scales to be computed. (Default: ``50``)
        cut (bool, optional): Cut audio signal to equal length (for equal batch sizes). (Default: False)
        max_len (int, optional): If cut is True, the audio file is cut to this length. (Default: 4 sec)
        f_min (float, optional): Minimal frequency being analyzed.
        f_max (float, optional): Maximal frequency being analyzed.
    """

    def __init__(
        self,
        sample_rate: float = 16000.0,
        n_lin: int = 60,
        f_min: float = 80.0,
        f_max: float = 2000.0,
        wavelet: str = "cmor0.5-1.0",
    ) -> None:
        """Calculate scales for cwt, set object params."""
        super().__init__()

        self.sample_rate = sample_rate
        self.sample_period = 1.0 / self.sample_rate
        self.n_lin = n_lin
        self.wavelet = wavelet
        self.transform = AmplitudeToDB(stype="power", top_db=80.0)

        nyquist_freq = self.sample_rate / 2.0  # maximum frequency that can be analyzed

        if f_max >= nyquist_freq:
            f_max = nyquist_freq
        if f_min >= f_max:
            f_min = 80.0

        # equally spaced normalized frequencies to be analyzed
        freqs = np.linspace(f_max, f_min, self.n_lin) / self.sample_rate
        self.scales = pywt.frequency2scale(self.wavelet, freqs)
        # self.scales = np.linspace(self.scales[0], self.scales[-1], self.n_lin)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return cwt audio signal of correct dimensions.

        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            torch.Tensor: scaleogram, dims (1, number of scales, number of audio samples).
        """
        sig, _freqs = ptwt.cwt(
            waveform, self.scales, self.wavelet, sampling_period=self.sample_period
        )

        scaleogram = sig.squeeze(1)
        scaleogram = torch.abs(scaleogram) ** 2
        scaleogram = scaleogram.to(torch.float32)
        scaleogram = self.transform(scaleogram)
        scaleogram = torch.unsqueeze(scaleogram, dim=0)
        return scaleogram
