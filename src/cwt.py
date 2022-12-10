"""A torchaudio conform CWT call."""
import numpy as np
import ptwt
import pywt
import torch
from torchaudio.transforms import AmplitudeToDB

center_freq = 0.87
bandwith = 0.001
RES = 60
SAMPLE_RATE = 16000.0


class CWT(torch.nn.Module):
    """Create time-scale/frequency-representation of audio-signal with cwt.

    By default, this calculates the continuous wavelet transform on the DB-scaled scaleogram.

    Args:
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``22050``)
        n_lin (int, optional): Number of scales to be computed. (Default: ``50``)
        cut (bool, optional): Cut audio signal to equal length (for equal batch sizes). (Default: False)
        max_len (int, optional): If cut is True, the audio file is cut to this length. (Default: 4 sec)
    """

    def __init__(
        self,
        sample_rate: float = SAMPLE_RATE,
        n_lin: int = RES,
        cut: bool = False,
        max_len: int = 64600,
    ) -> None:
        """Calculate scales for cwt, set object params."""
        super().__init__()

        self.sample_rate = sample_rate
        self.sample_period = 1.0 / self.sample_rate
        self.n_lin = n_lin
        self.cut = cut
        self.max_len = max_len
        self.wavelet = f"shan{bandwith}-{center_freq}"
        self.transform = AmplitudeToDB(stype="power", top_db=80.0)

        nyquist_freq = self.sample_rate / 2.0  # maximum frequency that can be analyzed

        # equally spaced normalized frequencies to be analyzed
        freqs = np.linspace(nyquist_freq, 1, self.n_lin) / self.sample_rate
        self.scales = pywt.frequency2scale(self.wavelet, freqs)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return cwt audio signal of correct dimensions.

        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: scaleogram, time/freq representation of
                    dims: (1, number of scales, number of audio samples).
        """
        if self.cut is True:
            length = waveform.shape[1]
            if length >= self.max_len:
                waveform = waveform[:, : self.max_len]
            if length < self.max_len:
                num_repeats = int(self.max_len / length) + 1
                waveform = torch.tile(waveform, (1, num_repeats))[:, : self.max_len][0]

        sig, _freqs = ptwt.cwt(
            waveform, self.scales, self.wavelet, sampling_period=self.sample_period
        )

        scaleogram = sig.squeeze(1)
        scaleogram = torch.abs(scaleogram) ** 4
        scaleogram = self.transform(scaleogram)
        scaleogram = torch.unsqueeze(scaleogram, dim=0)
        return scaleogram
