"""Module implementing wavelet related math functions.

The idea is to provide functionality to make the cwt useful
for audio analysis and gan-content recognition.
"""
from typing import Union

import numpy as np
import ptwt
import torch
from pywt import ContinuousWavelet
from torchaudio.transforms import AmplitudeToDB

from .ptwt_continuous_transform import cwt


def wavelet_preprocessing(
    audio: torch.Tensor,
    scales: np.ndarray,
    wavelet: Union[ContinuousWavelet, str] = "cmor4.6-0.87",
    log_scale: bool = False,
    cuda: bool = True,
) -> np.ndarray:
    """Preprocess audios by computing the continuous wavelet transform.

    The raw as well as an absolute log scaled version can be computed.

    Args:
        audio (np.ndarray): An audio of shape (C, S, L)
        wavelet (str): A pywt-compatible wavelet string. Defaults to 'db1'.
        log_scale (bool): Use log-scaling if True.
            Log-scaled coefficients aren't invertible. Defaults to False.
        cuda (bool): If False computations take place on the cpu.

    Returns:
        [np.ndarray]: The wavelet coefficents [channels, num of scales, audio length].
    """
    if cuda:
        audio = audio.cuda()

    # transform  to C, H, W
    sig, _freqs = ptwt.cwt(audio, scales, wavelet)
    scalgram = sig.squeeze(1)
    if log_scale:
        # scalgram = torch.abs(scalgram)
        # scalgram = torch.log(scalgram + eps)
        scalgram = torch.abs(scalgram) ** 2
        scalgram = AmplitudeToDB(stype="power", top_db=80.0)(scalgram)
    scalgram = scalgram.to(torch.float32)
    scalgram = torch.unsqueeze(scalgram, dim=0)

    return scalgram


def wavelet_direct(
    audio: torch.Tensor,
    scales: torch.Tensor,
    wavelet: Union[ContinuousWavelet, str] = "cmor4.6-0.87",
    log_scale: bool = False,
    cuda: bool = True,
) -> torch.Tensor:
    """Preprocess audios by computing the continuous wavelet transform.

    The raw as well as an absolute log scaled version can be computed.

    Args:
        audio (np.ndarray): An audio of shape (C, S, L)
        wavelet (str): A pywt-compatible wavelet string. Defaults to 'db1'.
        log_scale (bool): Use log-scaling if True.
            Log-scaled coefficients aren't invertible. Defaults to False.
        cuda (bool): If False computations take place on the cpu.

    Returns:
        [np.ndarray]: The wavelet coefficents [channels, num of scales, audio length].
    """
    if cuda:
        audio = audio.cuda()

    # transform  to C, H, W
    sig = cwt(audio, scales, wavelet)
    scalgram = sig.squeeze(1)
    if log_scale:
        scalgram = torch.abs(scalgram) ** 2
        scalgram = 10.0 * torch.log(scalgram + 10e-13)
        # scalgram = AmplitudeToDB(stype="power", top_db=80.0)(scalgram)
    scalgram = scalgram.to(torch.float32)
    scalgram = torch.unsqueeze(scalgram, dim=0)

    return scalgram.cpu()


def learn_wavelet(
    audio: torch.Tensor,
    freqs: torch.Tensor,
    wavelet: Union[ContinuousWavelet, str] = "cmor4.6-0.87",
    log_scale: bool = False,
    cuda: bool = True,
    batch_size: int = 256,
) -> torch.Tensor:
    """Preprocess audios by computing the continuous wavelet transform.

    The raw as well as an absolute log scaled version can be computed.

    Args:
        audio (np.ndarray): An audio of shape (C, S, L)
        wavelet (str): A pywt-compatible wavelet string. Defaults to 'db1'.
        log_scale (bool): Use log-scaling if True.
            Log-scaled coefficients aren't invertible. Defaults to False.
        cuda (bool): If False computations take place on the cpu.

    Returns:
        [np.ndarray]: The wavelet coefficents [channels, num of scales, audio length].
    """
    if cuda:
        audio = audio.cuda()

    scales = (wavelet.center / freqs).detach()

    # unbatch and rebatch...
    scalgram = torch.zeros(
        (audio.shape[0], audio.shape[1], freqs.shape[0], audio.shape[2]),
        device=audio.device,
    )
    for i in range(batch_size):
        # transform  to C, H, W
        sig = cwt(audio[i], scales, wavelet)
        sig = sig.squeeze(1)
        if log_scale:
            # scalgram = torch.abs(scalgram)
            # scalgram = torch.log(scalgram + eps)
            sig = torch.abs(sig) ** 2
            sig = 10.0 * torch.log(sig + 10e-13)
            # scalgram = AmplitudeToDB(stype="power", top_db=80.0)(scalgram)
        sig = sig.to(torch.float32)
        scalgram[i] = torch.unsqueeze(sig, dim=0)

    return scalgram


class CWTLayer(torch.nn.Module):
    """A base class for learnable Continuous Wavelets."""

    def __init__(self, wavelet, freqs, batch_size, log_scale=True, learn_wavelet=True):
        """Initialize wavelet config."""
        super().__init__()
        self.freqs = freqs
        self.batch_size = batch_size
        self.log_scale = log_scale
        self.wavelet = wavelet
        self.learn_wavelet = learn_wavelet

    def forward(self, input):
        """Transform input into scale-time-representation."""
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
