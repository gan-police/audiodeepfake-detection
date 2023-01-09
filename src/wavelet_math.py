"""Module implementing wavelet related math functions.

The idea is to provide functionality to make the cwt useful
for audio analysis and gan-content recognition.
"""

import numpy as np
import ptwt
import torch
from torchaudio.transforms import AmplitudeToDB


def wavelet_preprocessing(
    audio: torch.Tensor,
    scales: np.ndarray,
    wavelet: str = "db1",
    log_scale: bool = False,
    cuda: bool = True,
) -> np.ndarray:
    """Preprocess audios by computing the wavelet packet representation.

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
