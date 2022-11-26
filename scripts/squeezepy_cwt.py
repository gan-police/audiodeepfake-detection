"""Ssqueezepy test script for SSQ_CWT of WafeFake Dataset Audiofiles.

For generating scalograms of ssq_cwt of audio samples generated with different Deep Learning algorithms

See also: https://github.com/OverLordGoldDragon/ssqueezepy
"""
import os
import sys

import librosa
import matplotlib.pyplot as plt
import ssqueezepy

DEBUG = True
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if DEBUG:
    # Set python path automatically to base directory
    sys.path.append(BASE_PATH)

import src.util as sp


def cwt_plot(sig, title) -> None:
    """Plot SSQ_CWT of given signal."""
    signal, t = sig
    # coeffs, wx, *_ = ssqueezepy.cwt(signal, wavelet="bump")

    # ssqueezepy.wavs()
    # {'hhhat', 'gmw', 'morlet', 'bump', 'cmhat'}
    coeffs, wx, *_ = ssqueezepy.ssq_cwt(signal, wavelet="bump")
    # coeffs, Sxo, *_ = ssq_stft(signal)
    # coeffs = np.flipud(coeffs)

    fig, ax = plt.subplots(1)
    ax.set_title(title)
    ax.set_xlabel("Zeit (Frame)")
    ax.set_ylabel("Skale")

    power_to_db = librosa.power_to_db(abs(coeffs) ** 2)
    # vmin = np.amin(power_to_db)
    # vmax = np.amax(power_to_db)
    vmin = -35
    vmax = -100
    im = ax.imshow(power_to_db, aspect="auto", cmap="turbo", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax)


if __name__ == "__main__":

    from_frame = 200
    to_frame = 4000

    wav_label = "LJ008-0217"
    data_base_dir = f"{BASE_PATH}/tests/data"
    audios = [
        f"real/{wav_label}.wav",
        f"ljspeech_melgan/{wav_label}_gen.wav",
        f"ljspeech_full_band_melgan/{wav_label}_gen.wav",
        f"ljspeech_multi_band_melgan/{wav_label}_gen.wav",
        f"ljspeech_hifiGAN/{wav_label}_gen.wav",
        f"ljspeech_waveglow/{wav_label}_gen.wav",
        f"ljspeech_parallel_wavegan/{wav_label}_gen.wav",
    ]

    titles = [
        "Original",
        "MelGAN",
        "Full-Band-MelGAN",
        "Multi-Band-MelGAN",
        "Hifi-GAN",
        "Waveglow",
        "Parallel WaveGAN",
    ]

    for i in range(len(audios)):
        path = f"{data_base_dir}/{audios[i]}"
        cwt_plot(sp.get_np_signal(path, from_frame, to_frame), titles[i])

    plt.show()
