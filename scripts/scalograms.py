"""Generate scalogram plots of CWT of audios from Wavefake-Dataset.

See Frank, Schönherr (2021): WaveFake: A Data Set to Facilitate Audio Deepfake Detection [FS21]

Simple script that generates scalograms that show apperent differences between
original audio files and the corresponding audio samples that [FS21] generated with different
Deep Learning methods and architectures that already exist, e.g. MelGAN, Waveglow, Hifi-GAN.

"""
import os
import sys

import numpy as np

DEBUG = True
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if DEBUG:
    # Set python path automatically to base directory
    sys.path.append(BASE_PATH)
import src.util as util

if __name__ == "__main__":
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

    fig_names = [
        "original",
        "melgan",
        "fb-melgan",
        "mb-melgan",
        "hifigan",
        "waveglow",
        "parallel-wavegan",
    ]

    from_frame = 0
    to_frame = 50000
    wavelet = "shan0.5-15.0"
    scales = np.arange(1, 256)

    for i in range(len(audios)):
        util.plot_scalogram(
            util.get_np_signal(f"{data_base_dir}/{audios[i]}", from_frame, to_frame),
            scales,
            wavelet,
            titles[i],
            fig_names[i],
            False,
        )
