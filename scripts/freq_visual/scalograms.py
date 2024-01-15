"""Generate scalogram plots of CWT of audios from Wavefake-Dataset.

See Frank, Schönherr (2021): WaveFake: A Data Set to Facilitate Audio Deepfake Detection [FS21]

Simple script that generates scalograms that show apperent differences between
original audio files and the corresponding audio samples that [FS21] generated with different
Deep Learning methods and architectures that already exist, e.g. MelGAN, Waveglow, Hifi-GAN.

"""
import os
import sys

import numpy as np
import pywt

DEBUG = True
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if DEBUG:
    # Set python path automatically to base directory
    sys.path.append(BASE_PATH)
import audiofakedetect.plot_util as plot_util

if __name__ == "__main__":
    wav_label = "LJ008-0217"
    data_base_dir = "./tests/data"
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

    # 512, [0,-1], symlog oder ohne (eher ohne), bandwith = 0.0001, center_freq = 0.87, shan for viewing
    # 1024, [0,100000], symlog oder ohne (eher ohne), bandwith = 0.0001, center_freq = 0.87, shan for saving and tex
    # 512, [0,100000], asinh, bandwith = 1.0, center_freq = 0.87, cmor [10,-80]
    # 512, [0,100000], asinh, bandwith = 0.001, center_freq = 0.87, cmor [25,-60], 1-2500 Hz, hot

    # 512 [39500, 49500], bandwith = 0.0001, center_freq = 0.87 for tikz and tex, comp. with stft [-20,-80]dB turbo

    from_frame = 39500
    to_frame = 49500  # -1 for last

    center_freq = 0.87
    bandwith = 0.0001
    wavelet = f"shan{bandwith}-{center_freq}"

    # The highest frequency that will not be aliased is equal to half the sampling frequency, f/2
    nyquist_freq = plot_util.SAMPLE_RATE / 2.0  # maximum frequency that can be analyzed
    resolution = 512
    freqs = (
        np.linspace(nyquist_freq, 1, resolution) / plot_util.SAMPLE_RATE
    )  # equally spaced normalized frequencies to be analyzed

    scales = pywt.frequency2scale(
        wavelet, freqs
    )  # generate corresponding scales to the freuqencies
    # also helpful, because then y axis ist linear

    print("Plotting Scalogram of LJ008 0217.wav")
    for i in range(len(fig_names)):
        path = f"{data_base_dir}/{audios[i]}"
        scal = plot_util.compute_cwt(path, wavelet, scales, from_frame, to_frame)
        plot_util.plot_scalogram(
            scal,
            from_frame,
            to_frame,
            titles[i],
            fig_names[i],
            False,
        )
    import matplotlib.pyplot as plt

    plt.show()
