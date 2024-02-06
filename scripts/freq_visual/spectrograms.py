"""Generate scalogram plots of CWT of audios from Wavefake-Dataset.

See Frank, Schönherr (2021): WaveFake: A Data Set to Facilitate Audio Deepfake Detection [FS21]

Simple script that generates scalograms that show apperent differences between
original audio files and the corresponding audio samples that [FS21] generated with different
Deep Learning methods and architectures that already exist, e.g. MelGAN, Waveglow, Hifi-GAN.

"""

import os
import sys

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

    # nfft 1024 [0,100000], bandwith = 0.0001, center_freq = 0.87 for tikz and tex [50,-50]dB turbo
    # nfft 1023, win_length 1023 and 256 [39500, 49500], bandwith = 0.0001, center_freq = 0.87 for tikz and tex,
    # comp. with stft [30,-60]dB hot

    from_frame = 39500
    to_frame = 49500  # -1 for last

    n_fft = 1023  # Frank et al. use 256 in statistics.py
    win_length = 1023
    rect_plot = False

    print("Plotting Spectrograms of LJ008 0217.wav")
    for i in range(len(fig_names)):
        path = f"{data_base_dir}/{audios[i]}"
        spec, frames = plot_util.compute_spectogram(
            path, from_frame, to_frame, n_fft, win_length=win_length
        )
        plot_util.plot_spectrogram(
            spec,
            frames,
            from_frame,
            to_frame,
            title=titles[i],
            fig_name=fig_names[i],
            rect_plot=rect_plot,
        )

    import matplotlib.pyplot as plt

    plt.show()
