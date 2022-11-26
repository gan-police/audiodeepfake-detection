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
    to_frame = 100000  # -1 for last

    n_fft = 1024  # Frank et al. use 256 in statistics.py...

    rect_plot = False

    print("Plotting Spectrograms of LJ008 0217.wav")
    for i in range(len(audios)):
        path = f"{data_base_dir}/{audios[i]}"
        spec, frames = util.compute_spectogram(path, from_frame, to_frame, n_fft)
        util.plot_spectrogram(
            spec,
            frames,
            from_frame,
            to_frame,
            title=titles[i],
            fig_name=fig_names[i],
            rect_plot=rect_plot,
        )
