"""Generate scalogram plots of CWT of audios from Wavefake-Dataset, see Frank, Schönherr (2021):
WaveFake: A Data Set to Facilitate Audio Deepfake Detection [FS21]

Simple script that generates scalograms that show apperent differences between
original audio files and the corresponding audio samples that [FS21] generated with different
Deep Learning methods and architectures that already exist, e.g. MelGAN, Waveglow, Hifi-GAN.

"""
import librosa
import torch
import numpy as np
import ptwt
import matplotlib.pyplot as plt
import spectograms as sp

import tikzplotlib as tikz
from pathlib import Path


def plot_scalogram(signal,
                   mother_wavelet: str = "shan0.5-15.0",
                   widths: np.array = np.arange(1, 128),
                   title: str = "Skalogramm",
                   fig_name: str = "sample",
                   rect_plot: bool = True) -> None:
    sig, t = signal

    sampling_period = 1. / sp.SAMPLE_RATE
    cwtmatr_pt, freqs = ptwt.cwt(
        torch.from_numpy(sig), widths, mother_wavelet, sampling_period=sampling_period
    )
    cwtmatr = cwtmatr_pt.numpy()

    vmin = -80
    vmax = 0
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title)
    fig.set_dpi(200)

    # freqs contains very weird frequencies that are way too high
    im = axs.imshow(
        librosa.power_to_db(np.abs(cwtmatr) ** 2),
        cmap="plasma",
        aspect="auto",
        extent=[t[0], t[-1], widths[-1], widths[0]],
        vmin=vmin,
        vmax=vmax
    )

    axs.set_xlabel("Zeit (sek)")
    axs.set_ylabel("Skale")
    fig.colorbar(im, ax=axs, label="dB")


    print(f"saving {fig_name}-scalogram.tex")
    Path("standalone_plots/cwt").mkdir(parents=True, exist_ok=True)
    tikz.clean_figure()

    # for rectangular plots
    if rect_plot:
        fig_width, fig_height = "5in", "2.5in"
        fig.set_size_inches(10, 4, forward=True)
    else:
        fig_width, fig_height = None, None

    # tikz.save(f"standalone_plots/cwt/{fig_name}-scalogram.tex", encoding="utf-8",
    #          standalone=True, axis_width=fig_width, axis_height=fig_height)
    # plt.savefig(f"standalone_plots/cwt/{fig_name}-scalogram.png")

    plt.show()


if __name__ == "__main__":
    wav_label = "LJ008-0217"
    audios = [
        f"../data/LJspeech-1.1/wavs/{wav_label}.wav",
        f"../data/generated_audio/ljspeech_melgan/{wav_label}_gen.wav",
        f"../data/generated_audio/ljspeech_full_band_melgan/{wav_label}_gen.wav",
        f"../data/generated_audio/ljspeech_multi_band_melgan/{wav_label}_gen.wav",
        f"../data/generated_audio/ljspeech_hifiGAN/{wav_label}_generated.wav",
        f"../data/generated_audio/ljspeech_waveglow/{wav_label}.wav",
        f"../data/generated_audio/ljspeech_parallel_wavegan/{wav_label}_gen.wav",
    ]

    titles = [
        "Original",
        "MelGAN",
        "Full-Band-MelGAN",
        "Multi-Band-MelGAN",
        "Hifi-GAN",
        "Waveglow",
        "Parallel WaveGAN"
    ]

    fig_names = [
        "original",
        "melgan",
        "fb-melgan",
        "mb-melgan",
        "hifigan",
        "waveglow",
        "parallel-wavegan"
    ]

    from_frame = 0
    to_frame = 50000
    wavelet = "shan0.5-15.0"
    scales = np.arange(1, 256)

    for i in range(len(audios)):
        plot_scalogram(sp.get_np_signal(audios[i], from_frame, to_frame),
                       wavelet, scales, titles[i], fig_names[i], False)
