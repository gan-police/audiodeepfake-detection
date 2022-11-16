"""Ssqueezepy test script for SSQ_CWT of WafeFake Dataset Audiofiles

For generating scalograms of ssq_cwt of audio samples generated with different Deep Learning algorithms

See also: https://github.com/OverLordGoldDragon/ssqueezepy
"""

import matplotlib.pyplot as plt
from ssqueezepy import ssq_cwt, ssq_stft, cwt, stft
import src.spectograms as sp
import librosa


def cwt_plot(sig, title):
    signal, t = sig
    #coeffs, Wx, *_ = cwt(signal, wavelet="bump")

    # ssqueezepy.wavs()
    # {'hhhat', 'gmw', 'morlet', 'bump', 'cmhat'}
    coeffs, Wx, *_ = ssq_cwt(signal, wavelet="bump")
    #Tsxo, Sxo, *_ = ssq_stft(signal)

    fig, ax = plt.subplots(1)
    ax.set_title(title)
    ax.set_xlabel("Zeit (Frame)")
    ax.set_ylabel("Skale")

    vmin = -25
    vmax = -100

    power_to_db = librosa.power_to_db(abs(coeffs) ** 2)
    im = ax.imshow(power_to_db, aspect='auto', cmap='turbo', vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax)
    plt.show()


if __name__ == "__main__":

    from_frame = 0
    to_frame = 50000

    audios = [
        "../data/LJspeech-1.1/wavs/LJ008-0217.wav",
        "../data/generated_audio/ljspeech_melgan/LJ008-0217_gen.wav",
        "../data/generated_audio/ljspeech_full_band_melgan/LJ008-0217_gen.wav",
        "../data/generated_audio/ljspeech_multi_band_melgan/LJ008-0217_gen.wav",
        "../data/generated_audio/ljspeech_hifiGAN/LJ008-0217_generated.wav",
        "../data/generated_audio/ljspeech_waveglow/LJ008-0217.wav",
        "../data/generated_audio/ljspeech_parallel_wavegan/LJ008-0217_gen.wav",
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

    for i in range(len(audios)):
        cwt_plot(sp.get_np_signal(audios[i], from_frame, to_frame), titles[i])
