import numpy as np
from tqdm import tqdm
import torchaudio
import torch
from src.data_loader import LearnWavefakeDataset

        

def compress(in_signal: torch.Tensor, sample_rate: int=22050) -> torch.Tensor:
    """Compress the audiofile.

    Vorbis expects a compression number from -1 to 10; -1 is the highest compression and lowest quality. 

    Args:
        in_signal (torch.Tensor): Audio signals of shape [batch, time]
        sample_rate (int): The sampling rate.

    Returns:
        torch.Tensor: Return
    """
    # TODO: hide vorbis can't encode Vorbis to 16-bit warnings.
    # https://github.com/pytorch/audio/issues/2099
    # is not fixed.
    compint = np.random.randint(1, 5)
    param = {"format": "vorbis", "compression": compint} # -1
    # param = {"format": "mp3", "compression": -4.5}
    batch = torch.cat([torchaudio.functional.apply_codec(
        in_signal_el.unsqueeze(0), sample_rate=sample_rate, **param) for in_signal_el in in_signal], 0)
    return batch


def transpose(in_signal: torch.Tensor, sample_rate=22050):
    # shiftint = np.random.randint(-8, 8)
    shiftint = 8
    # pitch works in 100th semitones
    shift = shiftint*200
    param = [["pitch", str(shift)]]
    batch = torch.cat([torchaudio.sox_effects.apply_effects_tensor(in_signal_el.unsqueeze(0), sample_rate, param)[0]
                       for in_signal_el in in_signal], 0)
    return batch


def dc_shift(waveform: torch.Tensor) -> torch.Tensor:
    shift = np.random.uniform(-.25, .25)
    return torchaudio.functional.dcshift(waveform, shift)

def contrast(waveform: torch.Tensor) -> torch.Tensor:
    enhancement_amount = np.random.uniform(0, 100.)
    return torchaudio.functional.contrast(waveform, enhancement_amount)


def add_noise(waveform: torch.Tensor) -> torch.Tensor:
    noise = torch.randn(waveform.shape).to(waveform.device)
    noise_snr = np.random.uniform(20, 40)
    snr = noise_snr*torch.ones(waveform.shape[:-1]).to(waveform.device)
    return torchaudio.functional.add_noise(waveform, noise, snr)

if __name__ == '__main__':
    print(torchaudio.__version__)
    import matplotlib.pyplot as plt

    dataset = LearnWavefakeDataset(data_dir='/home/wolter/uni/audiofake/data/ljspeech_22050_33075_0.7_train')
    dataset = torch.utils.data.DataLoader(dataset, batch_size=10)
    mean_dict = {}

    for it, batch in enumerate(
        tqdm(
            iter(dataset),
            desc="computing wps",
            total=len(dataset)
        )
    ):
        batch_audios = batch['audio']
        # processed = compress(batch_audios.squeeze(1))
        # processed = transpose(batch_audios.squeeze(1))
        processed = add_noise(dc_shift(contrast(batch_audios.squeeze(1))))
        # processed = add_noise(batch_audios.squeeze(1))

        fig, axs = plt.subplots(2, 2)
        axs[0][0].specgram(batch['audio'][0][0], Fs=22050)
        axs[1][0].specgram(processed[0], Fs=22050)
        axs[0][1].plot(batch['audio'][0][0])
        axs[1][1].plot(processed[0])
        plt.show()

        import sounddevice as sd
        sd.play(processed[0], 22050)


        pass