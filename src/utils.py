"""Set utility functions."""
import os

import numpy as np
import torch
import torchaudio


def set_seed(seed: int):
    """Fix PRNG seed for reproducable experiments."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def contrast(waveform: torch.Tensor) -> torch.Tensor:
    """Add contrast to waveform."""
    enhancement_amount = np.random.uniform(0, 100.0)
    return torchaudio.functional.contrast(waveform, enhancement_amount)


def add_noise(waveform: torch.Tensor) -> torch.Tensor:
    """Add random uniform noise to waveform."""
    noise = torch.randn(waveform.shape).to(waveform.device)
    noise_snr = np.random.uniform(30, 40)
    snr = noise_snr * torch.ones(waveform.shape[:-1]).to(waveform.device)
    return torchaudio.functional.add_noise(waveform, noise, snr)


def print_results(res_eer, res_acc):
    """Print results from evaluation for tables in paper."""
    str_wf = ""
    str_avbig = ""
    str_all = ""
    wavefake = np.stack(
        [
            res_acc[0],
            res_acc[1],
            res_acc[2],
            res_acc[3],
            res_acc[4],
            res_acc[5],
            res_acc[9],
            res_acc[10],
            res_eer[11],
        ]
    ).mean(0)
    str_all += f"&${round(res_acc.mean(0).max()*100, 2)}$ & "
    str_all += rf"${round(res_acc.mean()*100, 2)} \pm {round(res_acc.mean(0).std()*100, 2)}$ &"
    str_wf += f"&${round(wavefake.max()*100, 2)}$ & "
    str_wf += (
        rf"${round(wavefake.mean()*100, 2)} \pm {round(wavefake.std()*100, 2)}$ &"
    )
    wavefake = np.stack(
        [
            res_eer[0],
            res_eer[1],
            res_eer[2],
            res_eer[3],
            res_eer[4],
            res_eer[5],
            res_eer[9],
            res_eer[10],
            res_eer[11],
        ]
    )
    str_all += f"${round(res_eer.mean(0).min(), 3)}$ & "
    str_all += rf"${round(res_eer.mean(), 3)} \pm {round(res_eer.mean(0).std(), 3)}$ "
    str_wf += f"${round(wavefake.mean(0).min(), 3)}$ & "
    str_wf += rf"${round(wavefake.mean(), 3)} \pm {round(wavefake.mean(0).std(), 3)}$ "

    avocodo_acc = res_acc[8]
    bigvgan_acc = np.stack([res_acc[6], res_acc[7]]).mean(0)

    avocodo_eer = res_eer[8]
    bigvgan_eer = np.stack([res_eer[6], res_eer[7]]).mean(0)

    str_avbig += f"&${round(avocodo_acc.max()*100, 2)}$ & "
    str_avbig += (
        rf"${round(avocodo_acc.mean()*100, 2)} \pm {round(avocodo_acc.std()*100, 2)}$ &"
    )
    str_avbig += f"${round(avocodo_eer.min(), 3)}$ & "
    str_avbig += rf"${round(avocodo_eer.mean(), 3)} \pm {round(avocodo_eer.std(), 3)}$ "
    str_avbig += "& "
    str_avbig += f"${round(bigvgan_acc.max()*100, 2)}$ & "
    str_avbig += (
        rf"${round(bigvgan_acc.mean()*100, 2)} \pm {round(bigvgan_acc.std()*100, 2)}$ &"
    )
    str_avbig += f"${round(bigvgan_eer.min(), 3)}$ & "
    str_avbig += rf"${round(bigvgan_eer.mean(), 3)} \pm {round(bigvgan_eer.std(), 3)}$ "

    print("all")
    print(str_all)
    print("wavefake")
    print(str_wf)

    print("avbigvgan")
    print(str_avbig)
