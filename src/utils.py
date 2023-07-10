"""Set utility functions."""
import itertools
import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import torchaudio

from scripts.gridsearch_config import get_config


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


def add_default_parser_args(parser: ArgumentParser) -> ArgumentParser:
    """Set default training and evaluation wide parser arguments."""
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./exp/log",
        help="Shared prefix of the data paths (default: ./exp/log).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="input batch size for testing (default: 128)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0001,
        help="learning rate for optimizer (default: 0.0001)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="weight decay for optimizer (default: 0.01)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs (default: 10)"
    )
    parser.add_argument(
        "--transform",
        choices=[
            "stft",
            "packets",
        ],
        default="stft",
        help="Use stft instead of cwt in transformation.",
    )
    parser.add_argument(
        "--features",
        choices=["lfcc", "delta", "doubledelta", "none"],
        default="none",
        help="Use features like lfcc, first and second derivatives. \
            Delta and Dooubledelta include lfcc computing. Default: none.",
    )
    parser.add_argument(
        "--num-of-scales",
        type=int,
        default=256,
        help="number of scales used in training set. (default: 256)",
    )
    parser.add_argument(
        "--wavelet",
        type=str,
        default="sym8",
        help="Wavelet to use in wavelet transformations. (default: sym8)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Sample rate of audio. (default: 22050)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=11025,
        help="Window size of audio. (default: 11025)",
    )
    parser.add_argument(
        "--f-min",
        type=float,
        default=1000,
        help="Minimum frequency to analyze in Hz. (default: 1000)",
    )
    parser.add_argument(
        "--f-max",
        type=float,
        default=11025,
        help="Maximum frequency to analyze in Hz. (default: 11025)",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=1,
        help="Hop length in cwt and stft. (default: 100).",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Log-scale transformed audio.",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=2.0,
        help="Calculate power spectrum of given factor (for stft and packets) (default: 2.0).",
    )
    parser.add_argument(
        "--dropout-cnn",
        type=float,
        default=0.6,
        help="Dropout of cnn layer (default: 0.6).",
    )
    parser.add_argument(
        "--dropout-lstm",
        type=float,
        default=0.3,
        help="Dropout of bi-lstm layer (default: 0.3).",
    )
    parser.add_argument(
        "--loss-less",
        choices=[
            "True",
            "False",
        ],
        default="False",
        help="If sign pattern is to be used as second channel, only works for packets.",
    )
    parser.add_argument(
        "--random-seeds",
        action="store_true",
        help="Use random seeds.",
    )
    parser.add_argument(
        "--aug-contrast",
        action="store_true",
        help="Use augmentation method contrast.",
    )
    parser.add_argument(
        "--aug-noise",
        action="store_true",
        help="Use augmentation method contrast.",
    )

    parser.add_argument(
        "--calc-normalization",
        action="store_true",
        help="calculate normalization for debugging purposes.",
    )
    parser.add_argument(
        "--mean",
        type=float,
        nargs="+",
        default=[0],
        help="Pre calculated mean. (default: 0)",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs="+",
        default=[1],
        help="Pre calculated std. (default: 1)",
    )
    parser.add_argument(
        "--data-prefix",
        type=str,
        default="../data/fake",
        help="Shared prefix of the data paths (default: ../data/fake).",
    )
    parser.add_argument(
        "--unknown-prefix",
        type=str,
        help="Shared prefix of the unknown source data paths (default: none).",
    )
    parser.add_argument(
        "--cross-dir",
        type=str,
        help="Shared directory of the unknown source data paths (default: none).",
    )
    parser.add_argument(
        "--cross-prefix",
        type=str,
        help="Shared prefix of the unknown source data paths (default: none).",
    )
    parser.add_argument(
        "--cross-sources",
        type=str,
        nargs="+",
        default=[
            "avocodo",
            "bigvgan",
            "bigvganl",
            "conformer",
            "hifigan",
            "melgan",
            "lmelgan",
            "mbmelgan",
            "pwg",
            "waveglow",
            "jsutmbmelgan",
            "jsutpwg",
        ],
        help="Shared source names of the unknown source data paths.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The random seed pytorch and numpy works with (default: 0).",
    )
    parser.add_argument(
        "--flattend-size",
        type=int,
        default=9600,
        help="Dense layer input size (default: 9600).",
    )
    parser.add_argument(
        "--model",
        choices=[
            "onednet",
            "learndeepnet",
            "lcnn",
        ],
        default="lcnn",
        help="The model type (default: lcnn).",
    )
    parser.add_argument(
        "--nclasses",
        type=int,
        default=2,
        help="Number of output classes in model (default: 2).",
    )
    parser.add_argument(
        "--enable-gs",
        action="store_true",
        help="Enables a grid search with values from the config file.",
    )

    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="enables a tensorboard visualization.",
    )
    parser.add_argument(
        "--pbar",
        action="store_true",
        help="enables progress bars",
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=1,
        help="Number of training epochs after which the model is being tested "
        " on the validation data set. (default: 1)",
    )
    parser.add_argument(
        "--ckpt-every",
        type=int,
        default=1,
        help="Save model after a fixed number of epochs. (default: 1)",
    )

    return parser


def contrast(waveform: torch.Tensor) -> torch.Tensor:
    """Add contrast to waveform."""
    enhancement_amount = np.random.uniform(5.0, 20.0)
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
            res_acc[11],
        ]
    ).mean(0)
    str_all += f"&${round(res_acc.mean(0).max()*100, 2)}$ & "
    str_all += (
        rf"${round(res_acc.mean()*100, 2)} \pm {round(res_acc.mean(0).std()*100, 2)}$ &"
    )
    str_wf += f"&${round(wavefake.max()*100, 2)}$ & "
    str_wf += rf"${round(wavefake.mean()*100, 2)} \pm {round(wavefake.std()*100, 2)}$ &"
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


class _Griderator:
    """Create an iterator for grid search."""

    def __init__(
        self, config: dict[str, list], init_seeds: list = None, num_exp: int = 5
    ) -> None:
        """Initialize grid search instance.

        Raises:
            TypeError: If given config file is not of type dict.
        """
        if type(config) is not dict:
            raise TypeError(f"Config file must be of type dict but is {type(config)}.")

        if len(init_seeds) == 0:
            rand = random.SystemRandom()
            self.init_config = {"seed": [rand.randrange(10000) for _ in range(num_exp)]}
        else:
            self.init_config = {"seed": init_seeds}

        self.init_config.update(config)
        self.grid_values = list(itertools.product(*self.init_config.values()))
        self.current = 0

    def get_keys(self):
        """Get key names of grid item."""
        return self.init_config.keys()

    def get_len(self):
        """Get number of runs for this grid."""
        return len(self.grid_values)

    def __iter__(self):
        """Return the majesty herself."""
        return self

    def __next__(self):
        """Define what to do on next step call.

        Raises:
            StopIteration: If iterator came to an end.
        """
        self.current += 1
        if self.current < len(self.grid_values):
            return self.grid_values[self.current]
        raise StopIteration

    def next(self):
        """Make next step dot call possible."""
        return self.__next__()

    def reset(self):
        """Set iterator to initial value."""
        self.current = 0

    def update_args(self, args):
        """Update args with current step values."""
        for value, key in zip(self.grid_values[self.current], self.get_keys()):
            if args.get(key) is not None:
                args[key] = value
            else:
                print(f"Skipping invalid config key {key}.")

        return args

    def update_step(self, args):
        """Update given config variable with values from the current grid step and make one."""
        new_args = self.update_args(args)
        try:
            new_step = self.__next__()
        except StopIteration:
            return new_args, StopIteration
        return new_args, new_step


def init_grid(num_exp: int = 5, init_seeds: list = None) -> _Griderator:
    """Return a grid iterator using the given config.

    Args:
        num_exp (int): Number of random seeds to use for grid search.
    """
    return _Griderator(get_config(), num_exp=num_exp, init_seeds=init_seeds)


class DotDict(dict):
    """Dot.notation access to dictionary attributes."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__  # type: ignore
    __delattr__ = dict.__delitem__  # type: ignore
