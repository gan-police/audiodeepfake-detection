"""Set utility functions."""

from __future__ import annotations  # Required for forward references

import itertools
import os
import random
from argparse import ArgumentParser
from typing import Any

import numpy as np
import torch
import torchaudio

from .data_loader import get_costum_dataset


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
        "--epochs",
        type=int,
        default=10,
        help="number of epochs (default: 10)",
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
        "--block-norm",
        action="store_true",
        help="Normalize frequency bin with maximum value in packet transformation.",
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
        "--init-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="The random seed pytorch and numpy works with (default: 0, 1, 2, 3, 4).",
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
            "lcnn",
            "gridmodel",
            "modules",
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
        "--only-testing",
        type=bool,
        default=False,
        help="If you only want to test the given model. (default: False)",
    )
    parser.add_argument(
        "--ckpt-every",
        type=int,
        default=1,
        help="Save model after a fixed number of epochs. (default: 1)",
    )
    parser.add_argument(
        "--time-dim-add",
        type=int,
        default=0,
        help="Add to input dim in dil conv layer. (default: 0)",
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Use distributed data parallel from pytorch.",
    )
    parser.add_argument(
        "--only-ig",
        action="store_true",
        help="Use distributed data parallel from pytorch.",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the grid search config file. IMPORTANT: This will execute code "
        "inside the given file. Make sure to only use config files, you trust entirely. "
        "This poses a security threat and will be removed in future releases.",
    )

    return parser


# type: ignore[attr-defined]
class DotDict(dict):
    """Dot.notation access to dictionary attributes."""

    __getattr__ = dict.get  # type: ignore
    __delattr__ = dict.__delitem__  # type: ignore
    __setattr__ = dict.__setitem__  # type: ignore

    only_use: Any
    save_path: Any
    limit_train: Any
    asvspoof_name: Any
    file_type: Any
    sample_rate: Any
    seconds: Any
    batch_size: Any
    transform: Any
    num_of_scales: Any
    hop_length: Any
    features: Any
    log_scale: Any
    power: Any
    wavelet: Any
    loss_less: Any
    f_min: Any
    f_max: Any
    num_workers: Any
    block_norm: Any
    mean: Any
    std: Any
    input_dim: Any
    ochannels1: Any
    kernel1: Any
    ochannels2: Any
    ochannels3: Any
    ochannels4: Any
    ochannels5: Any
    dropout_cnn: Any
    time_dim_add: Any
    dropout_lstm: Any
    flattend_size: Any
    ddp: Any
    unknown_prefix: Any
    cross_data_path: Any
    only_test_folders: Any
    cross_sources: Any
    cross_limit: Any
    asvspoof_name_cross: Any
    ckpt_every: Any
    validation_interval: Any
    enable_gs: Any
    test: Any
    seed: Any
    data_path: Any
    module: Any
    log_dir: Any
    target: Any
    ig_times_per_target: Any
    pbar: Any
    tensorboard: Any
    config: Any
    init_seeds: Any
    aug_contrast: Any
    aug_noise: Any
    random_seeds: Any
    model: Any
    data_prefix: Any
    calc_normalization: Any
    nclasses: Any
    learning_rate: Any
    weight_decay: Any
    epochs: Any
    window_size: Any
    only_testing: Any
    only_ig: Any
    get_details: Any


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
        self,
        config: dict[str, list[Any]],
        init_seeds: list | None = None,
        num_exp: int = 5,
    ) -> None:
        """Initialize grid search instance.

        Args:
            config (dict[str, list[Any]]): The config for grid search.
            init_seeds (list | None): List of seeds if fixed seeds should be used. Defaults to None.
            num_exp (int): Number of random seeds (and therefore experiments) if init_seeds is None,
                             this is used. Defaults to 5.

        Raises:
            TypeError: If config is not a dictionary.
        """
        if type(config) is not dict:
            raise TypeError(f"Config file must be of type dict but is {type(config)}.")

        self.init_config: dict[str, Any] = {}
        if init_seeds is None:
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

    def update_args(self, args: DotDict) -> DotDict:
        """Update args with current step values."""
        for value, key in zip(self.grid_values[self.current], self.get_keys()):
            # if args.get(key) is None:
            #    print(f"Added new config key: {key}.")
            args[key] = value
        return args

    def update_step(self, args):
        """Update given config variable with values from the current grid step and make one."""
        new_args = self.update_args(args)
        try:
            new_step = self.__next__()
        except StopIteration:
            return new_args, StopIteration
        return new_args, new_step


def build_new_grid(
    config: dict, random_seeds: bool = False, seeds: list | None = None
) -> _Griderator:
    """Build a new iterable grid object using given seeds.

    Args:
        config (dict): The gridsearch config dictionary.
        random_seeds (bool): True if random seeds should be used.
        seeds (list): List of predefined seeds to use. Defaults to None.

    Returns:
        _Griderator: Iterable grid search object.
    """
    if random_seeds:
        return _Griderator(config, num_exp=3)

    init_seeds = [0, 1, 2, 3, 4]
    if isinstance(seeds, list):
        init_seeds = seeds
        for i in range(len(init_seeds)):
            init_seeds[i] = int(init_seeds[i])
    return _Griderator(config, init_seeds=init_seeds)


def get_input_dims(args: DotDict, transforms) -> list:
    """Return dimensions of transformed audio."""
    dataset = get_costum_dataset(
        data_path=args.data_path,
        ds_type="train",
        only_use=args.only_use,
        save_path=args.save_path,
        limit=args.limit_train[0],
        asvspoof_name=(
            f"{args.asvspoof_name}_T"
            if args.asvspoof_name is not None and "LA" in args.asvspoof_name
            else args.asvspoof_name
        ),
        file_type=args.file_type,
        resample_rate=args.sample_rate,
        seconds=args.seconds,
    )
    with torch.no_grad():
        if torch.cuda.is_available():
            freq_time_dt, _ = transforms(
                dataset.__getitem__(0)["audio"].cuda(non_blocking=True)
            )
        else:
            freq_time_dt, _ = transforms(dataset.__getitem__(0)["audio"])

    shape = list(freq_time_dt.shape)

    if len(shape) < 4:
        shape.insert(0, args.batch_size)
    else:
        shape[0] = args.batch_size

    return shape
