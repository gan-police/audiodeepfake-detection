"""Evaluate models with accuracy and eer metric."""
import argparse
from typing import Any

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data_loader import LearnWavefakeDataset
from .models import get_model
from .ptwt_continuous_transform import get_diff_wavelet
from .utils import print_results, set_seed
from .wavelet_math import get_transforms


def calculate_eer(y_true, y_score):
    """Return the equal error rate for a binary classifier output.

    Based on:
    https://github.com/scikit-learn/scikit-learn/issues/15247
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return eer


def val_test_loop(
    data_loader,
    model: torch.nn.Module,
    normalize: torch.nn.Sequential,
    transforms: torch.nn.Sequential,
    batch_size: int,
    name: str = "",
    pbar: bool = False,
) -> tuple[float, Any]:
    """Test the performance of a model on a data set by calculating the prediction accuracy and loss of the model.

    Args:
        data_loader (DataLoader): A DataLoader loading the data set on which the performance should be measured,
            e.g. a test or validation set in a data split.
        model (torch.nn.Module): The model to evaluate.
        normalize (torch.nn.Sequential): The modules to normalize the input data.
        transforms (torch.nn.Sequential): The modules to transform the input data (e.g. packets/stft+lfcc).
        batch_size (int): Batch size of training process.
        name (str): Name for tqdm bar (default: "").
        pbar (bool): Disable/Enable tqdm bar (default: False).

    Returns:
        Tuple[float, Any]: The measured accuracy and eer of the model on the data set.
    """
    ok_sum = 0
    total = 0
    bar = tqdm(
        iter(data_loader), desc="validate", total=len(data_loader), disable=not pbar
    )
    model.eval()
    ok_dict = {}
    count_dict = {}
    y_list = []
    out_list = []
    correct_labels = []
    predicted_labels: list = []
    for val_batch in bar:
        with torch.no_grad():
            freq_time_dt = transforms(val_batch["audio"].cuda())
            freq_time_dt_norm = normalize(freq_time_dt)

            out = model(freq_time_dt_norm)
            out_max = torch.argmax(out, -1)
            ok_mask = out_max == (val_batch["label"] != 0).cuda()
            correct_labels.extend((val_batch["label"].cuda()).type(torch.long))
            predicted_labels.extend(out_max.cpu())
            ok_sum += sum(ok_mask).cpu().numpy().astype(int)
            total += batch_size
            bar.set_description(f"{name} - acc: {ok_sum/total:2.8f}")
            for lbl, okl in zip(val_batch["label"], ok_mask):
                if lbl.item() not in ok_dict:
                    ok_dict[lbl.item()] = [okl]
                else:
                    ok_dict[lbl.item()].append(okl)
                if lbl.item() not in count_dict:
                    count_dict[lbl.item()] = 1
                else:
                    count_dict[lbl.item()] += 1

            y_list.append((val_batch["label"] != 0))
            out_list.append(out_max.cpu())

    common_keys = ok_dict.keys() & count_dict.keys()
    print(
        f"{name} - ",
        [(key, (sum(ok_dict[key]) / count_dict[key]).item()) for key in common_keys],
    )

    ys = torch.cat(y_list).numpy()
    outs = torch.cat(out_list).numpy()
    eer = calculate_eer(ys, outs)
    print(f"{name} - eer: {eer:2.8f}, Val acc: {ok_sum/total:2.8f}")
    return ok_sum / total, eer


def main() -> None:
    """Evaluate all models with different seeds on given gans."""
    args = _parse_args()
    print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = (args.model_path_prefix).split("_")

    wavelet = get_diff_wavelet(args.wavelet)
    num_workers = args.num_workers
    gans = args.train_gans
    c_gans = args.crosseval_gans
    seeds = args.seeds
    sample_rate = args.sample_rate
    model_name = args.model
    batch_size = args.batch_size
    flattend_size = args.flattend_size
    nclasses = args.nclasses
    f_min = args.f_min
    f_max = args.f_max
    num_of_scales = args.num_of_scales
    data_prefix = args.data_prefix
    features = args.features
    hop_length = args.hop_length

    gan_acc_dict = {}
    mean_eers = {}
    mean_accs = {}
    allres_eer_seeds = {}
    allres_acc_seeds = {}

    if "doubledelta" in features:
        channels = 60
    elif "delta" in features:
        channels = 40
    elif "lfcc" in features:
        channels = 20
    else:
        channels = int(args.num_of_scales)

    for gan in gans:
        aeer = []
        accs = []
        aeer_seeds = []
        aacc_seeds = []

        transforms, normalize = get_transforms(
            args,
            f"{data_prefix}_{gan}",
            features,
            device,
            wavelet,
            normalization=args.calc_normalization,
            pbar=args.pbar,
        )
        for c_gan in c_gans:
            print(f"Evaluating {gan} on {c_gan}...", flush=True)
            res_acc = []
            res_eer = []

            test_data_dir = f"{data_prefix}_{c_gan}"

            set_name = "_test"
            test_data_set = LearnWavefakeDataset(test_data_dir + set_name)

            test_data_loader = DataLoader(
                test_data_set,
                batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

            for seed in seeds:
                print(f"seed: {seed}")
                model_dir = f'{"_".join(model_path)}_{gan}_{str(seed)}.pt'

                set_seed(seed)

                model = get_model(
                    wavelet=wavelet,
                    model_name=model_name,
                    nclasses=nclasses,
                    batch_size=batch_size,
                    f_min=f_min,
                    f_max=f_max,
                    sample_rate=sample_rate,
                    num_of_scales=num_of_scales,
                    flattend_size=flattend_size,
                    stft=args.transform == "stft",
                    features=features,
                    hop_length=hop_length,
                    in_channels=2 if args.loss_less == "True" else 1,
                    channels=channels,
                )
                old_state_dict = torch.load(model_dir)
                model.load_state_dict(old_state_dict)

                model.to(device)

                acc, eer = val_test_loop(
                    test_data_loader,
                    model,
                    transforms=transforms,
                    normalize=normalize,
                    batch_size=batch_size,
                    name=c_gan,
                    pbar=args.pbar,
                )

                res_acc.append(acc)
                res_eer.append(eer)
            res_dict = {}
            res_dict["max_acc"] = np.max(res_acc)
            res_dict["mean_acc"] = np.mean(res_acc)
            res_dict["std_acc"] = np.std(res_acc)
            res_dict["min_eer"] = np.min(res_eer)
            res_dict["mean_eer"] = np.mean(res_eer)
            res_dict["std_eer"] = np.std(res_eer)
            gan_acc_dict[f"{gan}-{c_gan}"] = res_dict
            aeer.append(res_dict["mean_eer"])
            accs.append(res_dict["mean_acc"])
            aeer_seeds.append(res_eer)
            aacc_seeds.append(res_acc)
        allres_eer_seeds[gan] = np.asarray(aeer_seeds)
        allres_acc_seeds[gan] = np.asarray(aacc_seeds)
        if allres_acc_seeds[gan].shape[0] > 1:
            accs_arr = allres_acc_seeds[gan].mean(0)
            aeer_arr = allres_eer_seeds[gan].mean(0)
        else:
            accs_arr = allres_acc_seeds[gan]
            aeer_arr = allres_eer_seeds[gan]
        mean_eers[gan] = (aeer_arr.mean(), aeer_arr.mean(), aeer_arr.min())
        mean_accs[gan] = (accs_arr.mean(), accs_arr.mean(), accs_arr.max())
    gan_acc_dict["aEER"] = mean_eers
    gan_acc_dict["aACC"] = mean_accs
    gan_acc_dict["allres_eer"] = allres_eer_seeds
    gan_acc_dict["allres_acc"] = allres_acc_seeds

    print(gan_acc_dict)

    for gan in gans:
        for c_gan in c_gans:
            ind = f"{gan}-{c_gan}"
            pr_str = f"{ind}: acc: max {100 * gan_acc_dict[ind]['max_acc']:.3f} %"
            pr_str += f", mean {100 * gan_acc_dict[ind]['mean_acc']:.3f} +- "
            pr_str += f"{100 * gan_acc_dict[ind]['std_acc']:.3f} %"
            print(pr_str)
            pr_str = f"{ind}: eer: min {gan_acc_dict[ind]['min_eer']:.5f},"
            pr_str += f" mean {gan_acc_dict[ind]['mean_eer']:.5f} +- "
            pr_str += f"{gan_acc_dict[ind]['std_eer']:.5f}"
            print(pr_str)
        print_results(gan_acc_dict["allres_eer"][gan], gan_acc_dict["allres_acc"][gan])


def _parse_args():
    """Parse cmd line args for evaluating audio classification models."""
    parser = argparse.ArgumentParser(description="Eval models.")
    parser.add_argument(
        "--model-path-prefix",
        type=str,
        help="Prefix of model path (without '_seed.pt').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="input batch size for testing (default: 128)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=11025,
        help="window size of samples in dataset (default: 11025)",
    )
    parser.add_argument(
        "--num-of-scales",
        type=int,
        default=150,
        help="number of scales used in training set. (default: 150)",
    )
    parser.add_argument(
        "--wavelet",
        type=str,
        default="cmor3.3-4.17",
        help="Wavelet to use in cwt. (default: cmor3.3-4.17)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Sample rate of audio. (default: 22050)",
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
        default=9500,
        help="Maximum frequency to analyze in Hz. (default: 9500)",
    )
    parser.add_argument(
        "--data-prefix",
        type=str,
        help="shared prefix of the data paths",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default="./plots/eval/",
        help="path for plotting roc and auc",
    )
    parser.add_argument(
        "--nclasses", type=int, default=2, help="number of classes (default: 2)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="the random seeds that are evaluated.",
    )
    parser.add_argument(
        "--train-gans",
        type=str,
        nargs="+",
        default=[
            "melgan",
            "lmelgan",
            "mbmelgan",
            "fbmelgan",
            "hifigan",
            "waveglow",
            "pwg",
            "bigvgan",
            "bigvganl",
            "avocodo",
            "all",
        ],
        help="model postfix specifying the gan in the trainingset.",
    )
    parser.add_argument(
        "--crosseval-gans",
        type=str,
        nargs="+",
        default=[
            "melgan",
            "lmelgan",
            "mbmelgan",
            "fbmelgan",
            "hifigan",
            "waveglow",
            "pwg",
            "bigvgan",
            "bigvganl",
            "avocodo",
            "all",
        ],
        help="model postfix specifying the gan in the trainingset for cross validation.",
    )
    parser.add_argument(
        "--flattend-size",
        type=int,
        default=21888,
        help="dense layer input size (default: 21888)",
    )
    parser.add_argument(
        "--model",
        choices=[
            "onednet",
            "learndeepnet",
            "learnnet",
            "lcnn",
        ],
        default="learndeepnet",
        help="The model type. Default: learndeepnet.",
    )
    parser.add_argument(
        "--adapt-wavelet",
        action="store_true",
        help="If differentiable wavelets shall be used.",
    )
    parser.add_argument(
        "--pbar",
        action="store_true",
        help="enables progress bars",
    )
    parser.add_argument(
        "--calc-normalization",
        action="store_true",
        help="calculate normalization for debugging purposes.",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="If differentiable wavelets shall be used.",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=2.0,
        help="Calculate power spectrum of given factor (for stft and packets) (default: 2.0).",
    )
    parser.add_argument(
        "--loss-less",
        choices=[
            "True",
            "False",
        ],
        default="False",
        help="Ff sign pattern is to be used as second channel.",
    )
    parser.add_argument(
        "--mean",
        type=float,
        default=0,
        help="Pre calculated mean. (default: 0)",
    )
    parser.add_argument(
        "--std",
        type=float,
        default=1,
        help="Pre calculated std. (default: 1)",
    )
    parser.add_argument(
        "--transform",
        choices=[
            "stft",
            "cwt",
            "packets",
        ],
        default="stft",
        help="Use different transformations.",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=1,
        help="Hop length in transformation. (default: 1)",
    )
    parser.add_argument(
        "--features",
        choices=["lfcc", "delta", "doubledelta", "all", "none"],
        default="none",
        help="Use features like lfcc, its first and second derivatives or all of them. \
              Delta and Dooubledelta include lfcc computing. Default: none.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of worker processes started by the test and validation data loaders (default: 2)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
