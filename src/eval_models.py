"""Evaluate models with accuracy and eer metric."""
import argparse
import os
import sys

import numpy as np
import torch
from torch.distributed import destroy_process_group
from torch.utils.data import DataLoader, DistributedSampler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.data_loader import LearnWavefakeDataset
from src.models import get_model
from src.train_classifier import Trainer, ddp_setup
from src.utils import add_default_parser_args, print_results, set_seed
from src.wavelet_math import get_transforms


def main() -> None:
    """Evaluate all models with different seeds on given gans."""
    ddp_setup()
    args = _parse_args()
    print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = (args.model_path_prefix).split("_")

    gans = args.train_gans
    c_gans = args.crosseval_gans
    seeds = args.eval_seeds
    model_name = args.model
    batch_size = args.batch_size
    flattend_size = args.flattend_size
    nclasses = args.nclasses
    num_of_scales = args.num_of_scales
    data_prefix = args.data_prefix
    features = args.features

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
                pin_memory=True,
                sampler=DistributedSampler(
                    test_data_set, shuffle=False, seed=args.seed
                ),
            )

            for seed in seeds:
                print(f"seed: {seed}")
                model_dir = f'{"_".join(model_path)}_{gan}_{str(seed)}.pt'

                set_seed(seed)

                model = get_model(
                    model_name=model_name,
                    nclasses=nclasses,
                    num_of_scales=num_of_scales,
                    flattend_size=flattend_size,
                    in_channels=2 if args.loss_less == "True" else 1,
                    channels=channels,
                )

                trainer = Trainer(
                    model=model,
                    test_data_loader=test_data_loader,
                    normalize=normalize,
                    transforms=transforms,
                    snapshot_path=model_dir,
                    args=args,
                )

                trainer.load_snapshot(model_dir)
                trainer.testing()
                test_acc, test_eer, _, _ = trainer.test_results

                res_acc.append(test_acc)
                res_eer.append(test_eer)
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

    destroy_process_group()

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
        "--eval-seeds",
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
    parser = add_default_parser_args(parser)

    return parser.parse_args()


if __name__ == "__main__":
    main()
