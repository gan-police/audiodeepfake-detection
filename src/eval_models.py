"""Evaluate models with accuracy and eer metric."""
import argparse
import datetime
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy, MulticlassROC

from .ptwt_continuous_transform import get_diff_wavelet
from .train_classifier import create_data_loaders, get_model


def plot_roc(
    fpr: np.ndarray,
    tpr: np.ndarray,
    training_dataset_name: str,
    fake_dataset_name: str,
    path: str,
    lw: int = 2,
):
    """Plot roc of given false positive rate and true positive rate."""
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(
        fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Train: {training_dataset_name}\nEvaluated on {fake_dataset_name}")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(f"{path}/{training_dataset_name}_on_{fake_dataset_name}.pdf")
    plt.close(fig)


def classify_dataset(
    data_loader,
    model: torch.nn.Module,
    make_binary_labels,
):
    """Test the performance of a model on a data set by calculating the prediction accuracy and loss of the model.

    Args:
        data_loader (DataLoader): A DataLoader loading the data set on which the performance should be measured,
            e.g. a test or validation set in a data split.
        model (torch.nn.Module): The model to evaluate.
        make_binary_labels (bool): If flag is set, we only classify binarily, i.e. whether an audio is real or fake.
            In this case, the label 0 encodes 'real'. All other labels are cosidered fake data, and are set to 1.

    Returns:
        Tuple[float, Any]: The measured accuracy and loss of the model on the data set.
    """
    with torch.no_grad():
        model.eval()
        val_total = 0

        result_preds = []
        result_targets = []
        result_probs = []
        result_test = []
        for val_batch in iter(data_loader):
            batch_audios = val_batch[data_loader.dataset.key].cuda(non_blocking=True)
            batch_labels = val_batch["label"].cuda(non_blocking=True)
            out = model(batch_audios)
            if make_binary_labels:
                batch_labels[batch_labels > 0] = 1
            val_total += batch_labels.shape[0]

            result_preds.extend(torch.max(out, dim=-1)[1].tolist())
            result_test.extend(torch.max(out, dim=-1)[0].tolist())
            result_probs.extend(out.tolist())
            result_targets.extend(batch_labels.tolist())
            if val_total % 2000 == 0:
                print(
                    f"processed {val_total//batch_labels.shape[0]} of {data_loader.__len__()}"
                )

        metric = BinaryAccuracy()
        probs = torch.asarray(result_probs)
        probs = torch.nn.functional.softmax(probs, dim=-1)
        preds = torch.asarray(result_preds)
        targets = torch.asarray(result_targets)
        acc = metric(preds, targets)

        print(f"acc {acc.item()*100:.3f} %", flush=True)

        metric2 = MulticlassROC(num_classes=2, thresholds=None)
        fpr, tpr, thresholds = metric2(probs, targets)
        fnr = 1 - tpr[0]

        eer_threshold = thresholds[0][np.nanargmin(np.absolute((fnr - fpr[0])))]
        eer_1 = fpr[0][np.nanargmin(np.absolute((fnr - fpr[0])))]
        eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr[0])))]
        eer = (eer_1 + eer_2) / 2

        print(f"eer {eer:.5f}", flush=True)

    return acc, fpr[0], tpr[0], eer, eer_threshold


def main() -> None:
    """Evaluate all models with different seeds on given gans."""
    args = _parse_args()
    print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    plot_path = args.plot_path
    num_workers = args.num_workers
    gans = args.train_gans
    c_gans = args.crosseval_gans
    seeds = args.seeds
    wavelet = get_diff_wavelet(args.wavelet)
    sample_rate = args.sample_rate
    window_size = args.window_size
    model_name = args.model
    batch_size = args.batch_size
    flattend_size = args.flattend_size
    adapt_wav = args.adapt_wavelet
    nclasses = args.nclasses
    f_min = args.f_min
    f_max = args.f_max
    num_of_scales = args.num_of_scales
    cut = args.cut
    data_prefix = args.data_prefix

    Path(plot_path).mkdir(parents=True, exist_ok=True)

    gan_acc_dict = {}
    mean_eers = {}
    for gan in gans:
        aeer = []
        for c_gan in c_gans:
            print(f"Evaluating {gan} on {c_gan}...", flush=True)
            res_acc = []
            res_eer = []
            res_eer_thresh = []

            test_data_dir = f"{data_prefix}_{c_gan}"

            _, _, test_data_set = create_data_loaders(
                test_data_dir,
                batch_size,
                False,
                num_workers,
            )

            test_data_loader = DataLoader(
                test_data_set,
                batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            for seed in seeds:
                print(f"seed: {seed}")

                torch.manual_seed(seed)
                model_path = f"./log/fake_{args.wavelet}_{sample_rate}_{window_size}_"
                model_path += f"{num_of_scales}_{f_min}-{f_max}_0.7_{gan}_0.0001_"
                model_path += (
                    f"{batch_size}_{nclasses}_10e_{model_name}_{adapt_wav}_{seed}.pt"
                )
                print(model_path)

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
                    cut=cut,
                )
                old_state_dict = torch.load(model_path)
                model.load_state_dict(old_state_dict)

                model.to(device)

                acc, fpr, tpr, eer, eer_threshold = classify_dataset(
                    test_data_loader,
                    model,
                    make_binary_labels=True,
                )

                plot_roc(fpr, tpr, gan, c_gan, plot_path)

                res_acc.append(acc)
                res_eer.append(eer.item())
                res_eer_thresh.append(eer_threshold.item())
            res_dict = {}
            res_dict["max_acc"] = np.max(res_acc)
            res_dict["mean_acc"] = np.mean(res_acc)
            res_dict["std_acc"] = np.std(res_acc)
            res_dict["min_eer"] = (np.min(res_eer), np.min(res_eer_thresh))
            res_dict["mean_eer"] = (np.mean(res_eer), np.mean(res_eer_thresh))
            res_dict["std_eer"] = (np.std(res_eer), np.std(res_eer_thresh))
            gan_acc_dict[f"{gan}-{c_gan}"] = res_dict
            aeer.append(res_dict["mean_eer"][0])
        mean_eers[gan] = (np.mean(aeer), np.std(aeer))
    gan_acc_dict["aEER"] = mean_eers

    print(gan_acc_dict)
    time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    print(wavelet)
    for gan in gans:
        for c_gan in c_gans:
            ind = f"{gan}-{c_gan}"
            pr_str = f"{ind}: acc: max {100 * gan_acc_dict[ind]['max_acc']:.3f} %"
            pr_str += f", mean {100 * gan_acc_dict[ind]['mean_acc']:.3f} +- "
            pr_str += f"{100 * gan_acc_dict[ind]['std_acc']:.3f} %"
            print(pr_str)
            pr_str = f"{ind}: eer: min {gan_acc_dict[ind]['min_eer'][0]:.5f},"
            pr_str += f" mean {gan_acc_dict[ind]['mean_eer'][0]:.5f} +- "
            pr_str += f"{gan_acc_dict[ind]['std_eer'][0]:.5f}"
            print(pr_str)
            pr_str = f"{ind}: eer thresh: min {gan_acc_dict[ind]['min_eer'][1]:.5f},"
            pr_str += f" mean {gan_acc_dict[ind]['mean_eer'][1]:.5f} +- "
            pr_str += f"{gan_acc_dict[ind]['std_eer'][1]:.5f}"
            print(pr_str)
        print(f"average eer for {gan}: {gan_acc_dict['aEER'][gan]}")

    pickle.dump(
        gan_acc_dict,
        open(
            f"log/results/results_all_{args.wavelet}_{model_name}_{cut}_{sample_rate}_{time_now}.pkl",
            "wb",
        ),
    )


def _parse_args():
    """Parse cmd line args for evaluating audio classification models."""
    parser = argparse.ArgumentParser(description="Eval models.")
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
        "--cut",
        action="store_true",
        help="Cut sides of audios at input into cnn.",
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
