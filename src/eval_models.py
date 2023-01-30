"""Evaluate models with accuracy and eer metric."""
import datetime
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy, MulticlassROC

from .learn_direct_train_classifier import (
    create_data_loaders,
    create_data_loaders_learn,
    get_model,
)
from .ptwt_continuous_transform import get_diff_wavelet


def plot_roc(
    fpr: np.ndarray,
    tpr: np.ndarray,
    training_dataset_name: str,
    fake_dataset_name: str,
    path: str,
    lw: int = 2,
):
    """Plot roc of given fpr and tpr."""
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
    fig.savefig(f"{path}/{training_dataset_name}.pdf")
    plt.close(fig)


def classify_dataset(
    data_loader,
    model: torch.nn.Module,
    loss_fun,
    make_binary_labels,
):
    """Test the performance of a model on a data set by calculating the prediction accuracy and loss of the model.

    Args:
        data_loader (DataLoader): A DataLoader loading the data set on which the performance should be measured,
            e.g. a test or validation set in a data split.
        model (torch.nn.Module): The model to evaluate.
        loss_fun: The loss function, which is used to measure the loss of the model on the data set
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

    return acc, fpr[0], tpr[0], eer, eer_threshold


def main() -> None:
    """Evaluate all models with different seeds."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.multiprocessing.set_start_method("spawn")

    plot_path = "/home/s6kogase/code/plots/cwt/eval/"
    num_workers = 0
    gans = [
        "melgan",
        "lmelgan",
        "mbmelgan",
        "fbmelgan",
        "hifigan",
        "waveglow",
        "pwg",
        "all",
    ]
    seeds = [0, 1, 2, 3, 4]
    wavelets = ["cmor3.3-4.17", "cmor4.6-0.87", "shan0.01-0.4"]
    cu_wv = wavelets[0]
    gan_acc_dict = {}
    for gan in gans:
        print(f"Evaluating {gan}...", flush=True)
        res_acc = []
        res_eer = []
        res_eer_thresh = []
        for seed in seeds:
            print(f"seed: {seed}")
            torch.manual_seed(seed)

            model_path = [f"/home/s6kogase/code/log/fake_{cu_wv}_22050_11025_"]
            model_path[
                0
            ] += "150_1000-9500_0.7_{gan}_0.0001_128_2_10e_learndeepnet_False_{seed}.pt"
            data_args = model_path[0].split("/")[-1].split(".pt")[0].split("_")
            model_name = data_args[-3]
            nclasses = int(data_args[-5])
            batch_size = int(data_args[-6])
            wavelet = get_diff_wavelet(data_args[1])
            f_min = float(data_args[5].split("-")[0])
            f_max = float(data_args[5].split("-")[1])
            sample_rate = int(data_args[2])
            num_of_scales = int(data_args[4])
            loss_fun = torch.nn.CrossEntropyLoss()

            data_dir = "/home/s6kogase/data"
            test_data_dir = [
                f"/home/s6kogase/data/fake_cmor4.6-0.87_22050_8000_11025_224_80-4000_1_0.7_{gan}"
            ]
            """test_data_dir = [
                "/home/s6kogase/data/fake_cmor4.6-0.87_22050_8000_11025_224_80-4000_1_0.7_melgan"
            ]"""

            if test_data_dir is None:
                test_data_dir = [data_dir + "/" + "_".join(data_args[:10])]

            model = get_model(
                wavelet=wavelet,
                model_name=model_name,
                nclasses=nclasses,
                batch_size=batch_size,
                f_min=f_min,
                f_max=f_max,
                sample_rate=sample_rate,
                num_of_scales=num_of_scales,
            )
            old_state_dict = torch.load(model_path[0])
            model.load_state_dict(old_state_dict)
            # model = initialize_model(model, model_path)

            model.to(device)
            if model_name == "learndeepnet":
                _, _, test_data_set = create_data_loaders_learn(
                    test_data_dir,
                    batch_size,
                    False,
                    num_workers,
                )
            else:
                _, _, test_data_set = create_data_loaders(
                    test_data_dir,
                    batch_size,
                    False,
                    num_workers,
                    wavelet,
                    sample_rate,
                    num_of_scales,
                    f_min,
                    f_max,
                )

            test_data_loader = DataLoader(
                test_data_set,
                batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

            acc, fpr, tpr, eer, eer_threshold = classify_dataset(
                test_data_loader,
                model,
                loss_fun,
                make_binary_labels=True,
            )

            # plotting
            Path(plot_path).mkdir(parents=True, exist_ok=True)
            plot_roc(fpr, tpr, data_args[7], data_args[7], plot_path)

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
        gan_acc_dict[gan] = res_dict

    print(gan_acc_dict)
    time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    pickle.dump(
        gan_acc_dict,
        open(
            f"/home/s6kogase/code/log/results/results_all_{data_args[1]}_{model_name}_{time_now}.pkl",
            "wb",
        ),
    )

    print(wavelet)
    for dataset in gans:
        pr_str = f"{dataset}: acc: max {100 * gan_acc_dict[dataset]['max_acc']:.3f} %"
        pr_str += f", mean {100 * gan_acc_dict[dataset]['mean_acc']:.3f} +- "
        pr_str += f"{100 * gan_acc_dict[dataset]['std_acc']:.3f} %"
        print(pr_str)
        pr_str = f"{dataset}: eer: min {gan_acc_dict[dataset]['min_eer'][0]:.5f},"
        pr_str += f" mean {gan_acc_dict[dataset]['mean_eer'][0]:.5f} +- "
        pr_str += f"{gan_acc_dict[dataset]['std_eer'][0]:.5f}"
        print(pr_str)
        pr_str = f"{dataset}: eer: min {gan_acc_dict[dataset]['min_eer'][1]:.5f},"
        pr_str += f" mean {gan_acc_dict[dataset]['mean_eer'][1]:.5f} +- "
        pr_str += f"{gan_acc_dict[dataset]['std_eer'][1]:.5f}"
        print(pr_str)


if __name__ == "__main__":
    main()
