"""Source code to train audio deepfake detectors in wavelet space."""

import argparse
import os
import pickle
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from pywt import ContinuousWavelet
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import models as tv_models
from tqdm import tqdm

from .data_loader import (
    CombinedDataset,
    LearnWavefakeDataset,
    WavefakeDataset,
    WelfordEstimator,
)
from .models import (
    DeepTestNet,
    LearnDeepTestNet,
    LearnNet,
    OneDNet,
    Regression,
    TestNet,
    save_model,
)
from .optimizer import AdamCWT
from .ptwt_continuous_transform import (
    _DifferentiableContinuousWavelet,
    get_diff_wavelet,
)


def val_test_loop(
    data_loader,
    model: torch.nn.Module,
    loss_fun,
    make_binary_labels: bool = False,
    pbar: bool = False,
) -> Tuple[float, Any]:
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

        val_ok = 0.0
        for val_batch in iter(data_loader):
            if type(data_loader.dataset) is CombinedDataset:
                batch_audios = {
                    key: val_batch[key].cuda(non_blocking=True)
                    for key in data_loader.dataset.key
                }
            else:
                batch_audios = val_batch[data_loader.dataset.key].cuda(
                    non_blocking=True
                )
            batch_labels = val_batch["label"].cuda(non_blocking=True)
            out = model(batch_audios)
            if make_binary_labels:
                batch_labels[batch_labels > 0] = 1
            val_loss = loss_fun(torch.squeeze(out), batch_labels)
            ok_mask = torch.eq(torch.max(out, dim=-1)[1], batch_labels)
            val_ok += torch.sum(ok_mask).item()
            val_total += batch_labels.shape[0]
        val_acc = val_ok / val_total
        print("acc", val_acc, "ok", val_ok, "total", val_total)
    return val_acc, val_loss


def _parse_args():
    """Parse cmd line args for training an audio classifier."""
    parser = argparse.ArgumentParser(description="Train an audio classifier")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="input batch size for testing (default: 128)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="learning rate for optimizer (default: 1e-3)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
        help="weight decay for optimizer (default: 0.0001)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs (default: 10)"
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=200,
        help="number of training steps after which the model is tested on the validation data set (default: 200)",
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
        default="cmor4.6-0.87",
        help="Wavelet to use in cwt. (default: cmor4.6-0.87)",
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
        help="Minimum frequency to analyze. (default: 1 kHz)",
    )
    parser.add_argument(
        "--f-max",
        type=float,
        default=9500,
        help="Maximum frequency to analyze. (default: 9.5 kHz)",
    )
    parser.add_argument(
        "--data-prefix",
        type=str,
        nargs="+",
        default=["./data/fake"],
        help="shared prefix of the data paths (default: ./data/fake)",
    )
    parser.add_argument(
        "--nclasses", type=int, default=2, help="number of classes (default: 2)"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="the random seed pytorch works with."
    )

    parser.add_argument(
        "--model",
        choices=[
            "regression",
            "testnet",
            "resnet18",
            "resnet34",
            "deeptestnet",
            "onednet",
            "learndeepnet",
            "learnnet",
        ],
        default="testnet",
        help="The model type chosse regression or CNN. Default: testnet.",
    )
    parser.add_argument(
        "--adapt-wavelet",
        action="store_true",
        help="If differentiable wavelets shall be used.",
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
        "--calc-normalization",
        action="store_true",
        help="calculate normalization for debugging purposes.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of worker processes started by the test and validation data loaders. The training data_loader "
        "uses three times this argument many workers. Hence, this argument should probably be chosen below 10. "
        "Defaults to 2.",
    )

    parser.add_argument(
        "--class-weights",
        type=float,
        metavar="CLASS_WEIGHT",
        nargs="+",
        default=None,
        help="If specified, training samples are weighted based on their class "
        "in the loss calculation. Expects one weight per class.",
    )
    return parser.parse_args()


def get_model(
    wavelet: ContinuousWavelet,
    model_name: str,
    nclasses: int = 2,
    batch_size: int = 128,
    f_min: float = 1000,
    f_max: float = 9500,
    sample_rate: int = 22050,
    num_of_scales: int = 150,
    flattend_size: int = 21888,
    raw_input: Optional[bool] = True,
) -> Regression | TestNet | DeepTestNet | LearnDeepTestNet | OneDNet | tv_models.ResNet:
    """Get torch module model with given parameters."""
    if model_name == "regression":
        model = Regression(nclasses, 224 * 224)  # type: ignore
    elif model_name == "testnet":
        model = TestNet(classes=nclasses, batch_size=batch_size)  # type: ignore
    elif model_name == "deeptestnet":
        model = DeepTestNet(classes=nclasses, batch_size=batch_size)  # type: ignore
    elif model_name == "learndeepnet":
        model = LearnDeepTestNet(
            classes=nclasses,
            wavelet=wavelet,
            f_min=f_min,
            f_max=f_max,
            sample_rate=sample_rate,
            num_of_scales=num_of_scales,
            batch_size=batch_size,
            raw_input=raw_input,
            flattend_size=flattend_size,
        )  # type: ignore
    elif model_name == "learnnet":
        model = LearnNet(
            classes=nclasses,
            wavelet=wavelet,
            f_min=f_min,
            f_max=f_max,
            sample_rate=sample_rate,
            num_of_scales=num_of_scales,
            batch_size=batch_size,
            raw_input=raw_input,
            flattend_size=flattend_size,
        )  # type: ignore
    elif model_name == "onednet":
        model = OneDNet(
            classes=nclasses,
            wavelet=wavelet,
            f_min=f_min,
            f_max=f_max,
            sample_rate=sample_rate,
            num_of_scales=num_of_scales,
            batch_size=batch_size,
            flattend_size=flattend_size,
        )  # type: ignore
    elif model_name == "resnet18":
        model = tv_models.resnet18(weights="IMAGENET1K_V1")  # type: ignore
        model.fc = torch.nn.Linear(in_features=512, out_features=nclasses, bias=True)
        model.get_name = lambda: "ResNet18"  # type: ignore
    else:
        model = tv_models.resnet34(weights="IMAGENET1K_V1")  # type: ignore
        model.fc = torch.nn.Linear(in_features=512, out_features=nclasses, bias=True)
        model.get_name = lambda: "ResNet34"  # type: ignore

    return model


def create_data_loaders(
    data_prefix: List[str],
    batch_size: int,
    normal: bool,
    num_workers: int,
    wavelet: _DifferentiableContinuousWavelet,
    sample_rate: int,
    num_of_scales: int,
    f_min: float,
    f_max: float,
) -> tuple:
    """Create the data loaders needed for training.

    The test set is created outside a loader.

    Args:
        data_prefix (str): Where to look for the data.
        batch_size (int): preferred training batch size.
        normal (bool): True if mean and std need to be calculated again.
        num_workers (int): Number of workers for training.

    Raises:
        RuntimeError: Raised if the prefix is incorrect.

    Returns:
        dataloaders (tuple): train_data_loader, val_data_loader, test_data_set

    # noqa: DAR401
    """
    data_set_list = []
    data_prefix_el = data_prefix[0]
    key = "audio"
    if normal:
        with torch.no_grad():
            train_data_set = WavefakeDataset(
                data_prefix_el + "_train",
                key=key,
                wavelet=wavelet,
                num_of_scales=num_of_scales,
                f_max=f_max,
                f_min=f_min,
                sample_rate=sample_rate,
            )

            welford = WelfordEstimator()
            for aud_no in range(train_data_set.__len__()):
                welford.update(train_data_set.__getitem__(aud_no)["audio"])
            mean_new, std_new = welford.finalize()
        print("mean", mean_new.mean(), "std:", std_new.mean())
        with open(f"{data_prefix_el}_train/mean_std.pkl", "wb") as f:
            pickle.dump([mean_new.cpu().numpy(), std_new.cpu().numpy()], f)

    with open(f"{data_prefix_el}_train/mean_std.pkl", "rb") as file:
        mean, std = pickle.load(file)
        mean = torch.from_numpy(mean.astype(np.float32))
        std = torch.from_numpy(std.astype(np.float32))

    train_data_set = WavefakeDataset(
        data_prefix_el + "_train",
        mean=mean,
        std=std,
        key=key,
        wavelet=wavelet,
        num_of_scales=num_of_scales,
        f_max=f_max,
        f_min=f_min,
        sample_rate=sample_rate,
    )
    val_data_set = WavefakeDataset(
        data_prefix_el + "_val",
        mean=mean,
        std=std,
        key=key,
        wavelet=wavelet,
        num_of_scales=num_of_scales,
        f_max=f_max,
        f_min=f_min,
        sample_rate=sample_rate,
    )
    test_data_set = WavefakeDataset(
        data_prefix_el + "_test",
        mean=mean,
        std=std,
        key=key,
        wavelet=wavelet,
        num_of_scales=num_of_scales,
        f_max=f_max,
        f_min=f_min,
        sample_rate=sample_rate,
    )
    data_set_list.append((train_data_set, val_data_set, test_data_set))

    if len(data_set_list) == 1:
        train_data_loader = DataLoader(
            train_data_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_data_loader = DataLoader(
            val_data_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        test_data_sets: Any = test_data_set
    else:
        raise RuntimeError("Failed to load data from the specified prefixes.")

    return train_data_loader, val_data_loader, test_data_sets


def create_data_loaders_learn(
    data_prefix: List[str],
    batch_size: int,
    normal: bool,
    num_workers: int,
) -> tuple:
    """Create the data loaders needed for training.

    The test set is created outside a loader.

    Args:
        data_prefix (str): Where to look for the data.
        batch_size (int): preferred training batch size.
        normal (bool): True if mean and std need to be calculated again.
        num_workers (int): Number of workers for training.

    Returns:
        dataloaders (tuple): train_data_loader, val_data_loader, test_data_set

    # noqa: DAR401
    """
    data_set_list = []
    data_prefix_el = data_prefix[0]
    key = "audio"
    if normal:
        with torch.no_grad():
            train_data_set = LearnWavefakeDataset(
                data_prefix_el + "_train",
                key=key,
            )
            print("Computing mean and std...")
            welford = WelfordEstimator()
            ds_length = train_data_set.__len__()
            for aud_no in range(ds_length):
                if aud_no % 10000 == 0:
                    print(f"Computed {aud_no} files")
                welford.update(train_data_set.__getitem__(aud_no)["audio"])
            mean_new, std_new = welford.finalize()
        print("mean", mean_new.mean(), "std:", std_new.mean())
        with open(f"{data_prefix_el}_train/mean_std.pkl", "wb") as f:
            pickle.dump([mean_new.cpu().numpy(), std_new.cpu().numpy()], f)

    with open(f"{data_prefix_el}_train/mean_std.pkl", "rb") as file:
        mean, std = pickle.load(file)
        mean = torch.from_numpy(mean.astype(np.float32))
        std = torch.from_numpy(std.astype(np.float32))

    train_data_set = LearnWavefakeDataset(
        data_prefix_el + "_train",
        mean=mean,
        std=std,
        key=key,
    )
    val_data_set = LearnWavefakeDataset(
        data_prefix_el + "_val",
        mean=mean,
        std=std,
        key=key,
    )
    test_data_set = LearnWavefakeDataset(
        data_prefix_el + "_test",
        mean=mean,
        std=std,
        key=key,
    )
    data_set_list.append((train_data_set, val_data_set, test_data_set))

    train_data_loader = DataLoader(
        train_data_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_data_loader = DataLoader(
        val_data_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_data_sets: Any = test_data_set

    return train_data_loader, val_data_loader, test_data_sets


def main():
    """Trains a model to classify audios.

    All settings such as which model to use, parameters, normalization, data set path,
    seed etc. are specified via cmd line args.
    All training, validation and testing results are printed to stdout.
    After the training is done, the results are stored in a pickle dump in the 'log' folder.
    The state_dict of the trained model is stored there as well.
    """
    args = _parse_args()
    print(args)

    if not os.path.exists("./log/"):
        os.makedirs("./log/")

    path_name = args.data_prefix[0].split("/")[-1].split("_")
    model_file = (
        "./log/"
        + path_name[0]
        + "_"
        + str(args.wavelet)
        + "_"
        + str(args.sample_rate)
        + "_"
        + path_name[4]
        + "_"
        + str(args.num_of_scales)
        + "_"
        + str(int(args.f_min))
        + "-"
        + str(int(args.f_max))
        + "_"
        + path_name[8]
        + "_"
        + path_name[9]
        + "_"
        + str(args.learning_rate)
        + "_"
        + str(args.batch_size)
        + "_"
        + str(args.nclasses)
        + "_"
        + f"{args.epochs}e"
        + "_"
        + str(args.model)
        + "_"
        + str(args.adapt_wavelet)
        + "_"
        + str(args.seed)
    )

    # fix the seed in the interest of reproducible results.
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.multiprocessing.set_start_method("spawn")

    make_binary_labels = args.nclasses == 2

    wavelet = get_diff_wavelet(args.wavelet, args.adapt_wavelet)
    wavelet.bandwidth_par.requires_grad = args.adapt_wavelet
    wavelet.center_par.requires_grad = args.adapt_wavelet

    if (
        args.model == "learndeepnet"
        or args.model == "onednet"
        or args.model == "learnnet"
    ):
        train_data_loader, val_data_loader, test_data_set = create_data_loaders_learn(
            args.data_prefix,
            args.batch_size,
            args.calc_normalization,
            args.num_workers,
        )
    else:
        train_data_loader, val_data_loader, test_data_set = create_data_loaders(
            args.data_prefix,
            args.batch_size,
            args.calc_normalization,
            args.num_workers,
            wavelet,
            args.sample_rate,
            args.num_of_scales,
            args.f_min,
            args.f_max,
        )

    validation_list = []
    loss_list = []
    accuracy_list = []
    step_total = 0

    model = get_model(
        wavelet,
        args.model,
        args.nclasses,
        args.batch_size,
        args.f_min,
        args.f_max,
        args.sample_rate,
        args.num_of_scales,
    )

    model.to(device)

    if args.tensorboard:
        writer_str = "runs/"
        writer_str += "params_test2/"
        writer_str += f"{args.model}/"
        writer_str += f"{args.batch_size}/"
        writer_str += f"{args.wavelet}/"
        writer_str += f"{args.f_min}-"
        writer_str += f"{args.f_max}/"
        writer_str += f"{args.num_of_scales}/"
        writer_str += f"{args.adapt_wavelet}/"
        writer_str += f"{path_name[9]}/"
        writer_str += f"{args.learning_rate}_"
        writer_str += f"{args.weight_decay}_"
        writer_str += f"{args.epochs}_"
        writer_str += f"{args.seed}"
        writer = SummaryWriter(writer_str, max_queue=100)

    if args.class_weights:
        loss_fun = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(args.class_weights).cuda()
        )
    else:
        loss_fun = torch.nn.CrossEntropyLoss()

    if (
        args.model == "learndeepnet"
        or args.model == "onednet"
        or args.model == "learnnet"
    ):
        optimizer = AdamCWT(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )

    for e in tqdm(
        range(args.epochs), desc="Epochs", unit="epochs", disable=not args.pbar
    ):
        # iterate over training data.
        for it, batch in enumerate(
            tqdm(
                iter(train_data_loader),
                desc="Training",
                unit="batches",
                disable=not args.pbar,
            )
        ):
            model.train()
            optimizer.zero_grad()
            batch_audios = batch[train_data_loader.dataset.key].cuda(non_blocking=True)

            batch_labels = batch["label"].cuda(non_blocking=True)
            if make_binary_labels:
                batch_labels[batch_labels > 0] = 1

            out = model(batch_audios)
            loss = loss_fun(torch.squeeze(out), batch_labels)
            ok_mask = torch.eq(torch.max(out, dim=-1)[1], batch_labels)
            acc = torch.sum(ok_mask.type(torch.float32)) / len(batch_labels)

            if it % 10 == 0:
                print(
                    "e",
                    e,
                    "it",
                    it,
                    "total",
                    step_total,
                    "loss",
                    loss.item(),
                    "acc",
                    acc.item(),
                    "bandwidth",
                    round(
                        model.state_dict()["cwt.wavelet.bandwidth_par"].item() ** 2, 5
                    ),
                    "center freq",
                    round(model.state_dict()["cwt.wavelet.center_par"].item() ** 2, 5),
                    flush=True,
                )
                # GPUtil.showUtilization()
                torch.cuda.empty_cache()
            loss.backward()
            optimizer.step()
            step_total += 1
            loss_list.append([step_total, e, loss.item()])
            accuracy_list.append([step_total, e, acc.item()])

            if args.tensorboard:
                writer.add_scalar("loss/train", loss.item(), step_total)
                writer.add_scalar("accuracy/train", acc.item(), step_total)
                if step_total == 0:
                    writer.add_graph(model, batch_audios)

            # iterate over val batches.
            if step_total % args.validation_interval == 0:
                val_acc, val_loss = val_test_loop(
                    val_data_loader,
                    model,
                    loss_fun,
                    make_binary_labels=make_binary_labels,
                    pbar=args.pbar,
                )
                validation_list.append([step_total, e, val_acc])
                if validation_list[-1] == 1.0:
                    print("val acc ideal stopping training.", flush=True)
                    break

                if args.tensorboard:
                    writer.add_scalar("loss/validation", val_loss, step_total)
                    writer.add_scalar("accuracy/validation", val_acc, step_total)

        if args.tensorboard:
            writer.add_scalar("epochs", e, step_total)

        save_model_epoch(model_file, model)

    print(validation_list)

    model_file = save_model_epoch(model_file, model)

    # Run over the test set.
    print("Training done testing....")
    test_data_loader = DataLoader(
        test_data_set,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    with torch.no_grad():
        test_acc, test_loss = val_test_loop(
            test_data_loader,
            model,
            loss_fun,
            make_binary_labels=make_binary_labels,
            pbar=not args.pbar,
        )
        print("test acc", test_acc)

    if args.tensorboard:
        writer.add_scalar("accuracy/test", test_acc, step_total)
        writer.add_scalar("loss/test", test_loss, step_total)

    _save_stats(
        model_file,
        loss_list,
        accuracy_list,
        validation_list,
        test_acc,
        args,
        len(iter(train_data_loader)),
    )

    if args.tensorboard:
        writer.close()


def save_model_epoch(model_file, model):
    """Save model each epoch, in case the script aborts for some reason."""
    save_model(model, model_file + ".pt")
    print(model_file, " saved.")
    return model_file


def plot_some_tensors(batch, plt):
    """Plot the first 5 tensors of the first batch."""
    for i in range(5):
        fig, axes = plt.subplots(1, 1)
        labels = batch["label"]
        x = batch["audio"]
        y = x.cpu()
        y = y[i]
        y = y.squeeze()
        y = y.numpy()
        plt.title("Fake" if labels[i] == 1 else "Real")
        im = axes.imshow(
            y,
            cmap="hot",
            extent=[0, 224 * 10, 2000, 4000],
            vmin=-2,
            vmax=1,
        )
        fig.colorbar(im, ax=axes)
        plt.savefig(f"plots/test-{i}.png")


def _save_stats(
    model_file: str,
    loss_list: list,
    accuracy_list: list,
    validation_list: list,
    test_acc: float,
    args,
    iterations_per_epoch: int,
):
    stats_file = model_file + "_" + str(args.seed) + ".pkl"
    try:
        res = pickle.load(open(stats_file, "rb"))
    except OSError as e:
        res = []
        print(
            e,
            "stats.pickle does not exist, \
              creating a new file.",
        )
    res.append(
        {
            "train_loss": loss_list,
            "train_acc": accuracy_list,
            "val_acc": validation_list,
            "test_acc": test_acc,
            "args": args,
            "iterations_per_epoch": iterations_per_epoch,
        }
    )
    pickle.dump(res, open(stats_file, "wb"))
    print(stats_file, " saved.")


if __name__ == "__main__":
    main()
