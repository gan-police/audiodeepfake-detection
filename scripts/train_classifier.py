"""Simple binary classificator with CNN for classification of audio deepfakes.

Audio data from WaveFake-Dataset is transformed with Continous Wavelet Transform and
fed to a CNN with a label if fake or real.
"""

import argparse
import datetime
import logging
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torchvision import models as tv_models

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

import src.models as models
import src.util as util

LOGGER = logging.getLogger()


def init_logger(log_file) -> None:
    """Init logger handler."""
    LOGGER.setLevel(logging.INFO)

    # create file handler
    fh = logging.FileHandler(log_file)

    # create console handler
    ch = logging.StreamHandler()

    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    LOGGER.addHandler(fh)
    LOGGER.addHandler(ch)


def _parse_args():
    """Parse cmd line args for training an audio classifier."""
    parser = argparse.ArgumentParser(description="Train an audio classifier")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="input batch size for testing (default: 512)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="learning rate for optimizer (default: 1e-3)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0,
        help="weight decay for optimizer (default: 0)",
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="number of epochs (default: 30)"
    )
    parser.add_argument(
        "--model",
        choices=["regression", "testnet", "resnet18", "resnet34"],
        default="testnet",
        help="The model type choose regression, TestNet, ResNet18, ResNet34. Default: testnet.",
    )
    parser.add_argument(
        "--frame-size",
        type=int,
        default=500,
        help="Size of window of audio file as number of samples. Default: 500.",
    )
    parser.add_argument(
        "--scales",
        type=int,
        default=128,
        help="Number of scales for the cwt. Default: 128.",
    )
    parser.add_argument(
        "--amount",
        type=int,
        default=6000,
        help="Max. number of training samples. Will be a little less. Default: 6000.",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=22050,
        help="Desired sample rate of audio in Hz. Default: 22050.",
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=80,
        help="Minimum frequency to be analyzed in Hz. Default: 80.",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=8000,
        help="Maximum frequency to be analyzed in Hz. Default: 8000.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=15000,
        help="Maximum number of samples that will be used from each audio file. Default: 15000.",
    )
    parser.add_argument(
        "--wavelet",
        type=str,
        default="cmor4.6-0.97",
        help="Wavelet to use in CWT. Default: cmor4.6-0.97.",
    )
    parser.add_argument(
        "--m",
        type=str,
        help="Message that will be logged.",
    )

    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="enables a tensorboard visualization.",
    )

    return parser.parse_args()


def get_dataloaders(train, val, batch_size) -> tuple[DataLoader, DataLoader]:
    """Get Dataloaders for training and validation dataset."""
    trn_dl = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val, batch_size=batch_size, shuffle=False)
    return trn_dl, val_dl


def get_mean_std_welford(train_data_set) -> tuple[float, float]:
    """Calculate mean and standard deviation of dataset."""
    LOGGER.info("Calculating mean and standard deviation...")
    welford = util.WelfordEstimator()
    for aud_no in range(train_data_set.__len__()):
        welford.update(train_data_set.__getitem__(aud_no)[0])
    mean, std = welford.finalize()

    return mean.mean().item(), std.mean().item()


def train_batch(x, y, model, opt, loss_fn) -> torch.Tensor:
    """Train a single batch: forward, loss, backward pass."""
    model.train()
    prediction = model(x)
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    opt.step()
    opt.zero_grad()
    return batch_loss.item()


def accuracy(batch_data, batch_labels, model) -> torch.Tensor:
    """Calculate accuracy of current model."""
    with torch.no_grad():
        model.eval()
        prediction = model(batch_data)
        ok_mask = torch.eq(torch.max(prediction, dim=-1)[1], batch_labels)
        acc = torch.sum(ok_mask) / len(batch_labels)
    return acc


def val_performance(
    data_loader,
    model: torch.nn.Module,
    loss_fun,
) -> tuple[float, Any]:
    """Test the performance of a model on a data set by calculating the prediction accuracy and loss of the model.

    Port of: https://github.com/gan-police/frequency-forensics/blob/main/src/freqdect/train_classifier.py
    Args:
        data_loader (DataLoader): A DataLoader loading the data set on which the performance should be measured,
            e.g. a test or validation set in a data split.
        model (torch.nn.Module): The model to evaluate.
        loss_fun: The loss function, which is used to measure the loss of the model on the data set.

    Returns:
        Tuple[float, Any]: The measured accuracy and loss of the model on the data set.
    """
    with torch.no_grad():
        model.eval()
        val_total = 0

        val_ok = 0.0
        for val_batch in iter(data_loader):
            batch_audios, batch_labels = val_batch
            out = model(batch_audios)
            val_loss = loss_fun(torch.squeeze(out), batch_labels)
            ok_mask = torch.eq(torch.max(out, dim=-1)[1], batch_labels)
            val_ok += torch.sum(ok_mask).item()
            val_total += batch_labels.shape[0]
        val_acc = val_ok / val_total
        LOGGER.info(f"ok/total: {val_ok} / {val_total}.")
    return val_acc, val_loss


# transformation params
lfcc_filter = 128
audio_channels = 1

# wavelet = "cmor4.6-0.95"
# wavelet = "mexh"
transform = "cwt"

# training params
lr_scheduler = True
use_mult_gpus = True

plotting = False


def main() -> None:
    """Trains model to classify audio files.

    Some parameters can be specified via cmd line arguments. Results are printed to
    stdout and to log file in a new directory 'logs'.

    Raises:
        ValueError: If args.amount is to little.
    """
    args = _parse_args()
    print(args)

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Logs
    time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    day = datetime.datetime.now().strftime("%Y_%m_%d")
    Path(f"logs/{day}/").mkdir(parents=True, exist_ok=True)
    Path("saved_models/").mkdir(parents=True, exist_ok=True)
    init_logger(f"logs/{day}/exp_{time_now}.log")

    # DATA LOADING
    fake_data_folder = [
        # "/home/s6kogase/wavefake/data/generated_audio/ljspeech_parallel_wavegan",
        # "/home/s6kogase/wavefake/data/generated_audio/ljspeech_hifiGAN",
        # "/home/s6kogase/wavefake/data/generated_audio/ljspeech_melgan",
        "/home/s6kogase/wavefake/data/generated_audio/ljspeech_multi_band_melgan",
        # "/home/s6kogase/wavefake/data/generated_audio/ljspeech_waveglow",
    ]
    real_data_folder = "/home/s6kogase/wavefake/data/LJspeech-1.1/wavs"

    data_string = "LJSpeech_pwg_hifiGAN_melgan_mbmelgan_waveglow"
    data_string = "LJSpeech_mbmelgan"

    fake_test_data_folder = fake_data_folder

    amount = args.amount * args.frame_size // args.max_length  # max train samples used

    if args.model == "resnet18" or args.model == "resnet34":
        input_channels = 3
    else:
        input_channels = 1

    if amount <= len(real_data_folder) + len(fake_data_folder):
        raise ValueError("To little training samples.")

    if args.fmax > args.sample_rate / 2:
        f_max = args.sample_rate / 2
    else:
        f_max = args.fmax
    if args.fmin >= f_max:
        f_min = 80.0
    else:
        f_min = args.fmin

    train_data_real = util.TransformDataset(
        real_data_folder,
        device=device,
        sample_rate=args.sample_rate,
        max_length=args.max_length,
        frame_size=args.frame_size,
        f_min=f_min,
        f_max=f_max,
        resolution=args.scales,
        lfcc_filter=lfcc_filter,
        transform=transform,
        wavelet=args.wavelet,
        from_path=0,
        to_path=amount // 2,
        channels=input_channels,
    )
    train_data_fake = util.TransformDataset(
        fake_data_folder,
        device=device,
        sample_rate=args.sample_rate,
        max_length=args.max_length,
        frame_size=args.frame_size,
        f_min=f_min,
        f_max=f_max,
        resolution=args.scales,
        lfcc_filter=lfcc_filter,
        transform=transform,
        wavelet=args.wavelet,
        from_path=0,
        to_path=amount // 2,
        channels=input_channels,
    )

    train_data = torch.utils.data.ConcatDataset([train_data_real, train_data_fake])

    test_size = int(amount * 0.2)  # 30% of total training samples

    val_data_real = util.TransformDataset(
        real_data_folder,
        device=device,
        sample_rate=args.sample_rate,
        max_length=args.max_length,
        frame_size=args.frame_size,
        f_min=f_min,
        f_max=f_max,
        resolution=args.scales,
        lfcc_filter=lfcc_filter,
        transform=transform,
        wavelet=args.wavelet,
        from_path=(amount + 5) // 2,
        to_path=(amount + test_size + 1) // 2,
        channels=input_channels,
    )

    val_data_fake = util.TransformDataset(
        fake_data_folder,
        device=device,
        sample_rate=args.sample_rate,
        max_length=args.max_length,
        frame_size=args.frame_size,
        f_min=f_min,
        f_max=f_max,
        resolution=args.scales,
        lfcc_filter=lfcc_filter,
        transform=transform,
        wavelet=args.wavelet,
        from_path=(amount + 5) // 2,
        to_path=(amount + test_size + 5) // 2,
        channels=input_channels,
    )

    val_data = torch.utils.data.ConcatDataset([val_data_real, val_data_fake])

    test_len = len(val_data)
    train_len = len(train_data)

    out_classes = 2
    if args.model == "regression":
        model = models.Regression(
            classes=out_classes, flt_size=args.frame_size * args.scales
        )
    elif args.model == "testnet":
        model = models.TestNet(classes=out_classes, batch_size=args.batch_size)
    elif args.model == "resnet18":
        model = tv_models.resnet18(weights="IMAGENET1K_V1")
        model.fc = torch.nn.Linear(in_features=512, out_features=out_classes, bias=True)
        model.get_name = lambda: "ResNet18"
    else:
        model = tv_models.resnet34(weights="IMAGENET1K_V1")
        model.fc = torch.nn.Linear(in_features=512, out_features=out_classes, bias=True)
        model.get_name = lambda: "ResNet34"

    model_str = model.get_name()

    if torch.cuda.device_count() > 1:
        if use_mult_gpus:
            model = nn.DataParallel(model)
        else:
            LOGGER.info("Recommended to use multiple gpus.")

    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    reduce_lr_each = args.epochs // 4
    # reduce the learning after reduce_lr_each epochs by a factor of 2
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=reduce_lr_each, gamma=0.4
    )

    LOGGER.info("Loading data...")
    trn_dl, val_dl = get_dataloaders(train_data, val_data, args.batch_size)
    mean, std = get_mean_std_welford(
        train_data
    )  # calculate mean, std only on train data
    train_data_real.set_mean_std(mean, std)
    train_data_fake.set_mean_std(mean, std)
    val_data_real.set_mean_std(mean, std)
    val_data_fake.set_mean_std(mean, std)

    test_data_fake = util.TransformDataset(
        fake_test_data_folder,
        device=device,
        sample_rate=args.sample_rate,
        max_length=args.max_length,
        frame_size=args.frame_size,
        f_min=f_min,
        f_max=f_max,
        resolution=args.scales,
        lfcc_filter=lfcc_filter,
        transform=transform,
        wavelet=args.wavelet,
        from_path=(amount + test_size + 5) // 2,
        to_path=(amount + 2 * (test_size + 5)) // 2,
        mean=mean,
        std=std,
        channels=input_channels,
    )
    test_data_real = util.TransformDataset(
        real_data_folder,
        device=device,
        sample_rate=args.sample_rate,
        max_length=args.max_length,
        frame_size=args.frame_size,
        f_min=f_min,
        f_max=f_max,
        resolution=args.scales,
        lfcc_filter=lfcc_filter,
        transform=transform,
        wavelet=args.wavelet,
        from_path=(amount + test_size + 5) // 2,
        to_path=(amount + 2 * (test_size + 5)) // 2,
        mean=mean,
        std=std,
        channels=input_channels,
    )
    test_data = torch.utils.data.ConcatDataset([test_data_real, test_data_fake])

    test_dl = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    LOGGER.info(f"Test dataset length: {len(test_dl) * args.batch_size}")

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    steps = 0

    LOGGER.info(summary(model, (input_channels, args.scales, args.frame_size)))
    if args.m:
        LOGGER.info(args.m)
    LOGGER.info("-------------------------------")
    LOGGER.info("New Experiment")
    LOGGER.info(f"mean: {mean}, std: {std}")
    LOGGER.info("Parameters:")
    LOGGER.info(f"Trainset length: {train_len}")
    LOGGER.info(f"Valset length: {test_len}")
    LOGGER.info(f"Learning Rate: {args.learning_rate}")
    LOGGER.info(f"Weight decay: {args.weight_decay}")
    LOGGER.info(f"Number of scales: {args.scales}")
    LOGGER.info(f"LFCCs: {lfcc_filter}")
    LOGGER.info(f"[f_min, f_max]: {f_min, f_max}")
    LOGGER.info(f"Sample rate: {args.sample_rate}")
    LOGGER.info(f"Batch Size: {args.batch_size}")
    LOGGER.info(f"Using Arch: {model_str}")
    LOGGER.info(f"Transform: {transform}")
    LOGGER.info(f"data: {data_string}")
    LOGGER.info(f"Frame size: {args.frame_size}")
    LOGGER.info(f"Samples per file: {args.max_length}")
    LOGGER.info(f"Wavelet: {args.wavelet}")

    if args.tensorboard:
        writer_str = "runs/"
        writer_str += f"{model_str}/"
        writer_str += f"{args.batch_size}/"
        writer_str += f"{data_string}/"
        writer_str += f"{args.learning_rate}_"
        writer_str += f"{args.weight_decay}_"
        writer_str += f"{args.scales}_"
        writer_str += f"{args.frame_size}_"
        writer_str += time_now
        writer = SummaryWriter(writer_str, max_queue=100)

    if plotting:
        LOGGER.info("Plotting some input tensors...")
        # plotting first 10 data tensors
        for i in range(6):
            fig, axes = plt.subplots(1, 1)
            labels = next(iter(trn_dl))[1]
            x = next(iter(trn_dl))[0]
            y = x.cpu()
            y = y[i]
            y = y.squeeze()
            y = y.numpy()
            plt.title("Fake" if labels[i].item() == 1 else "Real")
            im = axes.imshow(
                y,
                cmap="hot",
                extent=[0, args.frame_size * 10, f_min, f_max],
                vmin=-1,
                vmax=3,
            )
            fig.colorbar(im, ax=axes)
            plt.savefig(f"plots/test-{i}.png")

    # TRAINER
    for epoch in range(args.epochs):
        LOGGER.info("+------+")
        LOGGER.info(f"Training data in epoch {epoch+1} of {args.epochs}.")
        LOGGER.info(f"Learning rate: {scheduler.get_last_lr()}")
        train_epoch_losses, train_epoch_accuracies = [], []

        # training
        for batch_idx, (data, target) in enumerate(iter(trn_dl)):
            batch_loss = train_batch(data, target, model, optimizer, loss_fn)
            train_epoch_losses.append(batch_loss)
            steps += 1

            acc = accuracy(data, target, model)
            train_epoch_accuracies.append(acc.item())

            if batch_idx % 2 == 0:
                LOGGER.info(
                    f"Train Epoch: {epoch+1} [{batch_idx * len(data)}/{train_len} \
                    ({100. * (batch_idx * len(data)) / train_len:.0f}%)]\tLoss: {batch_loss:.6f}"
                )

        train_epoch_loss = np.array(train_epoch_losses).mean()
        train_epoch_accuracy = np.mean(train_epoch_accuracies)

        # validation
        val_acc, val_loss = val_performance(val_dl, model, loss_fn)

        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
        val_losses.append(val_loss.item())
        val_accuracies.append(val_acc)

        if lr_scheduler:
            scheduler.step()

        if args.tensorboard:
            writer.add_scalar("Loss/train", train_epoch_loss, epoch)
            writer.add_scalar("Accuracy/train", train_epoch_accuracy, epoch)
            writer.add_scalar("Loss/validation", val_loss, epoch)
            writer.add_scalar("Accuracy/validation", val_acc, epoch)

            writer.add_scalar("epochs", epoch, steps)

        LOGGER.info(f"Training loss: {train_epoch_loss}")
        LOGGER.info(f"Training Accuracy: {train_epoch_accuracy}")
        LOGGER.info(f"Validation Accuracy: {val_acc}")

        if (
            train_epoch_accuracy == 1.0
            and train_accuracies[-2] == 1.0
            and train_accuracies[-3] == 1.0
        ):
            LOGGER.info("Training accuracy ideal, stopping training.")
            break

    if args.tensorboard:
        writer.flush()
        writer.close()

    LOGGER.info("-----------------")
    LOGGER.info("Testing")

    test_acc, test_loss = val_performance(test_dl, model, loss_fn)
    LOGGER.info(f"Test loss: {test_loss.item()}")
    LOGGER.info(f"Test Accuracy: {test_acc}")

    models.save_model(model, f"./saved_models/{model_str}_{time_now}.pth")

    if plotting:
        plt.clf()
        # Plotting
        epochs_scala = np.arange(args.epochs) + 1
        print("train losses: ", train_losses)
        print("val losses: ", val_losses)
        print("train accuracies: ", train_accuracies)
        print("val accuracies: ", val_accuracies)
        plt.subplot(211)
        plt.plot(epochs_scala, train_losses, "bo", label="Training loss")
        plt.plot(epochs_scala, val_losses, "r", label="Validation loss")
        plt.title("Training and validation loss with CNN")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid("off")
        plt.subplot(212)
        plt.plot(epochs_scala, train_accuracies, "bo", label="Training accuracy")
        plt.plot(epochs_scala, val_accuracies, "r", label="Validation accuracy")
        plt.title("Training and validation accuracy with CNN")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.gca().set_yticklabels(
            ["{:.0f}%".format(x * 100) for x in plt.gca().get_yticks()]
        )
        plt.legend()
        plt.grid("off")
        # plt.show()
        plt.savefig("test-run.png")


if __name__ == "__main__":
    main()
