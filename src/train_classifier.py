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
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torchvision import models as tv_models

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

import src.data_loader as data_util
from src.models import Regression, TestNet, save_model

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
        "--mean",
        type=float,
        help="Mean of dataset if known.",
    )
    parser.add_argument(
        "--std",
        type=float,
        help="std of dataset if known.",
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
        "--fakedir",
        type=str,
        nargs="+",
        help="Directory containing fake audios. To classify between more than two different GANs chain the \
            folders in this argument, e.g. --fakedir 'gan_folder_one' 'gan_folder_two' 'gan_folder_three'.",
    )
    parser.add_argument(
        "--out-classes",
        type=int,
        default=2,
        help="Number of classes to differentiate.",
    )
    parser.add_argument(
        "--realdir",
        type=str,
        help="Directory containing real audios. To classify between two different GANs put second GAN folder here.",
    )
    parser.add_argument(
        "--usemodel",
        type=str,
        help="Path of pre-trained model to be used in further training. For correct initialization the file name \
            should be of style '*_epoch_(int)_seed_(float).pth'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Manual seed to be used.",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="enables a tensorboard visualization.",
    )
    parser.add_argument(
        "--lrscheduler",
        type=float,
        default=None,
        help="Use Learning Rate scheduler with given factor.",
    )
    parser.add_argument(
        "--evalfirst",
        "--evalfirst",
        action="store_true",
        help="Set if given model shall only be evaluated.",
    )

    return parser.parse_args()


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

    epoch_start = 0
    if args.usemodel:
        path = args.usemodel
        epoch_start = int(path.split("_")[-3])  # extract epoch from file name

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.multiprocessing.set_start_method("spawn")

    # Logs
    time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    day = datetime.datetime.now().strftime("%Y_%m_%d")
    Path(f"logs/{day}/").mkdir(parents=True, exist_ok=True)
    Path("saved_models/").mkdir(parents=True, exist_ok=True)
    init_logger(f"logs/{day}/exp_{time_now}.log")

    # DATA LOADING
    if args.fakedir:
        fake_data_folder = args.fakedir
        data_string = f"Fake: {args.fakedir}"
    else:
        fake_data_folder = [
            # "/home/s6kogase/wavefake/data/generated_audio/ljspeech_parallel_wavegan",
            # "/home/s6kogase/wavefake/data/generated_audio/ljspeech_hifiGAN",
            "/home/s6kogase/wavefake/data/generated_audio/ljspeech_melgan",
            # "/home/s6kogase/wavefake/data/generated_audio/ljspeech_melgan_large",
            # "/home/s6kogase/wavefake/data/generated_audio/ljspeech_multi_band_melgan",
            # "/home/s6kogase/wavefake/data/generated_audio/ljspeech_full_band_melgan",
            # "/home/s6kogase/wavefake/data/generated_audio/ljspeech_waveglow",
            # "/home/s6kogase/wavefake/data/generated_audio/jsut_parallel_wavegan",
            # "/home/s6kogase/wavefake/data/generated_audio/jsut_multi_band_melgan",
            # "/home/s6kogase/wavefake/data/generated_audio/common_voices_prompts_from_conformer_fastspeech2_pwg_ljspeech",
        ]
        data_string = "Fake: melgan"

    if args.out_classes == 2 and len(args.fakedir) > 1:
        raise ValueError(
            "--out-classes cannot be 2 if args.realdir is set and several args.fakedirs are set. \
                Binary classification of two GANs can be done by setting realdir to the second GAN."
        )
    if args.realdir:
        real_data_folder = args.realdir
        data_string += f", {args.realdir}"
    else:
        real_data_folder = "/home/s6kogase/wavefake/data/LJspeech-1.1/wavs"

    fake_test_data_folder = fake_data_folder

    # TODO: amount into used files
    if args.amount % args.frame_size != 0:
        args.amount += args.frame_size - (args.amount % args.frame_size)
    amount = args.amount // (
        args.max_length // args.frame_size
    )  # max train samples used

    if args.model == "resnet18" or args.model == "resnet34":
        input_channels = 3
    else:
        input_channels = 1

    if amount <= len(real_data_folder) + len(fake_data_folder):
        raise ValueError("To little training samples.")

    f_max = args.fmax
    # if args.fmax > args.sample_rate / 2:
    #    f_max = args.sample_rate / 2
    f_min = args.fmin
    # if args.fmin > f_max:
    #    f_min = 80.0

    LOGGER.info("Loading data...")
    trn_dl, val_dl, test_dl = data_util.prepare_dataloaders(
        args,
        device,
        fake_data_folder,
        real_data_folder,
        fake_test_data_folder,
        amount,
        input_channels,
        f_max,
        f_min,
    )

    train_len = len(trn_dl) * args.batch_size

    if args.model == "regression":
        model = Regression(args.out_classes, args.frame_size * args.scales)  # type: ignore
    elif args.model == "testnet":
        model = TestNet(classes=args.out_classes, batch_size=args.batch_size)  # type: ignore
    elif args.model == "resnet18":
        model = tv_models.resnet18(weights="IMAGENET1K_V1")  # type: ignore
        model.fc = torch.nn.Linear(
            in_features=512, out_features=args.out_classes, bias=True
        )
        model.get_name = lambda: "ResNet18"  # type: ignore
    else:
        model = tv_models.resnet34(weights="IMAGENET1K_V1")  # type: ignore
        model.fc = torch.nn.Linear(
            in_features=512, out_features=args.out_classes, bias=True
        )
        model.get_name = lambda: "ResNet34"  # type: ignore

    if args.usemodel:
        old_state_dict = torch.load(args.usemodel)
        model.load_state_dict(old_state_dict)
        model_str = args.usemodel.split("/")[-1].split("_")[0]
    else:
        model_str = model.get_name()

    if torch.cuda.device_count() > 1:
        if use_mult_gpus:
            model = nn.DataParallel(model)  # type: ignore
        else:
            LOGGER.info("Recommended to use multiple gpus.")

    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    gamma = 0.001
    change_lr = False
    if args.lrscheduler is not None:
        change_lr = True
        gamma = args.lrscheduler
    # reduce the learning after 1 epochs by a factor of args.lrscheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    steps = 0

    LOGGER.info(summary(model, (input_channels, args.scales, args.frame_size)))
    if args.m:
        LOGGER.info(args.m)
    LOGGER.info("-------------------------------")
    LOGGER.info("New Experiment")
    LOGGER.info("Parameters:")
    LOGGER.info(f"Seed: {args.seed}")
    LOGGER.info(f"Batch Size: {args.batch_size}")
    LOGGER.info(f"Using Arch: {model_str}")
    LOGGER.info(f"data: {data_string}")
    LOGGER.info(f"Learning Rate: {args.learning_rate}")
    LOGGER.info(f"Weight decay: {args.weight_decay}")
    LOGGER.info(f"Number of scales: {args.scales}")
    LOGGER.info(f"[f_min, f_max]: {f_min, f_max}")
    LOGGER.info(f"Sample rate: {args.sample_rate}")
    LOGGER.info(f"Frame size: {args.frame_size}")
    LOGGER.info(f"Samples per file: {args.max_length}")
    LOGGER.info(
        f"Using {train_len // (args.max_length // args.frame_size)} different files."
    )
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

    """ones = 0
    twos = 0
    threes = 0
    fours = 0
    zeros = 0
    for val_batch in iter(trn_dl):
        ones += torch.sum(val_batch[-1] == 1)
        twos += torch.sum(val_batch[-1] == 2)
        threes += torch.sum(val_batch[-1] == 3)
        fours += torch.sum(val_batch[-1] == 4)
        zeros += torch.sum(val_batch[-1] == 0)
    import pdb; pdb.set_trace()"""

    if args.evalfirst:
        test_dataset(test_dl, model, loss_fn)

    # TRAINER
    for epoch in range(args.epochs):
        if epoch == 0:
            epoch += epoch_start + 1
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

            if batch_idx % 5 == 0:
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

        if change_lr and epoch == 1:
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
        time_now_cur = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        save_model(
            model,
            f"./saved_models/{model_str}_{time_now_cur}_epoch_{epoch}_seed_{args.seed}.pth",
        )

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

    test_dataset(test_dl, model, loss_fn)
    LOGGER.info("-------------------- End of Experiment --------------------")
    LOGGER.info("")

    save_model(model, f"./saved_models/{model_str}_{time_now}.pth")


def test_dataset(test_dl, model, loss_fn):
    """Test classifier with test dataset."""
    LOGGER.info("-----------------")
    LOGGER.info("Testing")

    test_acc, test_loss = val_performance(test_dl, model, loss_fn)
    LOGGER.info(f"Test loss: {test_loss.item()}")
    LOGGER.info(f"Test Accuracy: {test_acc}")
    print(f"Test loss: {test_loss.item()}")
    print(f"Test Accuracy: {test_acc}")


if __name__ == "__main__":
    main()
