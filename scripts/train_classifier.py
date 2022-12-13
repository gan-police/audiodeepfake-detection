"""Simple binary classificator with CNN for classification of audio deepfakes.

Audio data from WaveFake-Dataset is transformed with Continous Wavelet Transform and
fed to a CNN with a label if fake or real.
"""

import logging
import os
import sys
from pathlib import Path
from time import gmtime, strftime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

import src.util as util
import src.models as models

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


def get_data(train, val) -> tuple[DataLoader, DataLoader]:
    """Get Dataloaders for training and validation dataset."""
    trn_dl = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val, batch_size=len(val), shuffle=True)
    return trn_dl, val_dl


def train_batch(x, y, model, opt, loss_fn) -> torch.Tensor:
    """Train a single batch: forward, loss, backward pass."""
    model.train()
    prediction = model(x)
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
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
    make_binary_labels: bool = True,
    _description: str = "Validation",
    pbar: bool = False,
) -> tuple[float, Any]:
    """Test the performance of a model on a data set by calculating the prediction accuracy and loss of the model.

    Port of: https://github.com/gan-police/frequency-forensics/blob/main/src/freqdect/train_classifier.py
    Args:
        data_loader (DataLoader): A DataLoader loading the data set on which the performance should be measured,
            e.g. a test or validation set in a data split.
        model (torch.nn.Module): The model to evaluate.
        loss_fun: The loss function, which is used to measure the loss of the model on the data set
        make_binary_labels (bool): If flag is set, we only classify binarily, i.e. whether an image is real or fake.
            In this case, the label 0 encodes 'real'. All other labels are cosidered fake data, and are set to 1.
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
        print("acc", val_acc, "ok", val_ok, "total", val_total)
    return val_acc, val_loss


# transformation params
length = 40000  # length of audio signal in samples
resol = 128  # number of scales of cwt, and lfccs
lfcc_filter = 128
audio_channels = 1
sample_rate = 22050.0
f_min = 5000.0
f_max = 9000.0

cut = True
max_length = 40000

# training params
amount = 1000  # max train samples used
epochs = 50
batch_size = 32
learning_rate = 0.01

tensorboard = True

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_folder = "/home/s6kogase/wavefake/data/classifier"
    train_data = util.TransformDataset(
        data_folder,
        device=device,
        sample_rate=sample_rate,
        amount=amount,
        length=length,
        f_min=f_min,
        f_max=f_max,
        resolution=resol,
        lfcc_filter=lfcc_filter,
        transform="cwt",
    )

    test_size = 0.2  # 20% of total
    test_len = int(len(train_data) * test_size)
    train_len = len(train_data) - test_len
    lengths = [train_len, test_len]
    train, val = torch.utils.data.random_split(train_data, lengths)

    model = models.TestNet()
    model_str = model.get_name()
    model.double()
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # reduce the learning after 3 epochs by a factor of 10
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    LOGGER.info("Loading data...")
    trn_dl, val_dl = get_data(train, val)

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    steps = 0

    Path("logs").mkdir(parents=True, exist_ok=True)
    init_logger(f"logs/experiments_{strftime('%Y-%m-%d_%H-%M-%S', gmtime())}.log")
    LOGGER.info("-------------------------------")
    LOGGER.info("New Experiment")
    log_str = f"Parameters:\nTrainset length: {train_len}\n"
    log_str += f"Valset length: {test_len}\n"
    log_str += f"Length of input: {max_length}\n"
    log_str += f"Number of scales: {resol}\n"
    log_str += f"LFCCs: {lfcc_filter}\n"
    log_str += f"[f_min, f_max]: {f_min, f_max}\n"
    log_str += f"Sample rate: {sample_rate}\n"
    log_str += f"Batch Size: {batch_size}\n"
    log_str += f"Using Arch: {model_str}"
    LOGGER.info(log_str)

    if tensorboard:
        writer_str = "runs/"
        writer_str += f"{model_str}/"
        writer_str += f"{batch_size}/"
        writer_str += data_folder
        writer_str += f"{learning_rate}_"
        writer_str += f"{resol}_"
        writer_str += f"{length}_"
        writer_str += f"{strftime('%Y-%m-%d_%H:%M:%S', gmtime())}"
        writer = SummaryWriter(writer_str, max_queue=100)

    # plotting first data tensor
    for i in range(10):
        fig, axes = plt.subplots(1, 1)
        labels = next(iter(trn_dl))[1]
        x = next(iter(trn_dl))[0]
        y = x.cpu()
        y = y[0]
        y = y.squeeze()
        y = y.numpy()
        plt.title("Fake" if labels[i].item() == 1 else "Real")
        im = axes.imshow(
            y,
            cmap="hot",
            extent=[0, max_length, f_min, f_max],
            vmin=-100,
            vmax=100,
        )
        plt.savefig(f"plots/test-{i}.png")

    # Trainer
    for epoch in range(epochs):
        LOGGER.info(f"Training data in epoch {epoch+1} of {epochs}.")
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

        # testing
        val_acc, val_loss = val_performance(val_dl, model, loss_fn)

        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
        val_losses.append(val_loss.item())
        val_accuracies.append(val_acc)

        if tensorboard:
            writer.add_scalar("Loss/train", train_epoch_loss, epoch)
            writer.add_scalar("Accuracy/train", train_epoch_accuracy, epoch)
            writer.add_scalar("Loss/validation", val_loss, epoch)
            writer.add_scalar("Accuracy/validation", val_acc, epoch)

            writer.add_scalar("epochs", epoch, steps)

        LOGGER.info(f"Training loss: {train_epoch_loss}")
        LOGGER.info(f"Training Accuracy: {train_epoch_accuracy}")
        LOGGER.info(f"Validation Accuracy: {val_acc}")

    if tensorboard:
        writer.flush()
        writer.close()

    # summary(model, (audio_channels, resol, length))
    epochs = np.arange(epochs) + 1
    print("train losses: ", train_losses)
    print("val losses: ", val_losses)
    print("train accuracies: ", train_accuracies)
    print("val accuracies: ", val_accuracies)
    plt.subplot(211)
    plt.plot(epochs, train_losses, "bo", label="Training loss")
    plt.plot(epochs, val_losses, "r", label="Validation loss")
    plt.title("Training and validation loss with CNN")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid("off")
    plt.subplot(212)
    plt.plot(epochs, train_accuracies, "bo", label="Training accuracy")
    plt.plot(epochs, val_accuracies, "r", label="Validation accuracy")
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
