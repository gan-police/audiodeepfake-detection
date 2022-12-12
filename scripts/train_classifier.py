"""Simple binary classificator with CNN for classification of audio deepfakes.

Audio data from WaveFake-Dataset is transformed with Continous Wavelet Transform and
fed to a CNN with a label if fake or real.
"""

import os
import sys
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
from src.models import Net


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
length = 500  # length of audiosamples
resol = 64  # number of scales of cwt
audio_channels = 1
sample_rate = 16000.0
f_min = 80.0
f_max = 2000.0

# training params
amount = 1000  # max train samples used
epochs = 5
batch_size = 256
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
        f_max=f_max,
        f_min=f_min,
    )

    test_size = 0.2  # 20% of total
    test_len = int(len(train_data) * test_size)
    train_len = len(train_data) - test_len
    lengths = [train_len, test_len]
    train, val = torch.utils.data.random_split(train_data, lengths)

    print("Trainset length: ", train_len)
    print("Valset length: ", test_len)

    # model = CNN(n_input=audio_channels)
    model = Net(n_classes=2)
    model_str = "Net()"
    model.double()
    model.to(device)
    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # reduce the learning after 3 epochs by a factor of 10
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    trn_dl, val_dl = get_data(train, val)

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    steps = 0

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
    fig, axes = plt.subplots(1, 1)
    x = next(iter(trn_dl))[0]
    y = x.cpu()
    y = y[0]
    y = y.squeeze()
    y = y.numpy()
    im = axes.imshow(
        y,
        cmap="turbo",
        extent=[0, length, 0, sample_rate / 2],
        vmin=-100,
        vmax=0,
    )
    plt.savefig("test.png")

    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        train_epoch_losses, train_epoch_accuracies = [], []

        # training
        for batch_idx, (data, target) in enumerate(iter(trn_dl)):
            batch_loss = train_batch(data, target, model, optimizer, loss_fn)
            train_epoch_losses.append(batch_loss)
            steps += 1

            acc = accuracy(data, target, model)
            train_epoch_accuracies.append(acc.item())

            if batch_idx % 5 == 0:
                print(
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

        print("Training loss: ", train_epoch_loss)
        print("Training Accuracy: ", train_epoch_accuracy)
        print("Validation Accuracy: ", val_acc)

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
