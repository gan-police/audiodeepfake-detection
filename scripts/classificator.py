"""Simple binary classificator with CNN for classification of audio deepfakes.

Audio data from WaveFake-Dataset is transformed with Continous Wavelet Transform and
fed to a CNN with a label if fake or real.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

import src.util as util
from src.models import CNN


def get_data() -> tuple[DataLoader, DataLoader]:
    """Get Dataloaders for training and validation dataset."""
    trn_dl = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val, batch_size=batch_size, shuffle=True)
    return trn_dl, val_dl


def train_batch(x, y, model, opt, loss_fn) -> torch.float64:
    """Train a single batch: forward, loss, backward pass."""
    model.train()
    prediction = model(x)
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()


@torch.no_grad()
def accuracy(x, y, model) -> list[torch.BoolTensor]:
    """Calculate accuracy of current model."""
    model.eval()
    prediction = model(x)
    _max_values, argmaxes = prediction.max(-1)
    is_correct = argmaxes == y
    return is_correct.cpu().numpy().tolist()


@torch.no_grad()
def val_loss(x, y, model) -> torch.float64:
    """Forward pass with validation data to calculate val loss."""
    model.eval()
    prediction = model(x)
    val_loss = loss_fn(prediction, y)
    return val_loss.item()


# transformation params
length = 2000  # length of audiosamples
resol = 50  # number of scales of cwt
audio_channels = 1
sample_rate = 16000.0

# training params
amount = 100  # max train samples used
epochs = 5
batch_size = 10

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_folder = "/home/s6kogase/wavefake/data/classifier"
    train_data = util.TransformDataset(
        data_folder,
        device=device,
        sample_rate=sample_rate,
        amount=amount,
        length=length,
    )

    test_size = 0.2  # 20% of total
    test_len = int(len(train_data) * test_size)
    train_len = len(train_data) - test_len
    lengths = [train_len, test_len]
    train, val = torch.utils.data.random_split(train_data, lengths)

    print("Trainset length: ", train_len)
    print("Valset length: ", test_len)

    model = CNN(n_input=audio_channels)
    model.double()
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    # reduce the learning after 5 epochs by a factor of 10
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    trn_dl, val_dl = get_data()

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

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

            if batch_idx % 5 == 0:
                print(
                    f"Train Epoch: {epoch+1} [{batch_idx * len(data)}/{train_len} \
                    ({100. * (batch_idx * len(data)) / train_len:.0f}%)]\tLoss: {batch_loss:.6f}"
                )

        train_epoch_loss = np.array(train_epoch_losses).mean()

        for _batch_idx, (data, target) in enumerate(iter(trn_dl)):
            is_correct = accuracy(data, target, model)
            train_epoch_accuracies.extend(is_correct)
        train_epoch_accuracy = np.mean(train_epoch_accuracies)

        # testing
        for _batch_idx, (data, target) in enumerate(iter(val_dl)):
            val_is_correct = accuracy(data, target, model)
            validation_loss = val_loss(data, target, model)
        val_epoch_accuracy = np.mean(val_is_correct)

        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
        val_losses.append(validation_loss)
        val_accuracies.append(val_epoch_accuracy)

        print("Training loss: ", train_epoch_loss)
        print("Training Accuracy: ", train_epoch_accuracy)
        print("Validation Accuracy: ", val_epoch_accuracy)

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
