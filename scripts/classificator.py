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

amount = 500  # max train samples used

device = "cuda" if torch.cuda.is_available() else "cpu"

data_folder = f"{BASE_PATH}/../data/Test"
# real_dir = f"{data_folder}/LJSpeech-1.1/wavs/"
# fake_dir = f"{data_folder}/generated_audio/ljspeech_hifiGAN/"
train_data = util.TransformDataset(
    data_folder, device=device, sample_rate=util.SAMPLE_RATE, amount=amount
)

test_size = 0.2  # 20% of total
test_len = int(len(train_data) * test_size)
train_len = len(train_data) - test_len
lengths = [train_len, test_len]
train, val = torch.utils.data.random_split(train_data, lengths)


class CNN(nn.Module):
    """For classifying deepfakes."""

    def __init__(self) -> None:
        """Define network sturcture."""
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(349184, 256),
            nn.ReLU(),
            nn.Linear(256, 2, bias=False),
        )

    def forward(self, input) -> torch.float64:
        """Forward pass."""
        x = self.cnn(input)
        x = self.fc(x)
        return x


def get_data() -> tuple[DataLoader, DataLoader]:
    """Get Dataloaders for training and validation dataset."""
    trn_dl = DataLoader(train, batch_size=64, shuffle=True)
    val_dl = DataLoader(val, batch_size=len(val), shuffle=True)
    return trn_dl, val_dl


def train_batch(x, y, model, opt, loss_fn) -> torch.float64:
    """Train a single batch: forward, loss, backward pass."""
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


if __name__ == "__main__":
    model = CNN()
    model.double()
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    trn_dl, val_dl = get_data()

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    epochs = 6
    for epoch in range(epochs):
        print(f"epoch {epoch+1} of {epochs}")
        train_epoch_losses, train_epoch_accuracies = [], []
        for _ix, batch in enumerate(iter(trn_dl)):
            x, y = batch
            batch_loss = train_batch(x, y, model, optimizer, loss_fn)
            train_epoch_losses.append(batch_loss)
        train_epoch_loss = np.array(train_epoch_losses).mean()

        for _ix, batch in enumerate(iter(trn_dl)):
            x, y = batch
            is_correct = accuracy(x, y, model)
            train_epoch_accuracies.extend(is_correct)
        train_epoch_accuracy = np.mean(train_epoch_accuracies)

        for _ix, batch in enumerate(iter(val_dl)):
            x, y = batch
            val_is_correct = accuracy(x, y, model)
            validation_loss = val_loss(x, y, model)
        val_epoch_accuracy = np.mean(val_is_correct)

        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
        val_losses.append(validation_loss)
        val_accuracies.append(val_epoch_accuracy)

        print("Training loss: ", train_epoch_loss)
        print("Accuracy: ", train_epoch_accuracy)

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
    plt.show()
