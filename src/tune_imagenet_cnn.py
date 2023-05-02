import argparse
import os
import pickle
from typing import Any, Optional

import numpy as np
import torch
import torchvision
from pywt import ContinuousWavelet
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from .data_loader import LearnWavefakeDataset, WelfordEstimator
from .models import LearnDeepTestNet, LearnNet, OneDNet, save_model
from .wavelet_math import compute_pytorch_packet_representation

from .augmentation import add_noise, dc_shift, contrast


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
        "--epochs", type=int, default=10, help="number of epochs (default: 10)"
    )



def main():
    epochs = 50
    batch_size = 64
    torch.manual_seed(42)
    # dataset = LearnWavefakeDataset(data_dir='/home/wolter/uni/audiofake/data/ljspeech_22050_44100_0.7_train')
    dataset = LearnWavefakeDataset(data_dir='/home/wolter/uni/audiofake/data/fake_22050_44100_0.7_melgan_train')
    # mean, std = dataset._load_mean_std()

    weights = torchvision.models.ResNet50_Weights.DEFAULT
    model = torchvision.models.resnet50(weights=weights)
    # use the mean color filters.
    model.conv1.in_channels = 1
    model.conv1.weight = torch.nn.Parameter(torch.mean(model.conv1.weight, 1).unsqueeze(1))
    model.fc = torch.nn.Linear(2048, 9, bias=True)
    model = model.cuda()

    cost_fun = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    # preprocess = weights.transforms()

    # normalize
    print("computing mean and std values.", flush=True)
    norm_dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=5000)
    welford = WelfordEstimator()
    with torch.no_grad():
        for batch in tqdm(iter(norm_dataset_loader),
                desc="comp normalization",
                total=len(norm_dataset_loader)
            ):
            packets = compute_pytorch_packet_representation(batch['audio'].cuda(), wavelet_str="sym8", max_lev=8)
            welford.update(packets.unsqueeze(-1))
        mean, std = welford.finalize()
    print("mean", mean, "std:", std)

    # with open(f"{target_dir}_train/mean_std.pkl", "wb") as f:
    #    pickle.dump([mean.cpu().numpy(), std.cpu().numpy()], f)

    transforms = torch.nn.Sequential(
        torchvision.transforms.Resize(size=(224, 224), antialias=True),
        torchvision.transforms.Normalize(mean, std)
    )


    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    val_dataset = LearnWavefakeDataset(data_dir='/home/wolter/uni/audiofake/data/ljspeech_22050_44100_0.7_val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    for epoch in range(epochs):
        bar = tqdm(
                iter(dataset_loader),
                desc="training cnn",
                total=len(dataset_loader)
            )
        loss_list = []
        for batch in bar:
            batch_audios = batch['audio'].cuda(non_blocking=True)
            batch_audios = add_noise(dc_shift(contrast(batch_audios)))
            batch_labels = batch['label'].cuda(non_blocking=True)
            packets = compute_pytorch_packet_representation(batch_audios, wavelet_str="sym8", max_lev=8)
            packets = packets.unsqueeze(1)
            pad_packets = transforms(packets)
            opt.zero_grad()
            out = model(pad_packets)
            cost = cost_fun(out, batch_labels)
            cost.backward()
            opt.step()
            bar.set_description(f'ce-cost: {cost.item():2.5f}')
            loss_list.append(cost.item())
            # print(cost.item())
            pass
        print(f"epch: {epoch:2.2f}, cost: {cost.item():2.2f}")
        # validate
        ok = 0
        total = 0
        for val_batch in tqdm(iter(val_loader), desc='validate', total=len(val_loader)):
            with torch.no_grad():
                val_packets = compute_pytorch_packet_representation(val_batch['audio'].cuda(), wavelet_str="sym8", max_lev=8)
                val_packets_norm = transforms(val_packets.unsqueeze(1))
                out = model(val_packets_norm)
                ok += sum((torch.argmax(out, -1) == (val_batch['label'] != 0).cuda()).cpu().numpy().astype(int))
                total += batch_size
        print(f"Val acc: {ok/total:2.2f}")
    pass

if __name__ == '__main__':
    main()
