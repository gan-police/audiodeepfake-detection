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
from .models import save_model, compute_parameter_total
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
    parser.add_argument("--dc-shift",
        action="store_true")
    parser.add_argument("--add-noise",
        action="store_true")
    parser.add_argument('--snr',  metavar='N', type=int, nargs='+',
        help='Min Max signal to noise ratio values.')
    parser.add_argument("--contrast",
        action="store_true")
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs (default: 10)"
    )
    return parser.parse_args()



class AudioNet(torch.nn.Module):
    """CNN models used for packet or pixel classification."""

    def __init__(self, classes: int, feature_size = (187, 256)):
        """Create a convolutional neural network (CNN) model.
        Args:
            classes (int): The number of classes or sources to classify.
            feature (str)): A string which tells us the input feature
                we are using.
        """
        super().__init__()
        factor = 64
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, factor, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(factor, factor, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(factor, factor, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(factor, factor*2, 3, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(factor*2, factor*2, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(factor*2, factor*2, 3),
            torch.nn.ReLU(),
        )
        self.classes = classes
        # TODO choose fc. # 55936 626_580 671_580
        self.fc = torch.nn.Linear(616448, self.classes, True)

    def forward(self, inx):
        feats = self.layers(inx)
        feats = torch.reshape(feats, [feats.shape[0], -1])
        return self.fc(feats)


def main():
    args = _parse_args()
    print(args.snr)
    epochs = 50
    batch_size = args.batch_size
    log_scale = True
    block_norm = False
    torch.manual_seed(42)
    # dataset = LearnWavefakeDataset(data_dir='/home/wolter/uni/audiofake/data/ljspeech_22050_44100_0.7_train')
    # dataset = LearnWavefakeDataset(data_dir='/home/wolter/uni/audiofake/data/fake_22050_44100_0.7_melgan_train')
    dataset = LearnWavefakeDataset(data_dir='/home/wolter/uni/audiofake/data/fake_22050_22050_0.7_fb_melgan_train')
    model_str = "ResNet"

    if model_str == "ResNet":
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        model = torchvision.models.resnet18(weights=weights)
        # use the mean color filters.
        model.conv1.in_channels = 1
        model.conv1.weight = torch.nn.Parameter(torch.mean(model.conv1.weight, 1).unsqueeze(1))
        model.avgpool = torch.nn.Identity()
        model.fc = torch.nn.Linear(12288, 2, bias=True)
        model = model.cuda()
    else:
        model = AudioNet(2).cuda()

    print(f"Network size: {compute_parameter_total(model)}")

    cost_fun = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    # preprocess = weights.transforms()

    # normalize
    print("computing mean and std values.", flush=True)
    norm_dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=8000)
    welford = WelfordEstimator()
    with torch.no_grad():
        for batch in tqdm(iter(norm_dataset_loader),
                desc="comp normalization",
                total=len(norm_dataset_loader)
            ):
            packets = compute_pytorch_packet_representation(batch['audio'].cuda(),
                wavelet_str="sym8", max_lev=7, log_scale=log_scale, block_norm=block_norm)
            welford.update(packets.unsqueeze(-1))
        mean, std = welford.finalize()
    print("mean", mean, "std:", std)

    # with open(f"{target_dir}_train/mean_std.pkl", "wb") as f:
    #    pickle.dump([mean.cpu().numpy(), std.cpu().numpy()], f)

    if model_str == "ResNet":
        transforms = torch.nn.Sequential(
            # torchvision.transforms.Resize(size=(224, 224), antialias=True), # padded if missing.
            torchvision.transforms.Normalize(mean, std)
        )
    else:
        transforms = torch.nn.Sequential(
            torchvision.transforms.Normalize(mean, std)
        )


    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_dataset = LearnWavefakeDataset(data_dir='/home/wolter/uni/audiofake/data/fake_22050_22050_0.7_fb_melgan_val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    unkown_val_dataset = LearnWavefakeDataset(data_dir='/home/wolter/uni/audiofake/data/ljspeech_22050_22050_0.7_val')
    unkown_val_loader = torch.utils.data.DataLoader(unkown_val_dataset, batch_size=batch_size)

    def validate(val_loader, name=""):
        ok_sum = 0
        total = 0
        bar  = tqdm(iter(val_loader), desc='validate', total=len(val_loader))
        model.eval()
        ok_dict = {}
        count_dict = {}
        for val_batch in bar:
            with torch.no_grad():
                val_packets = compute_pytorch_packet_representation(val_batch['audio'].cuda(), wavelet_str="sym8", max_lev=7,
                                                                    log_scale=log_scale, block_norm=block_norm)
                val_packets_norm = transforms(val_packets.unsqueeze(1))
                out = model(val_packets_norm)
                ok_mask = (torch.argmax(out, -1) == (val_batch['label'] != 0).cuda())
                ok_sum += sum(ok_mask).cpu().numpy().astype(int)
                total += batch_size
                bar.set_description(f'{name} acc: {ok_sum/total:2.8f}')
                for lbl, okl in zip(val_batch['label'], ok_mask):
                    if lbl.item() not in ok_dict:
                        ok_dict[lbl.item()] = [okl]
                    else:
                        ok_dict[lbl.item()].append(okl)
                    if lbl.item() not in count_dict:
                        count_dict[lbl.item()] = 1
                    else:
                        count_dict[lbl.item()] += 1
        common_keys = ok_dict.keys() & count_dict.keys()
        print([(key, (sum(ok_dict[key])/count_dict[key]).item()) for key in common_keys])

        # print(f"Val acc: {ok/total:2.8f}")


    for epoch in range(epochs):
        bar = tqdm(
                iter(dataset_loader),
                desc="training cnn",
                total=len(dataset_loader)
            )
        loss_list = []

        print("validating...")
        for pos, batch in enumerate(bar):
            if pos % 100 == 0 and pos > 0:
                validate(val_loader, "kown")
                validate(unkown_val_loader, "unkown")

            model.train()
            batch_audios = batch['audio'].cuda()
            if args.contrast:
                batch_audios = contrast(batch_audios)
            if args.dc_shift:
                batch_audios = dc_shift(batch_audios)
            if args.add_noise:
                batch_audios = add_noise(batch_audios)
            batch_labels = (batch['label'].cuda() != 0).type(torch.long)
            packets = compute_pytorch_packet_representation(batch_audios, wavelet_str="sym8", max_lev=7, log_scale=log_scale,
                                                            block_norm=block_norm)
            packets = packets.unsqueeze(1)
            pad_packets = transforms(packets)
            opt.zero_grad()
            out = model(pad_packets)
            cost = cost_fun(out, batch_labels)
            cost.backward()
            opt.step()
            acc = torch.sum((torch.argmax(out, -1) == (batch_labels != 0).cuda()))/args.batch_size
            bar.set_description(f'ce-cost: {cost.item():2.8f}, acc: {acc.item():2.2f}')
            loss_list.append(cost.item())
            # print(cost.item())
        print(f"epch: {epoch:2.2f}, cost: {cost.item():2.8f}")
        

        
        validate(val_loader, "kown")
        validate(unkown_val_loader, "unkown")
    pass

if __name__ == '__main__':
    main()
