"""Caculate integrated gradients of trained models."""
import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import IntegratedGradients, NoiseTunnel, Saliency
from torch.utils.data import DataLoader

from .learn_direct_train_classifier import (
    create_data_loaders,
    create_data_loaders_learn,
    get_model,
)
from .ptwt_continuous_transform import get_diff_wavelet


def main() -> None:
    """Calculate integradet gradients."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # torch.multiprocessing.set_start_method("spawn")

    num_workers = 0
    gans = ["melgan", "lmelgan", "mbmelgan", "fbmelgan", "hifigan", "waveglow", "pwg"]
    seeds = [0, 1, 2, 3, 4]
    gans = ["melgan"]
    seeds = [0]
    wavelets = ["cmor3.3-4.17", "cmor4.6-0.87", "shan0.01-0.4"]
    cu_wv = wavelets[0]

    for gan in gans:
        print(f"Evaluating {gan}...", flush=True)
        for seed in seeds:
            print(f"seed: {seed}")
            torch.manual_seed(seed)
            model_path = [
                f"/home/s6kogase/code/log/fake_{cu_wv}_22050_11025_150_1000-9500"
            ]
            model_path[0] += f"_0.7_{gan}_0.0001_128_2_10e_learndeepnet_False_{seed}.pt"
            data_args = model_path[0].split("/")[-1].split(".pt")[0].split("_")
            model_name = data_args[-3]
            nclasses = int(data_args[-5])
            batch_size = 1  # int(data_args[-6])
            wavelet = get_diff_wavelet(data_args[1])
            f_min = float(data_args[5].split("-")[0])
            f_max = float(data_args[5].split("-")[1])
            sample_rate = int(data_args[2])
            num_of_scales = int(data_args[4])

            data_dir = "/home/s6kogase/data"
            test_data_dir = [
                f"/home/s6kogase/data/fake_cmor4.6-0.87_22050_8000_11025_224_80-4000_1_0.7_{gan}"
            ]

            if test_data_dir is None:
                test_data_dir = [data_dir + "/" + "_".join(data_args[:10])]

            model = get_model(
                wavelet,
                model_name,
                nclasses,
                batch_size,
                f_min,
                f_max,
                sample_rate,
                num_of_scales,
                raw_input=False,
                flattend_size=21888,
            )
            old_state_dict = torch.load(model_path[0])
            model.load_state_dict(old_state_dict)
            # model = initialize_model(model, model_path)

            model.to(device)
            if (
                model_name == "learndeepnet"
                or model_name == "learnnet"
                or model_name == "onednet"
            ):
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
                shuffle=True,
                num_workers=num_workers,
            )
            with torch.no_grad():
                attributions_ig = torch.zeros(
                    (batch_size, 1, 150, 11025), device="cuda"
                )
                attributions_occ = torch.zeros(
                    (batch_size, 1, 150, 11025), device="cuda"
                )
                attributions_ig_nt = torch.zeros(
                    (batch_size, 1, 150, 11025), device="cuda"
                )
                attributions_sals = torch.zeros(
                    (batch_size, 1, 150, 11025), device="cuda"
                )
            index = 0

            times = 100
            for batch in iter(test_data_loader):
                index += 1
                batch_audios = batch[test_data_loader.dataset.key].cuda(  # type: ignore
                    non_blocking=True
                )
                batch_labels = batch["label"].cuda(non_blocking=True)
                audio = batch_audios
                label = batch_labels

                audio_cwt = model.cwt(audio)  # type: ignore
                integrated_gradients = IntegratedGradients(model)

                nt = NoiseTunnel(integrated_gradients)
                model.zero_grad()

                attributions_ig += integrated_gradients.attribute(
                    audio_cwt, target=label, n_steps=100
                ).squeeze(0)

                attributions_ig_nt += nt.attribute(
                    audio_cwt,
                    target=label,
                    baselines=audio_cwt * 0,
                    nt_type="smoothgrad_sq",
                    nt_samples=5,
                    stdevs=0.2,
                    n_steps=10,
                ).squeeze(0)

                """occlusion = Occlusion(model)
                model.zero_grad()
                attributions_occ += occlusion.attribute(
                    audio_cwt,
                    target=label,
                    strides=30,
                    sliding_window_shapes=(1, 100, 100),
                    baselines=0,
                ).squeeze(0)"""
                # import pdb; pdb.set_trace()

                saliency = Saliency(model)
                attributions_sals += saliency.attribute(
                    audio_cwt, target=label
                ).squeeze(0)

                torch.cuda.empty_cache()
                print(index)
                if index == times:
                    attributions_ig /= times
                    attributions_occ /= times
                    attributions_ig_nt /= times
                    attributions_sals /= times
                    break

            audio_cwt = torch.mean(audio_cwt, dim=0)
            attributions_ig = torch.mean(attributions_ig, dim=0)
            attributions_occ = torch.mean(attributions_occ, dim=0)
            attributions_ig_nt = torch.mean(attributions_ig_nt, dim=0)
            attributions_sals = torch.mean(attributions_sals, dim=0)
            # audio_cwt = audio_cwt.squeeze(0)

            inital = np.transpose(audio_cwt.cpu().detach().numpy()[:, :, :], (1, 2, 0))
            attr_ig = np.transpose(
                attributions_ig.cpu().detach().numpy()[:, :, :], (1, 2, 0)
            )
            attr_ig_nt = np.transpose(
                attributions_ig_nt.cpu().detach().numpy()[:, :, :], (1, 2, 0)
            )
            attr_occ = np.transpose(
                attributions_occ.cpu().detach().numpy()[:, :, :], (1, 2, 0)
            )
            attr_sal = np.transpose(
                attributions_sals.cpu().detach().numpy()[:, :, :], (1, 2, 0)
            )

            # Todo: some refactoring...
            extent = [0, 11025, 1000, 9500]
            cmap = "PRGn"
            fig, axes = plt.subplots(1, 1)
            im = axes.imshow(inital.squeeze(2), cmap="turbo", extent=extent)
            fig.set_size_inches(40, 20, forward=True)

            fig.colorbar(im, ax=axes)
            fig.set_dpi(200)
            plt.savefig(f"test_cwt_{gan}.png")

            fig, axes = plt.subplots(1, 1)
            im = axes.imshow(attr_ig.squeeze(2), cmap=cmap, extent=extent)
            fig.set_size_inches(40, 20, forward=True)
            fig.colorbar(im, ax=axes)
            fig.set_dpi(200)
            plt.savefig(f"test_cwt_attr_{gan}.png")

            fig, axes = plt.subplots(1, 1)
            im = axes.imshow(attr_ig_nt.squeeze(2), cmap="hot", extent=extent)
            fig.set_size_inches(40, 20, forward=True)
            fig.colorbar(im, ax=axes)
            fig.set_dpi(200)
            plt.savefig(f"test_cwt_attr_nt_{gan}.png")

            fig, axes = plt.subplots(1, 1)
            im = axes.imshow(attr_occ.squeeze(2), cmap=cmap, extent=extent)
            fig.set_size_inches(40, 20, forward=True)
            fig.colorbar(im, ax=axes)
            fig.set_dpi(200)
            plt.savefig(f"test_cwt_attr_occl_{gan}.png")

            fig, axes = plt.subplots(1, 1)
            im = axes.imshow(
                attr_sal.squeeze(2), cmap="hot", extent=extent, vmax=0.0012, vmin=0.0
            )
            fig.set_size_inches(40, 20, forward=True)
            fig.colorbar(im, ax=axes)
            fig.set_dpi(200)
            plt.savefig(f"test_cwt_attr_sal_{gan}.png")


if __name__ == "__main__":
    main()
