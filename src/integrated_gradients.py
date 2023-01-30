"""Caculate integrated gradients of trained models."""
import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import IntegratedGradients, Occlusion
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
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
    torch.multiprocessing.set_start_method("spawn")

    num_workers = 0
    gans = ["melgan", "lmelgan", "mbmelgan", "fbmelgan", "hifigan", "waveglow", "pwg"]
    seeds = [0, 1, 2, 3, 4]
    gans = ["melgan"]
    seeds = [0]
    wavelets = ["cmor4.6-0.87", "shan0.01-0.4"]
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
            batch_size = 10  # int(data_args[-6])
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
            )
            old_state_dict = torch.load(model_path[0])
            model.load_state_dict(old_state_dict)
            # model = initialize_model(model, model_path)

            model.to(device)
            if model_name == "learndeepnet":
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
                shuffle=False,
                num_workers=num_workers,
            )

            for batch in iter(test_data_loader):
                index = 0
                batch_audios = batch[test_data_loader.dataset.key].cuda(  # type: ignore
                    non_blocking=True
                )
                batch_labels = batch["label"].cuda(non_blocking=True)
                audio = batch_audios[index]
                label = batch_labels[index]

                audio_cwt = model.cwt(audio)  # type: ignore
                integrated_gradients = IntegratedGradients(model)

                """nt = NoiseTunnel(integrated_gradients)"""
                model.zero_grad()
                # Ask the algorithm to attribute our output target to
                attributions_ig = integrated_gradients.attribute(
                    audio_cwt, target=label, n_steps=200
                ).squeeze(0)

                """attributions_ig_nt = nt.attribute(
                    audio_cwt,
                    target=label,
                    baselines=audio_cwt * 0,
                    nt_type='smoothgrad_sq',
                    nt_samples=150,
                    stdevs=0.2,
                    n_steps=50,
                ).squeeze(0)"""

                occlusion = Occlusion(model)
                model.zero_grad()
                attributions_occ = occlusion.attribute(
                    audio_cwt,
                    target=label,
                    strides=2,
                    sliding_window_shapes=(1, 15, 15),
                    baselines=0,
                ).squeeze(0)

                audio_cwt = audio_cwt.squeeze(0)

                fig, axes = plt.subplots(1, 1)

                # Show the original image for comparison
                fig, axes = viz.visualize_image_attr(
                    None,
                    np.transpose(audio_cwt.cpu().detach().numpy(), (1, 2, 0)),
                    method="original_image",
                    title="Original Image",
                    fig_size=(40, 20),
                )
                fig.set_dpi(200)
                plt.savefig("test_cwt.png")

                default_cmap = LinearSegmentedColormap.from_list(
                    "custom blue",
                    [(0, "#ffffff"), (0.25, "#0000ff"), (1, "#0000ff")],
                    N=256,
                )

                fig, axes = plt.subplots(1, 1)
                fig, axes = viz.visualize_image_attr_multiple(
                    np.transpose(
                        attributions_ig.cpu().detach().numpy()[:, :, :500], (1, 2, 0)
                    ),
                    np.transpose(
                        audio_cwt.cpu().detach().numpy()[:, :, :500], (1, 2, 0)
                    ),
                    methods=["original_image", "heat_map", "heat_map", "masked_image"],
                    signs=["all", "positive", "negative", "positive"],
                    show_colorbar=True,
                    cmap=default_cmap,
                    titles=[
                        "Original",
                        "Positive Attribution",
                        "Negative Attribution",
                        "Masked",
                    ],
                    fig_size=(40, 20),
                )
                fig.set_dpi(200)
                plt.savefig("test_cwt_attr.png")

                """fig, axes = plt.subplots(1, 1)
                fig, axes = viz.visualize_image_attr_multiple(
                    np.transpose(
                        attributions_ig_nt.cpu().detach().numpy()[:, :, :500], (1, 2, 0)
                    ),
                    np.transpose(
                        audio_cwt.cpu().detach().numpy()[:, :, :500], (1, 2, 0)
                    ),
                    show_colorbar=True,
                    method="blended_heat_map", sign="absolute_value", outlier_perc=10,
                )
                fig.set_dpi(200)
                plt.savefig("test_cwt_attr_nt.png")"""

                fig, axes = plt.subplots(1, 1)

                fig, axes = viz.visualize_image_attr(
                    np.transpose(
                        attributions_occ.cpu().detach().numpy()[:, :, 500], (1, 2, 0)
                    ),
                    np.transpose(
                        audio_cwt.cpu().detach().numpy()[:, :, 500], (1, 2, 0)
                    ),
                    method="masked_image",
                    sign="positive",
                    show_colorbar=True,
                    title="Masked",
                    fig_size=(18, 6),
                )

                plt.savefig("test_cwt_occl_1.png")

                fig, axes = plt.subplots(1, 1)

                fig, axes = viz.visualize_image_attr_multiple(
                    np.transpose(
                        attributions_occ.cpu().detach().numpy()[:, :, 500], (1, 2, 0)
                    ),
                    np.transpose(
                        audio_cwt.cpu().detach().numpy()[:, :, 500], (1, 2, 0)
                    ),
                    ["original_image", "heat_map", "heat_map", "masked_image"],
                    ["all", "positive", "negative", "positive"],
                    show_colorbar=True,
                    titles=[
                        "Original",
                        "Positive Attribution",
                        "Negative Attribution",
                        "Masked",
                    ],
                    fig_size=(18, 6),
                )

                fig.set_dpi(200)
                plt.savefig("test_cwt_occl.png")


if __name__ == "__main__":
    main()
