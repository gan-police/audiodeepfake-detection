"""Source code to test deepfake detectors in wavelet space."""

import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import models as tv_models

from .data_loader import CombinedDataset
from .models import DeepTestNet, OneDNet, Regression, TestNet
from .new_train_classifier import create_data_loaders, val_test_loop


def _parse_args():
    """Parse cmd line args for training an image classifier."""
    parser = argparse.ArgumentParser(description="Train an image classifier")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="input batch size for testing (default: 128)",
    )
    parser.add_argument(
        "--data-prefix",
        type=str,
        nargs="+",
        default=["./data/source_data_packets"],
        help="shared prefix of the data paths (default: ./data/source_data_packets)",
    )
    parser.add_argument(
        "--nclasses", type=int, default=2, help="number of classes (default: 2)"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="the random seed pytorch works with."
    )
    parser.add_argument(
        "--model",
        choices=[
            "regression",
            "testnet",
            "resnet18",
            "resnet34",
            "onednet",
            "deeptestnet",
        ],
        default="regression",
        help="The model type chosse regression or CNN. Default: Regression.",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="enables a tensorboard visualization.",
    )
    parser.add_argument(
        "--pbar",
        action="store_true",
        help="enables progress bars",
    )
    parser.add_argument(
        "--calc-normalization",
        action="store_true",
        help="calculate normalization for debugging purposes.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of worker processes started by the test and validation data loaders. The training data_loader "
        "uses three times this argument many workers. Hence, this argument should probably be chosen below 10. "
        "Defaults to 2.",
    )
    parser.add_argument(
        "--usemodel",
        type=str,
        help="Path of pre-trained model to be used in testing.",
    )
    return parser.parse_args()


def main():
    """Test a given model."""
    args = _parse_args()
    print(args)

    # fix the seed in the interest of reproducible results.
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    make_binary_labels = args.nclasses == 2
    _train_data_loader, _val_data_loader, test_data_set = create_data_loaders(
        args.data_prefix,
        args.batch_size,
        args.calc_normalization,
    )

    if args.model == "regression":
        model = Regression(args.nclasses, args.frame_size * args.scales)  # type: ignore
    elif args.model == "testnet":
        model = TestNet(classes=args.nclasses, batch_size=args.batch_size)  # type: ignore
    elif args.model == "deeptestnet":
        model = DeepTestNet(classes=args.nclasses, batch_size=args.batch_size)  # type: ignore
    elif args.model == "onednet":
        model = OneDNet(n_input=args.num_of_scales, n_output=args.nclasses)  # type: ignore
    elif args.model == "resnet18":
        model = tv_models.resnet18(weights="IMAGENET1K_V1")  # type: ignore
        model.fc = torch.nn.Linear(
            in_features=512, out_features=args.nclasses, bias=True
        )
        model.get_name = lambda: "ResNet18"  # type: ignore
    else:
        model = tv_models.resnet34(weights="IMAGENET1K_V1")  # type: ignore
        model.fc = torch.nn.Linear(
            in_features=512, out_features=args.nclasses, bias=True
        )
        model.get_name = lambda: "ResNet34"  # type: ignore

    if args.usemodel:
        old_state_dict = torch.load(args.usemodel)
        model.load_state_dict(old_state_dict)

    model.to(device)

    loss_fun = torch.nn.CrossEntropyLoss()

    # Run over the test set.
    print("Testing....")
    if type(test_data_set) is list:
        test_data_set = CombinedDataset(test_data_set)

    test_data_loader = DataLoader(
        test_data_set,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    with torch.no_grad():
        test_acc, _test_loss = val_test_loop(
            test_data_loader,
            model,
            loss_fun,
            make_binary_labels=make_binary_labels,
            pbar=not args.pbar,
            _description="Testing",
        )
        print("test acc", test_acc)


if __name__ == "__main__":
    main()
