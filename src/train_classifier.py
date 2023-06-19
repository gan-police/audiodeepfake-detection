"""Source code to train audio deepfake detectors in wavelet space."""
import argparse
import os
import pickle

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from .data_loader import LearnWavefakeDataset
from .eval_models import val_test_loop
from .models import get_model, save_model
from .ptwt_continuous_transform import get_diff_wavelet
from .utils import add_noise, contrast, set_seed
from .wavelet_math import get_transforms


def create_data_loaders(
    data_prefix: str,
    batch_size: int = 64,
    num_workers: int = 2,
) -> tuple:
    """Create the data loaders needed for training.

    The test set is created outside a loader.

    Args:
        data_prefix (str): Where to look for the data.
        batch_size (int): preferred training batch size.
        num_workers (int): Number of workers for validation and testing.

    Returns:
        dataloaders (tuple): train_data_loader, val_data_loader, test_data_set
    """
    train_data_set = LearnWavefakeDataset(data_prefix + "_train")
    val_data_set = LearnWavefakeDataset(data_prefix + "_val")
    test_data_set = LearnWavefakeDataset(data_prefix + "_test")

    train_data_loader = DataLoader(
        train_data_set,
        batch_size=batch_size,
        shuffle=True,
    )
    val_data_loader = DataLoader(
        val_data_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_data_loader, val_data_loader, test_data_set


def main():
    """Trains a model to classify audios.

    All settings such as which model to use, parameters, normalization, data set path,
    seed etc. are specified via cmd line args.
    All training, validation and testing results are printed to stdout.
    After the training is done, the results are stored in a pickle dump in the 'log' folder.
    The state_dict of the trained model is stored there as well.

    Raises:
        NotImplementedError: If wrong features are combined with the wrong model.
        ValueError: If stft is started with signed log scaling.
    """
    args = _parse_args()
    print(args)
    base_dir = "./exp/log5"
    if not os.path.exists(base_dir + "/models"):
        os.makedirs(base_dir + "/models")
    if not os.path.exists(base_dir + "/tensorboard"):
        os.makedirs(base_dir + "/tensorboard")

    if args.f_max > args.sample_rate / 2:
        print("Warning: maximum analyzed frequency is above nyquist rate.")

    if args.features != "none" and args.model != "lcnn":
        raise NotImplementedError(
            f"LFCC features are currently not implemented for {args.model}."
        )

    path_name = args.data_prefix.split("/")[-1].split("_")

    ckpt_every = args.ckpt_every
    transform = args.transform
    features = args.features
    known_gen_name = path_name[4]
    loss_less = False if args.loss_less == "False" else True

    if args.model == "onednet" and loss_less:
        raise NotImplementedError(
            "OneDNet does not work together with the sign channel."
        )

    label_names = np.array(
        [
            "ljspeech",
            "melgan",
            "hifigan",
            "mbmelgan",
            "fbmelgan",
            "waveglow",
            "pwg",
            "lmelgan",
            "avocodo",
            "bigvgan",
            "bigvganl",
        ]
    )

    if transform == "cwt":
        print("Warning: cwt not tested.")
    elif transform == "stft" and loss_less:
        raise ValueError("Sign channel not possible for stft due to complex data type.")

    model_file = base_dir + "/models/test3/" + path_name[0] + "_"
    if transform == "cwt":
        model_file += "cwt" + str(args.wavelet)
    elif transform == "stft":
        model_file += "stft"
    elif transform == "packets":
        model_file += "packets" + str(args.wavelet)
    model_file += (
        "_"
        + str(args.features)
        + "_"
        + str(args.hop_length)
        + "_"
        + str(args.sample_rate)
        + "_"
        + str(args.window_size)
        + "_"
        + str(args.num_of_scales)
        + "_"
        + str(int(args.f_min))
        + "-"
        + str(int(args.f_max))
        + "_"
        + path_name[3]
        + "_"
        + str(args.learning_rate)
        + "_"
        + str(args.weight_decay)
        + "_"
        + str(args.batch_size)
        + "_"
        + str(args.nclasses)
        + "_"
        + f"{args.epochs}e"
        + "_"
        + str(args.model)
        + "_signs"
        + str(loss_less)
        + "_augc"
        + str(args.aug_contrast)
        + "_augn"
        + str(args.aug_noise)
        + "_power"
        + str(args.power)
        + "_"
        + known_gen_name
        + "_"
        + str(args.seed)
    )

    # fix the seed in the interest of reproducible results.
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.multiprocessing.set_start_method("spawn")

    wavelet = get_diff_wavelet(args.wavelet)
    if wavelet is not None:
        wavelet.bandwidth_par.requires_grad = args.adapt_wavelet
        wavelet.center_par.requires_grad = args.adapt_wavelet

    train_data_loader, val_data_loader, test_data_set = create_data_loaders(
        args.data_prefix,
        args.batch_size,
        args.num_workers,
    )

    if args.unknown_prefix is not None:
        cross_set_val = LearnWavefakeDataset(args.unknown_prefix + "_val")
        cross_set_test = LearnWavefakeDataset(args.unknown_prefix + "_test")
        cross_loader_val = DataLoader(
            cross_set_val,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
        )
        cross_loader_test = DataLoader(
            cross_set_test,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
        )

    validation_list = []
    loss_list = []
    accuracy_list = []
    step_total = 0

    if "doubledelta" in features:
        channels = 60
    elif "delta" in features:
        channels = 40
    elif "lfcc" in features:
        channels = 20
    else:
        channels = int(args.num_of_scales)

    model = get_model(
        model_name=args.model,
        nclasses=args.nclasses,
        num_of_scales=args.num_of_scales,
        flattend_size=args.flattend_size,
        in_channels=2 if loss_less else 1,
        channels=channels,
    )
    model.to(device)

    if args.tensorboard:
        writer_str = base_dir + "/tensorboard/"
        writer_str += f"{args.model}/"
        writer_str += f"{args.transform}/"
        if transform == "cwt" or transform == "packets":
            writer_str += f"{args.wavelet}/"
        writer_str += f"{args.features}/"
        writer_str += f"{args.batch_size}_"
        writer_str += f"{args.learning_rate}_"
        writer_str += f"{args.weight_decay}_"
        writer_str += f"{args.epochs}/"
        writer_str += f"{args.f_min}-"
        writer_str += f"{args.f_max}/"
        writer_str += f"{args.num_of_scales}/"
        writer_str += f"signs{loss_less}/"
        writer_str += f"augc{args.aug_contrast}/"
        writer_str += f"augn{args.aug_noise}/"
        writer_str += f"power{args.power}/"
        writer_str += f"{known_gen_name}/"
        writer_str += f"{args.seed}"
        writer = SummaryWriter(writer_str, max_queue=100)

    loss_fun = torch.nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.5,
            patience=2,
            verbose=True,
        )

    transforms, normalize = get_transforms(
        args,
        args.data_prefix,
        features,
        device,
        wavelet,
        args.calc_normalization,
        pbar=args.pbar,
    )

    for e in tqdm(
        range(args.epochs), desc="Epochs", unit="epochs", disable=not args.pbar
    ):
        # iterate over training data.
        bar = tqdm(
            iter(train_data_loader),
            desc="training cnn",
            total=len(train_data_loader),
            unit="batches",
            disable=not args.pbar,
        )
        print(f"+------------------- Epoch {e+1} -------------------+")
        for it, batch in enumerate(bar):
            model.train()
            batch_audios = batch[train_data_loader.dataset.key].cuda(non_blocking=True)
            batch_labels = (batch["label"].cuda() != 0).type(torch.long)

            if args.aug_contrast:
                batch_audios = contrast(batch_audios)
            if args.aug_noise:
                batch_audios = add_noise(batch_audios)

            optimizer.zero_grad()
            with torch.no_grad():
                freq_time_dt = transforms(batch_audios)
                freq_time_dt_norm = normalize(freq_time_dt)

            out = model(freq_time_dt_norm)
            loss = loss_fun(out, batch_labels)
            acc = (
                torch.sum((torch.argmax(out, -1) == (batch_labels != 0).cuda()))
                / args.batch_size
            )

            bar.set_description(f"ce-cost: {loss.item():2.8f}, acc: {acc.item():2.2f}")

            if it % ckpt_every == 0 and it > 0:
                save_model_epoch(model_file, model)

            loss.backward()
            optimizer.step()
            step_total += 1
            loss_list.append([step_total, e, loss.item()])
            accuracy_list.append([step_total, e, acc.item()])

            if args.tensorboard:
                writer.add_scalar("loss/train", loss.item(), step_total)
                writer.add_scalar("accuracy/train", acc.item(), step_total)
                if step_total == 0:
                    writer.add_graph(model, batch_audios)

            # iterate over val batches.
            if step_total % args.validation_interval == 0:
                val_acc, val_eer, _ = val_test_loop(
                    data_loader=val_data_loader,
                    model=model,
                    batch_size=args.batch_size,
                    normalize=normalize,
                    transforms=transforms,
                    pbar=args.pbar,
                    name="known",
                    label_names=label_names,
                )
                validation_list.append([step_total, e, val_acc])
                if validation_list[-1] == 1.0:
                    print("val acc ideal stopping training.", flush=True)
                    break

                if args.unknown_prefix is not None:
                    cr_val_acc, cr_val_eer, _ = val_test_loop(
                        data_loader=cross_loader_val,
                        model=model,
                        batch_size=args.batch_size,
                        normalize=normalize,
                        transforms=transforms,
                        pbar=args.pbar,
                        name="unknown",
                        label_names=label_names,
                    )

                if args.tensorboard:
                    writer.add_scalar("accuracy/validation", val_acc, step_total)
                    writer.add_scalar("eer/validation", val_eer, step_total)
                    writer.add_scalar(
                        "accuracy/cross_validation", cr_val_acc, step_total
                    )
                    writer.add_scalar("eer/cross_validation", cr_val_eer, step_total)

        if args.unknown_prefix is not None:
            cr_val_acc, cr_val_eer, _ = val_test_loop(
                data_loader=cross_loader_val,
                model=model,
                batch_size=args.batch_size,
                normalize=normalize,
                transforms=transforms,
                pbar=args.pbar,
                name="unknown",
                label_names=label_names,
            )
            if args.use_scheduler:
                scheduler.step(cr_val_eer)

        if args.tensorboard:
            writer.add_scalar("epochs", e, step_total)

        save_model_epoch(model_file, model)  # save model every epoch

    print(validation_list)

    model_file = save_model_epoch(model_file, model)

    # Run over the test set.
    print("Training done testing....")
    test_data_loader = DataLoader(
        test_data_set,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    with torch.no_grad():
        test_acc, test_eer, _ = val_test_loop(
            data_loader=test_data_loader,
            model=model,
            batch_size=args.batch_size,
            normalize=normalize,
            transforms=transforms,
            pbar=not args.pbar,
            name="known",
            label_names=label_names,
        )
        if args.unknown_prefix is not None:
            cr_test_acc, cr_test_eer, _ = val_test_loop(
                data_loader=cross_loader_test,
                model=model,
                batch_size=args.batch_size,
                normalize=normalize,
                transforms=transforms,
                pbar=not args.pbar,
                name="unknown",
                label_names=label_names,
            )
        print("test acc", test_acc)

    if args.tensorboard:
        writer.add_scalar("accuracy/test", test_acc, step_total)
        writer.add_scalar("eer/test", test_eer, step_total)
        writer.add_scalar("accuracy/cross_test", cr_test_acc, step_total)
        writer.add_scalar("eer/cross_test", cr_test_eer, step_total)

    _save_stats(
        model_file,
        loss_list,
        accuracy_list,
        validation_list,
        test_acc,
        args,
        len(iter(train_data_loader)),
    )

    if args.tensorboard:
        writer.close()


def save_model_epoch(model_file, model) -> str:
    """Save model each epoch, in case the script aborts for some reason."""
    save_model(model, model_file + ".pt")
    print(model_file, " saved.")
    return model_file


def _save_stats(
    model_file: str,
    loss_list: list,
    accuracy_list: list,
    validation_list: list,
    test_acc: float,
    args,
    iterations_per_epoch: int,
) -> None:
    stats_file = model_file + ".pkl"
    try:
        res = pickle.load(open(stats_file, "rb"))
    except OSError as e:
        res = []
        print(
            e,
            "stats.pickle does not exist, creating a new file.",
        )
    res.append(
        {
            "train_loss": loss_list,
            "train_acc": accuracy_list,
            "val_acc": validation_list,
            "test_acc": test_acc,
            "args": args,
            "iterations_per_epoch": iterations_per_epoch,
        }
    )
    pickle.dump(res, open(stats_file, "wb"))
    print(stats_file, " saved.")


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
        default=0.0001,
        help="learning rate for optimizer (default: 0.0001)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="weight decay for optimizer (default: 0.01)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs (default: 10)"
    )
    parser.add_argument(
        "--use-scheduler",
        action="store_true",
        help="If plateau scheduler should be used.",
    )

    parser.add_argument(
        "--transform",
        choices=[
            "stft",
            "packets",
        ],
        default="stft",
        help="Use stft instead of cwt in transformation.",
    )
    parser.add_argument(
        "--features",
        choices=["lfcc", "delta", "doubledelta", "none"],
        default="none",
        help="Use features like lfcc, first and second derivatives. \
            Delta and Dooubledelta include lfcc computing. Default: none.",
    )
    parser.add_argument(
        "--num-of-scales",
        type=int,
        default=256,
        help="number of scales used in training set. (default: 256)",
    )
    parser.add_argument(
        "--wavelet",
        type=str,
        default="sym8",
        help="Wavelet to use in wavelet transformations. (default: sym8)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Sample rate of audio. (default: 22050)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=11025,
        help="Window size of audio. (default: 11025)",
    )
    parser.add_argument(
        "--f-min",
        type=float,
        default=1000,
        help="Minimum frequency to analyze in Hz. (default: 1000)",
    )
    parser.add_argument(
        "--f-max",
        type=float,
        default=11025,
        help="Maximum frequency to analyze in Hz. (default: 11025)",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=1,
        help="Hop length in cwt and stft. (default: 100).",
    )
    parser.add_argument(
        "--adapt-wavelet",
        action="store_true",
        help="If differentiable wavelets shall be used.",
    )

    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Log-scale transformed audio.",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=2.0,
        help="Calculate power spectrum of given factor (for stft and packets) (default: 2.0).",
    )
    parser.add_argument(
        "--loss-less",
        choices=[
            "True",
            "False",
        ],
        default="False",
        help="If sign pattern is to be used as second channel, only works for packets.",
    )
    parser.add_argument(
        "--aug-contrast",
        action="store_true",
        help="Use augmentation method contrast.",
    )
    parser.add_argument(
        "--aug-noise",
        action="store_true",
        help="Use augmentation method contrast.",
    )

    parser.add_argument(
        "--calc-normalization",
        action="store_true",
        help="calculate normalization for debugging purposes.",
    )
    parser.add_argument(
        "--mean",
        type=float,
        default=0,
        help="Pre calculated mean. (default: 0)",
    )
    parser.add_argument(
        "--std",
        type=float,
        default=1,
        help="Pre calculated std. (default: 1)",
    )

    parser.add_argument(
        "--data-prefix",
        type=str,
        default="../data/fake",
        help="Shared prefix of the data paths (default: ../data/fake).",
    )
    parser.add_argument(
        "--unknown-prefix",
        type=str,
        help="Shared prefix of the unknown source data paths (default: none).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The random seed pytorch and numpy works with (default: 0).",
    )
    parser.add_argument(
        "--flattend-size",
        type=int,
        default=21888,
        help="Dense layer input size (default: 21888).",
    )
    parser.add_argument(
        "--model",
        choices=[
            "onednet",
            "learndeepnet",
            "lcnn",
        ],
        default="lcnn",
        help="The model type (default: lcnn).",
    )
    parser.add_argument(
        "--nclasses",
        type=int,
        default=2,
        help="Number of output classes in model (default: 2).",
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
        "--num-workers",
        type=int,
        default=2,
        help="Number of worker processes started by the test and validation data loaders (default: 2)",
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=200,
        help="number of training steps after which the model is tested "
        " on the validation data set (default: 200)",
    )
    parser.add_argument(
        "--ckpt-every",
        type=int,
        default=500,
        help="Save model after a fixed number of steps. (default: 500)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
