"""Prepare desired files of on GAN architecture and the real audio dataset.

The resulting files are resampled but not transformed yet to make
gradient flow through wavelets possible.
All files that have less than the desired amount of samples for each file
are discarded. These preprocessing routines help with varying window size
and number of files used in total dynamically. Even though not all files
will be used, at least the same amount of samples are used per audio.

Note: currently only for binary classification.
"""
import argparse
import functools
import pickle
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio

from .data_loader import LearnWavefakeDataset, WelfordEstimator
from .prepare_dataset import get_label, get_label_of_folder, save_to_disk
from .ptwt_continuous_transform import get_diff_wavelet


def load_transform_and_stack(
    path_list: np.ndarray,
    samples: int,
    window_size: int,
    resample_rate: int,
    binary_classification: bool = False,
) -> tuple:
    """Transform a lists of paths into a batches of numpy arrays and record their labels.

    Args:
        path_list (ndarray): An array of Poxis paths strings.
            The stings must follow the convention outlined
            in the get_label function.
        binary_classification (bool): If flag is set, we only classify binarily,
            i.e. whether an audio is real or fake.

    Returns:
        tuple: A numpy array of size
            (preprocessing_batch_size * (samples / window_size), number of channels, window_size)
            and a label list of length preprocessing_batch_size.
    """
    audio_list: list[np.ndarray] = []
    label_list = []
    sox = False

    audio_meta = torchaudio.info(path_list[0])
    rate_factor = resample_rate / audio_meta.sample_rate

    for path_to_audio in path_list:
        audio_meta = torchaudio.info(path_to_audio)
        # check if cut loading is possible: True if resampled audio has more samples than $samples
        if audio_meta.num_frames >= int(rate_factor * audio_meta.num_frames):
            if sox:
                # slower
                # audio should be much greater than rate_factor * num_frames
                audio, sample_rate = torchaudio.load(path_to_audio, normalize=True)
                audio, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
                    audio,
                    int(sample_rate),
                    [["silence", "1", "0.1", "1%", "-1", "0.1", "1%"]],
                )
            else:
                samples_res = int(samples / rate_factor)
                audio, sample_rate = torchaudio.load(
                    path_to_audio, normalize=True, num_frames=samples_res
                )
        else:
            print(
                f"Warning: --sample-number should be bigger than smallest audio with \
                length {int(audio_meta.num_frames * rate_factor)}."
            )
            audio, sample_rate = torchaudio.load(path_to_audio, normalize=True)

        # resample with better window (a bit slower than default hann window)
        audio_res = torchaudio.functional.resample(
            audio, sample_rate, resample_rate, resampling_method="kaiser_window"
        )

        audio_res = audio_res[:, :samples]  # transform
        # cut to non-overlapping equal-sized windows
        framed_audio = audio_res[0].unfold(0, window_size, window_size)

        framed_audio = framed_audio.unsqueeze(1)
        shape = framed_audio.shape

        audio_list.extend(np.array(framed_audio))
        label = np.array(get_label(path_to_audio, binary_classification))
        label_list.extend([label] * shape[0])
    return np.stack(audio_list), label_list


def load_process_store(
    file_list,
    preprocessing_batch_size,
    target_dir,
    label_string,
    window_size,
    samples,
    sample_rate,
    dir_suffix="",
    binary_classification: bool = False,
):
    """Load, process and store a file list according to a processing function.

    Args:
        file_list (list): PosixPath objects leading to source audios.
        preprocessing_batch_size (int): The number of files processed at once.
        target_dir (string): A directory where to save the processed files.
        label_string (string): A label we add to the target folder.
    """
    splits = int(len(file_list) / preprocessing_batch_size)
    batched_files = np.array_split(file_list, splits)
    file_count = 0
    directory = str(target_dir) + "_" + label_string
    all_labels = []
    for current_file_batch in batched_files:
        # load, process and store the current batch training set.
        # Also cut to max-sample size $samples and resample
        audio_batch, labels = load_transform_and_stack(
            current_file_batch,
            samples=samples,
            window_size=window_size,
            resample_rate=sample_rate,
            binary_classification=binary_classification,
        )
        all_labels.extend(labels)
        file_count = save_to_disk(audio_batch, directory, file_count, dir_suffix)
        print(file_count, label_string, "files processed", flush=True)

    # save labels
    with open(f"{directory}{dir_suffix}/labels.npy", "wb") as label_file:
        np.save(label_file, np.array(all_labels))


def load_folder(
    folder: Path,
    train_size: int,
    val_size: int,
    test_size: int,
    min_length: int,
    win_size: int,
    max_allowed_samples: int,
) -> np.ndarray:
    """Create posix-path lists for png files in a folder.

    Given a folder containing portable network graphics (*.png) files
    this functions will create Posix-path lists. A train, test, and
    validation set list is created.

    Args:
        folder: Path to a folder with images from the same source, i.e. A_ffhq .
        train_size: Desired size of the training set.
        val_size: Desired size of the validation set.
        test_size: Desired size of the test set.

    Returns:
        Numpy array with the train, validation and test lists, in this order.

    Raises:
        ValueError: if the requested set sizes are not smaller or equal to the number of images available

    # noqa: DAR401
    """
    file_list = list(folder.glob("./*.wav"))

    file_list = remove_small_audios(file_list, min_length, max_allowed_samples)
    if len(file_list) < train_size + val_size + test_size:
        poss_tr_size = int(len(file_list) * 0.7)
        poss_val_size = int(len(file_list) * 0.1)
        poss_te_size = len(file_list) - poss_tr_size - poss_val_size
        raise ValueError(
            f"Requested set sizes must be smaller or equal to the number of \
            audios with set min_length available. Use maximum {len(file_list)} \
            samples per folder. For example a train_size of {poss_tr_size}, a \
            validation size of {poss_val_size} and test size if {poss_te_size}."
        )
    # split the list into training, validation and test sub-lists.
    train_list = file_list[:train_size]
    validation_list = file_list[train_size : (train_size + val_size)]
    test_list = file_list[(train_size + val_size) : (train_size + val_size + test_size)]
    return np.asarray([train_list, validation_list, test_list], dtype=object)


def remove_small_audios(file_list, min_length, max_allowed_samples):
    """Remove audios with shorter length than min_length."""
    result_list = []
    lost_samples = 0
    max_samples = 0
    for i in range(len(file_list)):
        length = torchaudio.info(file_list[i]).num_frames
        if length >= min_length + 10:
            result_list.append(file_list[i])
            max_samples += min_length
        else:
            lost_samples += length
        if max_samples > max_allowed_samples:
            break
    print(f"Lost samples while loading folder due to audio lengths: {lost_samples}.")
    print(f"Maximum files to be used from this folder: {len(result_list)}")
    print(f"Maximum samples available in this folder: {max_samples}")
    return result_list


def get_max_files(folder_list, min_length, window_size):
    """Get maximum available files per folder that adhere to specifications.

    Used to guarantee that from each folder only the audios of min_length size
    are used and that each folder uses the same amount of files, adding up to
    the same amount of frames when cut with window_size.
    """
    minimal_length = 0
    max_samples = 0
    for folder in folder_list:
        file_list = list(folder.glob("./*.wav"))
        for i in range(len(file_list)):
            length = torchaudio.info(file_list[i]).num_frames
            if length >= min_length + 10:
                max_samples += min_length

        if minimal_length == 0:
            minimal_length = max_samples - (max_samples % window_size)
        minimal_length = min(minimal_length, max_samples - (max_samples % window_size))

    return minimal_length


def pre_process_folder(
    data_folder: str,
    real: Optional[str],
    fake: Optional[str],
    preprocessing_batch_size: int,
    train_size: int,
    val_size: int,
    test_size: int,
    missing_label: int,
    gan_split_factor: float = 1.0,
    wavelet: str = "cmor4.6-0.87",
    samples: int = 8_000,
    window_size: int = 200,
    num_of_scales: int = 128,
    sample_rate: int = 8_000,
    f_min: float = 80,
    f_max: float = 4_000,
    channels: int = 1,
) -> None:
    """Preprocess a folder containing sub-directories with audios from different sources.

    All audios are expected to have minimum the size of samples after being resampled to sample_rate.
    The sub-directories are expected to indicate to label their source in
    their name. For example,  A - for real and B - for GAN generated audios.

    Args:
        data_folder (str): The folder with the real and gan generated audio folders.
        preprocessing_batch_size (int): The batch_size used for audio conversion.
        train_size (int): Desired size of the train subset of each folder.
        val_size (int): Desired size of the validation subset of each folder.
        test_size (int): Desired size of the test subset of each folder.
        missing_label (int): label to leave out of training and validation set
                             (choose from {0, 1, 2, 3, 4, 5, 6, 7, None})
        gan_split_factor (float): factor by which the training and validation subset sizes are scaled for each GAN,
            if a missing label is specified.
        wavelet (str): Wavelet to use in cwt.

    Raises:
        ValueError: If real dir is set but not the fake dir in args.
    """
    nyquist_freq = sample_rate / 2.0  # maximum frequency that can be analyzed
    if f_max >= nyquist_freq:
        f_max = nyquist_freq
    if f_min >= f_max:
        f_min = 80.0

    samples -= samples % window_size  # make samples dividable by window_size
    frames_per_file = samples // window_size
    out_files_count = (train_size + val_size + test_size) * frames_per_file

    data_dir = Path(data_folder)
    folder_name = f"{data_dir.name}_{wavelet}_{int(sample_rate)}_{samples}"
    folder_name += f"_{window_size}_{num_of_scales}_{int(f_min)}-{int(f_max)}_{channels}_{out_files_count}"

    binary_classification = missing_label is not None
    if real is not None:
        # binary classification
        if fake is None:
            raise ValueError(
                "Fake directory is not set. If realdir is set, fakedir must be set as well."
            )
        binary_classification = True
        folder_name += f"_{fake.split('_')[-1]}"
        folder_list = [Path(real), Path(fake)]
    else:
        folder_list = sorted(data_dir.glob("./*"))

    target_dir = data_dir.parent / folder_name

    initial_sample_rate = 22050
    if preprocessing_batch_size > samples / window_size:
        preprocessing_batch_size //= (
            samples // window_size
        )  # batch size will be near given batch size
        preprocessing_batch_size = int(preprocessing_batch_size)

    downsampling_factor = initial_sample_rate / sample_rate
    des_length = int(samples * downsampling_factor)
    max_samples_per_folder = get_max_files(folder_list, des_length, window_size)
    if missing_label is not None:
        # split files in folders into training/validation/test
        func_load_folder = functools.partial(
            load_folder,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            min_length=des_length,
            win_size=window_size,
        )

        train_list = []
        validation_list = []
        test_list = []

        for folder in folder_list:
            if get_label_of_folder(folder) == missing_label:
                test_list.extend(
                    load_folder(
                        folder,
                        train_size=0,
                        val_size=0,
                        test_size=test_size,
                        min_length=des_length,
                        win_size=window_size,
                        max_allowed_samples=max_samples_per_folder,
                    )[2]
                )
            else:
                # real data
                if get_label_of_folder(folder, binary_classification=True) == 0:
                    train_result, val_result, test_result = load_folder(
                        folder,
                        train_size=train_size,
                        val_size=val_size,
                        test_size=test_size,
                        min_length=des_length,
                        win_size=window_size,
                        max_allowed_samples=max_samples_per_folder,
                    )
                # generated data
                else:
                    train_result, val_result, test_result = load_folder(
                        folder,
                        train_size=int(train_size * gan_split_factor),
                        val_size=int(val_size * gan_split_factor),
                        test_size=test_size,
                        min_length=des_length,
                        win_size=window_size,
                        max_allowed_samples=int(
                            max_samples_per_folder * gan_split_factor
                        ),
                    )
                train_list.extend(train_result)
                validation_list.extend(val_result)
                test_list.extend(test_result)

    else:
        # split files in folders into training/validation/test
        func_load_folder = functools.partial(
            load_folder,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            min_length=des_length,
            win_size=window_size,
            max_allowed_samples=max_samples_per_folder,
        )
        with ThreadPoolExecutor(max_workers=len(folder_list)) as pool:
            result_lst = list(pool.map(func_load_folder, folder_list))
        results = np.asarray(result_lst)

        train_list = [aud for folder in results[:, 0] for aud in folder]  # type: ignore
        validation_list = [
            aud for folder in results[:, 1] for aud in folder  # type: ignore
        ]
        test_list = [aud for folder in results[:, 2] for aud in folder]  # type: ignore

    print(f"Using {len(train_list) + len(test_list) + len(validation_list)} files.")
    print(
        f"training samples: {len(train_list) * frames_per_file}, \
            validation samples: {len(validation_list) * frames_per_file}, \
                test_samples: {len(test_list) * frames_per_file}"
    )

    # fix the seed to make results reproducible.
    random.seed(42)
    random.shuffle(train_list)
    random.shuffle(validation_list)
    random.shuffle(test_list)

    if missing_label is not None:
        dir_suffix = f"_missing_{missing_label}"
    else:
        dir_suffix = ""

    # group the sets into smaller batches to go easy on the memory.
    print("processing validation set.", flush=True)
    load_process_store(
        validation_list,
        preprocessing_batch_size,
        target_dir,
        "val",
        dir_suffix=dir_suffix,
        window_size=window_size,
        samples=samples,
        sample_rate=sample_rate,
        binary_classification=binary_classification,
    )
    print("validation set stored")

    # do not use binary label in test set to make performance measurements on the different classes possible
    print("processing test set", flush=True)
    load_process_store(
        test_list,
        preprocessing_batch_size,
        target_dir,
        "test",
        dir_suffix=dir_suffix,
        window_size=window_size,
        samples=samples,
        sample_rate=sample_rate,
        binary_classification=False,
    )
    print("test set stored")

    print("processing training set", flush=True)
    load_process_store(
        train_list,
        preprocessing_batch_size,
        target_dir,
        "train",
        dir_suffix=dir_suffix,
        window_size=window_size,
        samples=samples,
        sample_rate=sample_rate,
        binary_classification=binary_classification,
    )
    print("training set stored.", flush=True)

    # compute training normalization.
    # load train data and compute mean and std
    print("computing mean and std values.", flush=True)
    wavelet = get_diff_wavelet(wavelet)

    train_data_set = LearnWavefakeDataset(
        f"{target_dir}_train{dir_suffix}",
    )
    welford = WelfordEstimator()
    with torch.no_grad():
        for aud_no in range(train_data_set.__len__()):
            welford.update(train_data_set.__getitem__(aud_no)["audio"])
        mean, std = welford.finalize()
    print("mean", mean, "std:", std)
    with open(f"{target_dir}_train{dir_suffix}/mean_std.pkl", "wb") as f:
        pickle.dump([mean.cpu().numpy(), std.cpu().numpy()], f)


def parse_args():
    """Parse command line arguments.

    Folder structure could be for binary classification:
        binary
        ├── fake
        │   ├── B_melgan
        │   |   ├── LJ001-0001_gen.wav
        |   |   ├── ...
        │   |   └── LJ008-0217_gen.wav
        │   └── C_hifigan
        │       ├── LJ001-0001_gen.wav
        |       ├── ...
        │       └── LJ008-0217_gen.wav
        └── real
            └── C_hifigan
                ├── LJ001-0001.wav
                ├── ...
                └── LJ008-0217.wav
        or just:
        ├── A_ljspeech
        │   ├── LJ001-0001_gen.wav
        |   ├── ...
        │   └── LJ008-0217_gen.wav
        ├── B_melgan
        │   ├── LJ001-0001_gen.wav
        |   ├── ...
        │   └── LJ008-0217_gen.wav
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "directory",
        type=str,
        help="The folder with the real and gan generated audio folders.",
    )
    parser.add_argument(
        "--realdir",
        type=str,
        help="The folder with the real audios. If specified the directory argument will be ignored.",
    )
    parser.add_argument(
        "--fakedir",
        type=str,
        help="The folder with the gan generated audios. If specified the directory argument will be ignored.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=10_000,
        help="Desired size of the training subset of each folder. Real size is"
        " (train_size * 2 * floor(sample_number / window_size)). (default: 10_000).",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=2_000,
        help="Desired size of the test subset of each folder. Real size is"
        " (trest_size * 2 * floor(sample_number / window_size)). (default: 2_000).",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=1_100,
        help="Desired size of the validation subset of each folder. Real size is"
        " (val_size * 2 * floor(sample_number / window_size)). (default: 1_100).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="The batch_size used for audio conversion. (default: 2048).",
    )

    parser.add_argument(
        "--missing-label",
        type=int,
        choices=[0, 1, 2, 3, 4, 5, 6, 7],
        default=None,
        help="leave this label out of the training and validation set. Used to test how the models generalize to new "
        "GANs.",
    )
    parser.add_argument(
        "--gan-split-factor",
        type=float,
        default=1.0 / 6.0,
        help="scaling factor for GAN subsets in the binary classification split. If a missing label is specified, the"
        " classification task changes to classifying whether the data was generated or not. In this case, the share"
        " of the GAN subsets in the split sets should be reduced to balance both classes (i.e. real and generated"
        " data). So, for each GAN the training and validation split subset sizes are then calculated as the general"
        " subset size in the split (i.e. the size specified by '--train-size' etc.) times this factor."
        " Defaults to 1./6.",
    )
    parser.add_argument(
        "--wavelet",
        type=str,
        default="cmor4.6-0.87",
        help="The wavelet to use. Choose one from pywt.wavelist(). Defaults to cmor4.6-0.87.",
    )
    parser.add_argument(
        "--sample-number",
        type=int,
        default=44_000,
        help="Maximum number of samples that will be used from each audio file. Default: 44_000.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=4_400,
        help="Size of window of audio file as number of samples. Default: 4_400.",
    )
    parser.add_argument(
        "--scales",
        type=int,
        default=224,
        help="Number of scales for the cwt. Default: 224.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Number of channels of output. Audio has one channel, 3 migth be needed for training. Default: 1.",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=8_000,
        help="Desired sample rate of audio in Hz. Default: 8_000.",
    )
    parser.add_argument(
        "--f-min",
        type=float,
        default=2000,
        help="Minimum frequency to be analyzed in Hz. Default: 2000.",
    )
    parser.add_argument(
        "--f-max",
        type=float,
        default=4_000,
        help="Maximum frequency to be analyzed in Hz. Default: 4_000.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)

    pre_process_folder(
        data_folder=args.directory,
        real=args.realdir,
        fake=args.fakedir,
        preprocessing_batch_size=args.batch_size,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        missing_label=args.missing_label,
        gan_split_factor=args.gan_split_factor,
        wavelet=args.wavelet,
        samples=args.sample_number,
        window_size=args.window_size,
        num_of_scales=args.scales,
        sample_rate=args.sample_rate,
        f_min=args.f_min,
        f_max=args.f_max,
        channels=args.channels,
    )
