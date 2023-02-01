"""Prepare all files of on GAN architecture and the real audio dataset.

The resulting files are resampled but not transformed yet to make
gradient flow through wavelets possible.

Note: currently only for binary classification.
"""
import argparse
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio

from .data_loader import LearnWavefakeDataset, WelfordEstimator
from .prepare_dataset import get_label, get_label_of_folder, save_to_disk
from .ptwt_continuous_transform import get_diff_wavelet


def shuffle_random(a, b) -> tuple[list, list]:
    """Shuffle two arrays randomly in the same order."""
    c: np.ndarray = np.c_[a.reshape(len(a), -1), b.reshape(len(b), -1)]
    a2: np.ndarray = c[:, : a.size // len(a)].reshape(a.shape)
    b2: np.ndarray = c[:, a.size // len(a) :].reshape(b.shape)
    np.random.shuffle(c)

    return a2.tolist(), b2.tolist()


def load_transform_and_stack(
    path_list: np.ndarray,
    frame_list: np.ndarray,
    window_size: int,
    resample_rate: int,
    binary_classification: bool = False,
) -> tuple[np.ndarray, list]:
    """Transform a lists of paths into a batches of numpy arrays and record their labels.

    Args:
        path_list (ndarray): An array of Poxis paths strings.
        frame_list (ndarray): Array of frames that can be cut from audio at path in path_list
                              at the same index.
        window_size (int): Size of desired output tensor for each training sample before
                           audio is resampled.
        resample_rate (int): Desired sample rate of audio after resampling it in Hz.
        binary_classification (bool): If flag is set, we only classify binarily,
            i.e. whether an audio is real or fake.

    Returns:
        tuple: A numpy array of size
            (preprocessing_batch_size * (samples / window_size), number of channels, window_size)
            and a label list of length preprocessing_batch_size.
    """
    audio_list: list[np.ndarray] = []
    label_list = []

    old_win_size = window_size
    window_size *= resample_rate / torchaudio.info(path_list[0]).sample_rate
    window_size = int(window_size)
    for i in range(len(path_list)):
        # cut as much as possible from current audio
        audio, sample_rate = torchaudio.load(
            path_list[i], normalize=True, num_frames=int(old_win_size * frame_list[i])
        )
        # resample with better window (a bit slower than default hann window)
        audio_res = torchaudio.functional.resample(
            audio, sample_rate, resample_rate, resampling_method="kaiser_window"
        )
        # cut to non-overlapping equal-sized windows
        framed_audio = audio_res[0].unfold(0, window_size, window_size)

        framed_audio = framed_audio.unsqueeze(1)
        audio_list.extend(np.array(framed_audio))
        label = np.array(get_label(path_list[i], binary_classification))
        label_list.extend([label] * framed_audio.shape[0])
    return np.stack(audio_list), label_list


def load_process_store(
    file_list,
    frames_list,
    preprocessing_batch_size,
    target_dir,
    label_string,
    window_size,
    sample_rate,
    binary_classification: bool = False,
) -> None:
    """Load, process and store a file list according to a processing function.

    Args:
        file_list (list): PosixPath objects leading to source audios.
        preprocessing_batch_size (int): The number of files processed at once.
        target_dir (string): A directory where to save the processed files.
        label_string (string): A label we add to the target folder.

    Raises:
        ValueError: if datasets are not distributed between the different labels.
    """
    splits = int(len(file_list) / preprocessing_batch_size)
    batched_files = np.array_split(file_list, splits)
    batched_frames = np.array_split(frames_list, splits)
    file_count = 0
    directory = str(target_dir) + "_" + label_string
    all_labels = []

    for i in range(len(batched_files)):
        # load, process and store the current batch training set.
        # Also cut to max-sample size $samples and resample
        audio_batch, labels = load_transform_and_stack(
            batched_files[i],
            batched_frames[i],
            window_size=window_size,
            resample_rate=sample_rate,
            binary_classification=binary_classification,
        )
        all_labels.extend(labels)
        file_count = save_to_disk(audio_batch, directory, file_count)
        print(file_count, label_string, "files processed", flush=True)

    if binary_classification:
        zero_len = 0
        one_len = 0
        print(f"{label_string}")
        for label in all_labels:
            if label.item() == 1:
                one_len += 1
            else:
                zero_len += 1
        if one_len != zero_len:
            print(zero_len, one_len)
            raise ValueError(
                "The datasets are not equally distributed between the labels."
            )

    # save labels
    with open(f"{directory}/labels.npy", "wb") as label_file:
        np.save(label_file, np.array(all_labels))


def get_frames(
    window_size, file_list, max_len, start: int = 0
) -> tuple[list, list, int]:
    """Get list of given file list and frame number for each file.

    Each file gets labeled with a number of windows that can be cut out.

    Returns:
        result_lst (list): List of file names from given file_list that shall be used.
        frame_lst (list): List of number of windows for each file in result_lst.
        last_ind (int): The number of files that were used from given file_list + start.
                        Necessary if method is called multiple times.
    """
    result_lst = []
    frames_lst = []
    length = 0
    for i in range(len(file_list)):
        leng = torchaudio.info(file_list[i]).num_frames
        temp = int(leng - (leng % window_size))
        length += temp
        result_lst.append(file_list[i])
        if length > max_len:
            overset = int((temp - (length - max_len)) / window_size)
            frames_lst.append(overset)
            if frames_lst[-1] == 0:
                frames_lst[-1] = int(temp / window_size)
            break
        frames_lst.append(int(temp / window_size))

    return result_lst, frames_lst, i + start


def pre_process_folder(
    data_folder: str,
    real: Optional[str],
    fake: Optional[str],
    leave_out: Optional[list],
    preprocessing_batch_size: int,
    train_size: float,
    val_size: float,
    test_size: float,
    wavelet: str = "cmor4.6-0.87",
    samples: int = 8_000,
    window_size: int = 200,
    num_of_scales: int = 128,
    sample_rate: int = 8_000,
    f_min: float = 80,
    f_max: float = 4_000,
) -> None:
    """Preprocess a folder containing sub-directories with audios from different sources.

    The sub-directories are expected to indicate to label their source in
    their name. For example,  A - for real and B - for GAN generated audios.

    Args:
        data_folder (str): The folder with the real and gan generated audio folders.
        preprocessing_batch_size (int): The batch_size used for audio conversion.
        train_size (float): Desired size of the train subset of all files in decimal.
        val_size (float): Desired size of the validation subset of all files in decimal.
        test_size (float): Desired size of the test subset of all files in decimal.
        wavelet (str): Wavelet to use in cwt.

    Raises:
        ValueError: If real dir is set but not the fake dir in args.
    """
    nyquist_freq = sample_rate / 2.0  # maximum frequency that can be analyzed
    if f_max >= nyquist_freq:
        f_max = nyquist_freq
    if f_min >= f_max:
        f_min = 80.0

    if train_size + val_size + test_size > 1.0:
        raise ValueError(
            "Training, test and validation size factors should result to 1."
        )

    data_dir = Path(data_folder)
    folder_name = f"{data_dir.name}_{wavelet}_{int(sample_rate)}_{samples}"
    folder_name += (
        f"_{window_size}_{num_of_scales}_{int(f_min)}-{int(f_max)}_1_{train_size}"
    )

    binary_classification = True
    folder_list_all = sorted(data_dir.glob("./*"))

    if leave_out is not None and isinstance(leave_out, list):
        for folder in leave_out:
            if Path(folder) in folder_list_all:
                folder_list_all.remove(Path(folder))

    if real is not None:
        if fake is None:
            folder_list_all.append(Path(real))
            folder_list = folder_list_all
        else:
            # binary classification
            binary_classification = True
            folder_name += f"_{fake.split('_')[-1]}"
            folder_list = [Path(real), Path(fake)]
    else:
        folder_list = folder_list_all

    target_dir = data_dir.parent / folder_name

    train_list, val_list, test_list = split_dataset_random(
        train_size,
        val_size,
        window_size,
        folder_list,
        folder_list_all,
    )  # type: ignore

    print("processing validation set.", flush=True)
    load_process_store(
        val_list[0],
        val_list[1],
        preprocessing_batch_size,
        target_dir,
        "val",
        window_size=window_size,
        sample_rate=sample_rate,
        binary_classification=binary_classification,
    )
    print("validation set stored")

    print("processing test set.", flush=True)
    load_process_store(
        test_list[0],
        test_list[1],
        preprocessing_batch_size,
        target_dir,
        "test",
        window_size=window_size,
        sample_rate=sample_rate,
        binary_classification=False,
    )
    print("test set stored", flush=True)

    print("processing training set.", flush=True)
    load_process_store(
        train_list[0],
        train_list[1],
        preprocessing_batch_size,
        target_dir,
        "train",
        window_size=window_size,
        sample_rate=sample_rate,
        binary_classification=binary_classification,
    )
    print("training set stored", flush=True)

    # compute training normalization.
    # load train data and compute mean and std
    print("computing mean and std values.", flush=True)
    wavelet = get_diff_wavelet(wavelet)

    train_data_set = LearnWavefakeDataset(
        f"{target_dir}_train",
    )
    welford = WelfordEstimator()
    with torch.no_grad():
        for aud_no in range(train_data_set.__len__()):
            welford.update(train_data_set.__getitem__(aud_no)["audio"])
        mean, std = welford.finalize()
    print("mean", mean, "std:", std)
    with open(f"{target_dir}_train/mean_std.pkl", "wb") as f:
        pickle.dump([mean.cpu().numpy(), std.cpu().numpy()], f)


def split_dataset_random(
    train_size,
    val_size,
    window_size,
    folder_list,
    folder_list_all,
):
    """Split dataset in equal sized halfes.

    Multiple labels not implemented yet.
    """
    lengths = []
    sizes = []
    for folder in folder_list_all:
        length = 0
        file_list = list(folder.glob("./*.wav"))
        sizes.append(len(file_list))
        for i in range(len(file_list)):
            if i % 5000 == 0:
                print(i)
            leng = torchaudio.info(file_list[i]).num_frames
            length += leng - (leng % window_size)
        lengths.append(length)
    print("got lengths...")

    # sort in length ascending
    folder_list = [x for _, x in sorted(zip(lengths, folder_list))]

    max_len = min(lengths)  # max length of folder
    max_len -= max_len % window_size
    max_len = int(max_len)

    result_list = []
    frames_list = []

    train_size = int(max_len * train_size)
    train_size -= train_size % window_size
    val_size = int(max_len * val_size)
    val_size -= val_size % window_size
    test_size = max_len - train_size - val_size

    folder_num = len(folder_list) - 1
    if folder_num > 2:
        train_size -= train_size % folder_num
        val_size -= val_size % folder_num
        test_size -= test_size % folder_num

    for folder in folder_list:
        file_list = list(folder.glob("./*.wav"))
        last_ind = 0
        if folder_num + 1 == 2 or get_label_of_folder(folder, True) == 0:
            train_list_f, train_list_w, last_ind = get_frames(
                window_size, file_list, train_size
            )
            val_list_f, val_list_w, last_ind = get_frames(
                window_size, file_list[last_ind + 1 :], val_size, last_ind + 1
            )
            test_list_f, test_list_w, last_ind = get_frames(
                window_size, file_list[last_ind + 1 :], test_size, last_ind + 1
            )
            if folder == folder_list[0]:
                train_size = sum(train_list_w) * window_size
                val_size = sum(val_list_w) * window_size
                test_size = sum(test_list_w) * window_size
        else:
            train_list_f, train_list_w, last_ind = get_frames(
                window_size, file_list, train_size // folder_num
            )
            val_list_f, val_list_w, last_ind = get_frames(
                window_size,
                file_list[last_ind + 1 :],
                val_size // folder_num,
                last_ind + 1,
            )
            test_list_f, test_list_w, last_ind = get_frames(
                window_size,
                file_list[last_ind + 1 :],
                test_size // folder_num,
                last_ind + 1,
            )
            if folder == folder_list[0]:
                train_size = sum(train_list_w) * window_size * folder_num
                val_size = sum(val_list_w) * window_size * folder_num
                test_size = sum(test_list_w) * window_size * folder_num

        result_list.append(
            np.asarray([train_list_f, val_list_f, test_list_f], dtype=object)
        )
        frames_list.append(
            np.asarray([train_list_w, val_list_w, test_list_w], dtype=object)
        )

    files = np.asarray(result_list, dtype=object)
    frames = np.asarray(frames_list, dtype=object)

    train_list_f = [aud for folder in files[:, 0] for aud in folder]  # type: ignore
    val_list_f = [aud for folder in files[:, 1] for aud in folder]  # type: ignore
    test_list_f = [aud for folder in files[:, 2] for aud in folder]  # type: ignore
    train_list_w = [aud for folder in frames[:, 0] for aud in folder]  # type: ignore
    val_list_w = [aud for folder in frames[:, 1] for aud in folder]  # type: ignore
    test_list_w = [aud for folder in frames[:, 2] for aud in folder]  # type: ignore

    np.random.seed(42)
    train_list_f, train_list_w = shuffle_random(
        np.array(train_list_f), np.array(train_list_w)
    )
    val_list_f, val_list_w = shuffle_random(np.array(val_list_f), np.array(val_list_w))
    test_list_f, test_list_w = shuffle_random(
        np.array(test_list_f), np.array(test_list_w)
    )

    return (
        np.array([train_list_f, train_list_w]),
        np.array([val_list_f, val_list_w]),
        np.array([test_list_f, test_list_w]),
    )


def arange_datasets(train_list, val_list, test_list, window_size):
    """Arange the picked labels and audios in equally sized frames."""
    train = np.repeat(train_list[0], np.asarray(train_list[1], dtype=int))
    val = np.repeat(val_list[0], np.asarray(val_list[1], dtype=int))
    test = np.repeat(test_list[0], np.asarray(test_list[1], dtype=int))

    train_f = get_frames_array(train_list[1], window_size)
    val_f = get_frames_array(val_list[1], window_size)
    test_f = get_frames_array(test_list[1], window_size)

    return np.array([train, train_f]), np.array([val, val_f]), np.array([test, test_f])


def get_all_labels(list) -> np.ndarray:
    """Arange labels from list according to audio path."""
    labels = []
    for i in range(len(list[0])):
        labels.append(int(get_label(list[0, i], True)))
    return np.repeat(np.array(labels, dtype=int), np.asarray(list[1], dtype=int))


def get_frames_array(list, window_size) -> np.ndarray:
    """Create array with starting indices of audios for given number of frames per audio."""
    result = np.empty(0, dtype=int)
    for i in range(len(list)):
        temp = np.arange(0, list[i] * window_size, window_size, dtype=int)
        result = np.concatenate((result, temp))
    return result


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
        "--directory",
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
        "--leave-out",
        nargs="+",
        default=[],
        type=str,
        help="Wich gans to ignore in folder.",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.7,
        help="Desired size of the training subset of all files. (default: 0.7).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Desired size of the testing subset of all files. (default: 0.2).",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Desired size of the validation subset of all files. (default: 0.1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="The batch_size used for audio conversion. (default: 2048).",
    )
    parser.add_argument(
        "--wavelet",
        type=str,
        default="cmor4.6-0.87",
        help="The wavelet to use. Choose one from pywt.wavelist(). Defaults to cmor4.6-0.87.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=11025,
        help="Size of window of audio file as number of samples relative to initial sample rate. Default: 11025.",
    )
    parser.add_argument(
        "--scales",
        type=int,
        default=224,
        help="Number of scales for the cwt. Default: 224.",
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
        default=80,
        help="Minimum frequency to be analyzed in Hz. Default: 80.",
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
        leave_out=args.leave_out,
        preprocessing_batch_size=args.batch_size,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        wavelet=args.wavelet,
        num_of_scales=args.scales,
        window_size=args.window_size,
        sample_rate=args.sample_rate,
        f_min=args.f_min,
        f_max=args.f_max,
    )
