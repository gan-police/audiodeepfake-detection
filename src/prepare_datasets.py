"""Prepare all files of on GAN architecture and the real audio dataset.

The resulting files are resampled but not transformed yet to make
gradient flow through wavelets possible.
"""
import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torchaudio

from .train_classifier import set_seed


def shuffle_random(a, b) -> tuple[list, list]:
    """Shuffle two arrays randomly in the same order."""
    c: np.ndarray = np.c_[a.reshape(len(a), -1), b.reshape(len(b), -1)]
    a2: np.ndarray = c[:, : a.size // len(a)].reshape(a.shape)
    b2: np.ndarray = c[:, a.size // len(a) :].reshape(b.shape)
    np.random.shuffle(c)

    return a2.tolist(), b2.tolist()


def save_to_disk(
    data_batch: np.ndarray,
    directory: str,
    previous_file_count: int = 0,
    dir_suffix: str = "",
) -> int:
    """Save audios to disk using their position on the dataset as filename.

    Args:
        data_batch (np.ndarray): The audio batch to store.
        directory (str): The place to store the audios at.
        previous_file_count (int): The number of previously stored audios. Defaults to 0.
        dir_suffix (str): A comment which is attatched to the output directory.

    Returns:
        int: The new total of storage audios.
    """
    # loop over the batch dimension
    if not os.path.exists(f"{directory}{dir_suffix}"):
        print("creating", f"{directory}{dir_suffix}", flush=True)
        os.mkdir(f"{directory}{dir_suffix}")
    file_count = previous_file_count
    for pre_processed_audio in data_batch:
        with open(f"{directory}{dir_suffix}/{file_count:06}.npy", "wb") as numpy_file:
            np.save(numpy_file, pre_processed_audio)
        file_count += 1

    return file_count


def get_label_of_folder(
    path_of_folder: Path, binary_classification: bool = False
) -> int:
    """Get the label of the audios in a folder based on the folder path.

        We assume:
            A: Orignal data, B: First gan,
            C: Second gan, D-H: more gans.
        A working folder structure could look like:
            A_ljspeech B_melgan C_hifigan D_mbmelgan ...
        With each folder containing the audios from the corresponding
        source.

    Args:
        path_of_folder (Path):  Path string containing only a single underscore directly
                                after the label letter.
        binary_classification (bool):   If flag is set, we only classify binarily, i.e.
                                        whether an audio is real or fake. In this case,
                                        the prefix 'A' indicates real, which is encoded
                                        with the label 0. All other folders are considered
                                        fake data, encoded with the label 1.

    Returns:
        int: The label encoded as integer.
    """
    label_str = path_of_folder.name.split("_")[0]
    if binary_classification:
        # differentiate original and generated data
        if label_str == "A":
            return 0
        else:
            return 1
    else:
        # the the label based on the path, As are 0s, Bs are 1, etc.
        label = ord(label_str) - 65
        return label


def get_label(path_to_audio: Path, binary_classification: bool) -> int:
    """Get the label based on the audio path.

       We assume:
            A: Orignal data, B: First gan,
            C: Second gan, D-H: more gans.
        A working folder structure could look like:
            A_ljspeech B_melgan C_hifigan D_mbmelgan ...
       With each folder containing the audios from the corresponding source.

    Args:
        path_to_audio (Path):   Audio path string containing only a single underscore
                                directly after the label letter.
        binary_classification (bool):   If flag is set, we only classify binarily,
                                        i.e. whether an audio is real or fake. In this case,
                                        the prefix 'A' indicates real, which is encoded
                                        with the label 0. All other folders are considered
                                        fake data, encoded with the label 1.

    Returns:
        int: The label encoded as integer.
    """
    return get_label_of_folder(path_to_audio.parent, binary_classification)


def load_transform_and_stack(
    path_list: np.ndarray,
    frame_list: np.ndarray,
    window_size: int,
    resample_rate: int,
    binary_classification: bool = False,
) -> tuple[np.ndarray, list]:
    """Transform a lists of paths into a batches of numpy arrays and record their labels.

    Args:
        path_list (np.ndarray): An array of Poxis path objects.
        frame_list (np.ndarray): Array of frames that can be cut from audio at path in
                                 path_list at the same index.
        window_size (int): Size of desired output tensor for each training sample before
                           audio is resampled.
        resample_rate (int): Desired sample rate of audio after resampling it, in Hz.
        binary_classification (bool): If flag is set, we only classify binarily, i.e.
                                      whether an audio is real or fake.

    Returns:
        tuple (np.ndarray, list): A numpy array of size
            (preprocessing_batch_size * (samples / window_size), number of channels, window_size)
            and a label list of length preprocessing_batch_size.
    """
    audio_list: list[np.ndarray] = []
    label_list = []

    old_win_size = window_size
    window_size *= resample_rate / torchaudio.info(path_list[0]).sample_rate
    window_size = int(window_size)

    for i in range(len(path_list)):
        if frame_list[i] > 0:
            audio, sample_rate = torchaudio.load(
                path_list[i],
                normalize=True,
                num_frames=int(old_win_size * frame_list[i]),
            )
            # resample audio
            audio_res = torchaudio.functional.resample(
                audio, sample_rate, resample_rate, resampling_method="kaiser_window"
            )
            # cut to non-overlapping equal-sized windows
            framed_audio = audio_res[0].unfold(0, window_size, window_size)

            framed_audio = framed_audio.unsqueeze(1)
            audio_list.extend(np.array(framed_audio))
            label = np.array(get_label(path_list[i], binary_classification))
            label_list.extend([label] * framed_audio.shape[0])
        else:
            print(f"skipping: {path_list[i]}")
    return np.stack(audio_list), label_list


def load_process_store(
    file_list: np.ndarray,
    frames_list: np.ndarray,
    preprocessing_batch_size: int,
    target_dir: Path,
    label_string: str,
    window_size: int,
    sample_rate: int,
    binary_classification: bool = False,
) -> None:
    """Load, process and store a file list according to a processing function.

    Args:
        file_list (np.ndarray): PosixPath objects leading to source audios.
        frames_list (np.ndarray): Number of windows of size window_size that shall be
                                  cut from files in file_list at corresponding indices.
        preprocessing_batch_size (int): The number of files processed at once.
        target_dir (Path): A directory where to save the processed files.
        label_string (str): A label that is added to the target folder.
        window_size (int): Size of windows the audios will be cut to.
        sample_rate (int): Desired sample rate for audios that will be used to
                           downsample all audios.
        binary_classification (bool): If flag is set, we only classify binarily, i.e.
                                      whether an audio is real or fake.

    Raises:
        ValueError: If datasets are not distributed between the different labels.
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

    # check if ds is equal sized in labels
    if binary_classification:
        zero_len = 0
        one_len = 0
        print(f"{label_string}")
        for label in all_labels:
            if label.item() >= 1:
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
    window_size: int,
    file_list: list[Path],
    max_len: int,
    start: int = 0,
) -> tuple[list[Path], list[int], int]:
    """Get list of given files and frame count for each file.

    Each file in result_lst corresponds to a number of windows that can be cut out in frames_lst.

    Args:
        window_size (int): Size of windows the audios will be cut to.
        file_list (list): List of paths to audio files.
        max_len (int): Maximum number of samples in total for all audios in file_list.
        start (int): Start index for file_list. All files with a lower index will not be
                     taken into account.

    Returns:
        result_lst (list): List of file names from given file_list.
        frame_lst (list): List of corresponding number of windows for each file in result_lst.
        last_ind (int): The number of files that were used from given file_list + start.
                        Necessary if method is called multiple times, so that audio files are not
                        used twice.
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


def split_dataset_random(
    train_size: float,
    val_size: float,
    window_size: int,
    folder_list: list[Path],
    folder_list_all: list[Path],
    max_len: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get shuffled dataset windows and paths with equally distributed labels.

    To use as many samples from a sub-dataset as possible, set max_len to None and
    folder_list_all equal to folder_list.

    Args:
        train_size (float): Desired size of train set in decimal.
        val_size (float): Desired size of validation set in decimal.
        window_size (int): Size of windows the audios will be cut to in samples.
        folder_list (list): List of paths that will be taken into account.
        folder_list_all (list): List of paths of all generated audio folders that might be
                                used as comparison. It is used to make all datasets for
                                different gans the same size in samples, as the minimal
                                available sample number of a sub-dataset will be used for
                                all of the other training sets. This argument is only necessary
                                if max_len is not set manually.
        max_len (int): Maximum number of samples taken from each sub-datset (real or fake).
                       Can be used to make all datasets the same size. Default: None.

    Raises:
        ValueError: If file list is empty (e.g. due to a dir that is not existing).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of training, validation and test arrays
                                                   with file paths at index 0 and window count at
                                                   index 1.

    # noqa: DAR401
    """
    if max_len is None:
        lengths = []
        sizes = []
        for folder in folder_list_all:
            print("Counting ", folder)
            length = 0
            file_list = list(folder.glob("./*.wav"))
            sizes.append(len(file_list))
            for i in range(len(file_list)):
                leng = torchaudio.info(file_list[i]).num_frames
                length += leng - (leng % window_size)
                if i % 10000 == 0:
                    print(i)
            lengths.append(length)

        single_lengths = []
        if folder_list != folder_list_all:
            print("Counting ", folder)
            for folder in folder_list:
                length = 0
                file_list = list(folder.glob("./*.wav"))
                for i in range(len(file_list)):
                    leng = torchaudio.info(file_list[i]).num_frames
                    length += leng - (leng % window_size)
                    if i % 10000 == 0 and i > 0:
                        print(i)
                single_lengths.append(length)
        else:
            single_lengths = lengths

        print("got lengths...")

        # sort in length ascending
        folder_list = [x for _, x in sorted(zip(single_lengths, folder_list))]

        max_len = min(lengths)  # max length of folder
    max_len -= max_len % window_size
    max_len = int(max_len)
    print(max_len)

    result_list = []
    frames_list = []

    train_size = int(max_len * train_size)
    train_size -= train_size % window_size
    val_size = int(max_len * val_size)
    val_size -= val_size % window_size
    test_size = max_len - train_size - val_size

    folder_num = len(folder_list) - 1

    # set this to false if ds should contain 50 % real and 50 % fake
    # otherwise all folders will be equally distributed
    equal_distr = False

    if folder_num > 2:
        train_size -= train_size % folder_num
        val_size -= val_size % folder_num
        test_size -= test_size % folder_num

    if not equal_distr:
        # insert real folder at last position
        for folder in folder_list:
            if get_label_of_folder(folder, True) == 0:
                folder_list.remove(folder)
                folder_list.append(folder)
                break

    print(f"using folders: {folder_list}", flush=True)

    for folder in folder_list:
        print(f"splitting folder {folder}", flush=True)
        file_list = list(folder.glob("./*.wav"))
        if len(file_list) == 0:
            raise ValueError("File list does not contain any files.")
        last_ind = 0

        # if folder holds real data or only one gan is taken into account
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

        if folder_num + 1 > 2 and (get_label_of_folder(folder, True) != 0 or equal_distr):
            print(f"dataset will contain only {50 / folder_num if not equal_distr else 100 / folder_list} % of this folder...")
            train_list_f, train_list_w, _ = get_frames(
                window_size, train_list_f, train_size // folder_num
            )
            val_list_f, val_list_w, _ = get_frames(
                window_size, val_list_f, val_size // folder_num
            )
            test_list_f, test_list_w, _ = get_frames(
                window_size, test_list_f, test_size // folder_num
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


def pre_process_folder(
    data_folder: str,
    preprocessing_batch_size: int,
    real: Optional[str],
    fake: Optional[str],
    leave_out: Optional[list],
    max_samples: Optional[int] = None,
    train_size: float = 0.7,
    val_size: float = 0.1,
    test_size: float = 0.2,
    window_size: int = 11_025,
    sample_rate: int = 22_050,
    binary_classification: bool = False,
) -> None:
    """Preprocess a folder containing sub-directories with audios from different sources.

    The sub-directories are expected to indicate to label their source in
    their name. For example,  A - for real and B-H - for GAN generated audios.

    Args:
        data_folder (str): The folder with the real and gan generated audio folders.
        preprocessing_batch_size (int): The batch_size used for audio conversion.
        real (str, optional): The folder with real generated audio folders. If set
                              data_folder will be interpreted as parent folder for gan
                              generated audio folders. Use this parameter if real and fake
                              audio folders are seperated.
        fake (str, optional): The folder with the gan generated audios. If set, data_folder
                              will be ignored, but real must be set (otherwise it will be
                              ignored).
        train_size (float): Desired size of the train subset of all files in decimal Default: 0.7.
        val_size (float): Desired size of the validation subset of all files in decimal. Default: 0.1.
        test_size (float): Desired size of the test subset of all files in decimal. Default: 0.2.
        window_size (int): Size of windows the audios will be cut to. Default: 11_025.
        sample_rate (int): Desired sample rate for audios that will be used to downsample all audios.
        binary_classification (bool): If true consider this problem as a true or fake binary problem.

    Raises:
        ValueError: Raised if train_size, val_size and test_size don't add up to 1 or if directories
                    are not set properly.
    """
    set_seed(42)

    if train_size + val_size + test_size > 1.0:
        raise ValueError(
            "Training, test and validation size factors should result to 1."
        )

    data_dir = Path(data_folder)
    folder_name = f"{data_dir.name}_{int(sample_rate)}_{window_size}_{train_size}"

    folder_list_all = sorted(data_dir.glob("./*"))

    if leave_out is not None and isinstance(leave_out, list):
        for folder in leave_out:
            if Path(folder) in folder_list_all:
                folder_list_all.remove(Path(folder))
                folder_name += f"_x{folder.split('_')[-1]}"

    if real is not None:
        if fake is None:
            folder_list_all.append(Path(real))
            folder_list = folder_list_all
            folder_name += "_all"
        else:
            folder_name += f"_{fake.split('_')[-1]}"
            folder_list = [Path(real), Path(fake)]
            folder_list_all.append(Path(real))
    else:
        if len(folder_list_all) == 0:
            raise ValueError("Either directory and/or realdir must be set.")
        folder_list = folder_list_all

    if len(folder_list_all) <= 1:
        print("Warning: training will contain one or less labels.")
    target_dir = data_dir.parent / folder_name

    train_list, val_list, test_list = split_dataset_random(
        train_size=train_size,
        val_size=val_size,
        window_size=window_size,
        folder_list=folder_list,
        folder_list_all=folder_list_all,
        max_len=max_samples,
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
        binary_classification=binary_classification,
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--directory",
        type=str,
        default="./data/fake",
        help="The folder with the gan generated and/or the real audio folders.",
    )
    parser.add_argument(
        "--realdir",
        type=str,
        help="The folder with the real audios. If set data_folder will be interpreted"
        " as parent folder for gan generated audio folders. Use this parameter if real"
        " and fake audio folders are seperated.",
    )
    parser.add_argument(
        "--fakedir",
        type=str,
        help="The folder with the gan generated audios. If specified the directory"
        " argument will be ignored, but --realdir must be set.",
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
        default=512,
        help="The batch_size used for audio conversion. (default: 2048).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=11025,
        help="Size of window of audio file as number of samples relative to initial"
        " sample rate. Default: 11025.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22_050,
        help="Desired sample rate of audio in Hz. Default: 8_000.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples taken from a dataset. Only use values below"
        " or equal to the maximum number of samples available.",
    )
    parser.add_argument(
        "--binary",
        type=bool,
        help="Turns the problem into a fake or real binary classification problem.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)

    pre_process_folder(
        data_folder=args.directory,
        preprocessing_batch_size=args.batch_size,
        real=args.realdir,
        fake=args.fakedir,
        leave_out=args.leave_out,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        window_size=args.window_size,
        sample_rate=args.sample_rate,
        max_samples=args.max_samples,
        binary_classification=args.binary,
    )
