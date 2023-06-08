"""Clean up train, test and validation sets to be exactly equally distributed in all labels."""
import os

import numpy as np

from src.data_loader import LearnWavefakeDataset

path = "/home/s6kogase/data/run5"  # Replace with the path to the directory you want to count the files for

for end in ["_val", "_test", "_train"]:
    len_list = []
    dir_list = []
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if (
            os.path.isdir(folder_path)
            and folder.endswith(end)
            and "all" not in folder_path
        ):
            print(f"Counting {folder}")
            dataset = LearnWavefakeDataset(folder_path)
            len_list.append(dataset.labels[dataset.labels != 0].shape[0])
            len_list.append(dataset.labels[dataset.labels == 0].shape[0])
            dir_list.append(folder_path)

    minimum = min(len_list)
    print(f"minimum {minimum}")
    for i in range(0, len(len_list), 2):
        if len_list[i] > minimum or len_list[i + 1] > minimum:
            del_list = []
            print(f"processing {dir_list[i // 2]}")
            dataset = LearnWavefakeDataset(dir_list[i // 2])

            del_list.extend(
                dataset.audios[dataset.labels != 0][: len_list[i] - minimum].tolist()
            )
            del_list.extend(
                dataset.audios[dataset.labels == 0][
                    : len_list[i + 1] - minimum
                ].tolist()
            )

            exclude = []
            for file in del_list:
                exclude.append(np.where(dataset.audios == file)[0][0])
            all_labels = np.delete(dataset.labels, exclude)

            with open(f"{dir_list[i // 2]}/labels.npy", "wb") as label_file:
                np.save(label_file, np.array(all_labels))

            for file_path in del_list:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"{file_path} deleted")
                else:
                    print(f"{file_path} not found")
