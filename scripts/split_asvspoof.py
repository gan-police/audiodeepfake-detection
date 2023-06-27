"""Split ASVSpoof 2021 DeepFake Eval-Data into fake and real audios."""
import glob
import os
import shutil

import numpy as np
import pandas

path = "/home/s6kogase/data/asvspoof/ASVspoof2021_DF_eval"
audio_path = "flac"
label_path = "keys/DF/CM/trial_metadata.txt"


if __name__ == "__main__":
    p_names = [
        "speaker",
        "index_col",
        "compr",
        "source",
        "attack",
        "label",
        "trim",
        "subset",
        "vocoder",
        "task",
        "team",
        "gender-pair",
        "language",
    ]

    pd_protocol = pandas.read_csv(
        f"{path}/{label_path}", sep=" ", names=p_names, skipinitialspace=True
    )

    spoof_keys = pd_protocol["label"][pd_protocol["label"] == "spoof"].keys()
    bonaf_keys = pd_protocol["label"][pd_protocol["label"] == "bonafide"].keys()

    spoof_names = np.asarray(pd_protocol["index_col"][spoof_keys])
    bonaf_names = np.asarray(pd_protocol["index_col"][bonaf_keys])

    real_target_path = f"{path}/audios/real"
    fake_target_path = f"{path}/audios/fake"
    os.makedirs(real_target_path, exist_ok=True)
    os.makedirs(fake_target_path, exist_ok=True)

    file_list = glob.glob(f"{path}/{audio_path}/*.flac")

    for file in file_list:
        name = file.split("/")[-1].split(".")[0]
        if name in spoof_names:
            shutil.copy(file, fake_target_path)
        elif name in bonaf_names:
            shutil.copy(file, real_target_path)
        else:
            print(f"The file with id {name} does not exist in the labels file.")

    print("Done.")
