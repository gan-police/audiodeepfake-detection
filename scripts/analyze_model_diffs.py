"""Analyze which audios are being predicted correctly from which model."""

import numpy as np
import os
import torchaudio


def cut_and_save_wav(input_path, output_path, offset, duration):
    waveform, sample_rate = torchaudio.load(
        input_path,
        frame_offset=offset * duration,
        num_frames=duration,
    )

    torchaudio.save(output_path, waveform, sample_rate)


if __name__ == "__main__":
    file_base = ""
    file_comp = ""
    save_path = ""

    if not os.path.exists(file_base) or not os.path.exists(file_comp):
        raise RuntimeError("Files not found")

    results_base = np.load(file_base, allow_pickle=True).item()
    results_comp = np.load(file_comp, allow_pickle=True).item()

    if not "unknown" in results_base or not "unknown" in results_comp:
        raise RuntimeError("Missing key unknown")

    diff_ids_unknown = np.setdiff1d(results_base["unknown"], results_comp["unknown"])

    data_only_in_first = results_base["dataset"][diff_ids_unknown]

    # generate 10 sample audios
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    i = 0
    for data in data_only_in_first:
        print(i)
        file_name = data[0]
        if "A_ljspeech" in str(file_name) or "BASIC" in str(file_name):
            continue
        else:
            i += 1

        # assuming a file is saved as some_folder/B_melgan/L2934-234_gen.wav or similar
        split_name = str(file_name).split('/')
        new_name = f"{split_name[-2]}_{split_name[-1].split('.')[0]}_{i}.wav"
        print(f"Saved {str(file_name)} in {new_name}")
        cut_and_save_wav(
            file_name,
            f"{save_path}/{new_name}",
            data[1],
            data[2],
        )

        if i == 10:
            break
