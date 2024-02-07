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
    if not os.path.exists(file_base) or not os.path.exisits(file_comp):
        raise RuntimeError("Files not found")

    results_base = np.load(file_base)
    results_comp = np.load(file_comp)

    if not "unknown" in results_base or not "unknown" in results_comp:
        raise RuntimeError("Missing key unknown")

    diff_ids_unknown = np.setdiff1d(results_base["unknown"], results_comp["unknown"])

    data_only_in_first = results_base[diff_ids_unknown]

    for data in data_only_in_first:
        print(data)

    # generate 10 sample audios
    save_path = ""

    for i in range(10):
        cut_and_save_wav(
            data_only_in_first[i, 0],
            f"{save_path}/test_{i}.wav",
            data_only_in_first[i, 1],
            data_only_in_first[i, 2],
        )
