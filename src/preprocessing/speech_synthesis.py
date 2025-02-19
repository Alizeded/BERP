import os
import random

import numpy as np
import pandas as pd
import torch
import torchaudio

from src.preprocessing.occu_dist_dist import occu_gmm

seed = 2036
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# label_root_path = "/home/s2320016/workspace/acoustic/data/LibriSpeech/mask"
label_root_path = os.path.join(os.getcwd(), "data/LibriSpeech/mask")

shape = 6.1802
scale = 18.667


def speech_synthesis(num_occ: int, files: pd.DataFrame, max_seq_len: int):
    zeros_seq = torch.zeros(1, max_seq_len)
    mixed_speech = 0.0
    mixed_speech_label = 0.0
    start = max_seq_len
    tail = 0

    file_info = []
    for _ in range(num_occ):
        file_idx = random.randint(0, len(files) - 1)
        audio, fs = torchaudio.load(files.iloc[file_idx]["filename"])
        base_name = files.iloc[file_idx]["filename"].split("/")[-1].split(".")[0]
        label_path = os.path.join(
            label_root_path,
            base_name + ".pt",
        )
        label = torch.load(label_path, weights_only=True)
        start_idx = torch.randint(0, max_seq_len - audio.shape[-1], (1,)).item()
        start = start_idx if start_idx < start else start
        padded_audio = torch.cat(
            (
                zeros_seq[:, :start_idx],
                audio,
                zeros_seq[:, start_idx + audio.shape[-1] : max_seq_len],
            ),
            dim=-1,
        )
        assert padded_audio.shape[-1] == max_seq_len, "Padding error"

        padded_label = torch.cat(
            (
                zeros_seq[:, :start_idx],
                label,
                zeros_seq[:, start_idx + audio.shape[-1] : max_seq_len],
            ),
            dim=-1,
        )

        end_idx = start_idx + audio.shape[-1]
        tail = end_idx if end_idx > tail else tail
        dist_speech = np.random.gamma(shape, scale)
        dist_speech = torch.tensor(dist_speech / 100).abs()  # convert to meter
        if dist_speech < 1:
            dist_speech = dist_speech + 1
        elif dist_speech > 6:
            dist_speech = torch.tensor(6.0)
        dist_attenuation = 1 / dist_speech

        mixed_speech += dist_attenuation * padded_audio
        # padded_label = masking_generator(padded_audio, fs)
        mixed_speech_label += padded_label

        file_info.append([base_name, start_idx, end_idx, float(dist_speech)])

    file_info = [
        [base_name, start_idx - start, end_idx - start, dist_speech]
        for base_name, start_idx, end_idx, dist_speech in file_info
    ]
    mixed_speech = mixed_speech[:, start:tail]
    mixed_speech_label = mixed_speech_label[:, start:tail]

    return mixed_speech, mixed_speech_label, file_info


def mix_speech(num_occ: pd.DataFrame, files: pd.DataFrame):
    rir_volume = num_occ["rir_volume"]
    num_occ = int(num_occ["num_occ"])
    if num_occ != 0:
        if rir_volume < 400:
            max_seq_len = 10 * 16000  # 10s
            files = files[files["length"] < max_seq_len]
            mixed_speech, mixed_speech_label, file_info = speech_synthesis(
                num_occ, files, max_seq_len
            )

        elif rir_volume >= 400 and rir_volume < 4000:
            max_seq_len = 20 * 16000  # 15s
            files = files[files["length"] < max_seq_len]
            mixed_speech, mixed_speech_label, file_info = speech_synthesis(
                num_occ, files, max_seq_len
            )

        elif rir_volume >= 4000:
            max_seq_len = 25 * 16000  # 20s
            files = files[files["length"] < max_seq_len]
            mixed_speech, mixed_speech_label, file_info = speech_synthesis(
                num_occ, files, max_seq_len
            )
    else:
        mixed_speech = torch.zeros(1, 10 * 16000)  # 10s
        mixed_speech_label = torch.zeros(1, 10 * 16000)
        file_info = [0, 0, 0, 0]

    mixed_speech_label = mixed_speech_label.to(dtype=torch.int8)

    return mixed_speech, mixed_speech_label, file_info
