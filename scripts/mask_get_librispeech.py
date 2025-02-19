# Data preparation for occupant level estimation
import os

import pandas as pd
import rootutils
import torch
import torchaudio
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Data preparation for occupant level estimation
vad_model, vad_utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=True,
    trust_repo=True,
)

(get_speech_timestamps, _, _, _, collect_chunks) = vad_utils


def masking_generator(audio: torch.Tensor, fs: int):
    # Load VAD model
    speech_timestamps = get_speech_timestamps(audio, vad_model, sampling_rate=fs)
    zeros_seq = torch.zeros((1, audio.shape[-1]))
    for i in range(len(speech_timestamps)):
        masked = torch.ones(
            1, speech_timestamps[i]["end"] - speech_timestamps[i]["start"]
        )
        padded_masked = torch.cat(
            (
                torch.zeros(1, speech_timestamps[i]["start"]),
                masked,
                torch.zeros(1, audio.shape[-1] - speech_timestamps[i]["end"]),
            ),
            dim=-1,
        )
        zeros_seq += padded_masked
    # zeros_seq[zeros_seq >= 1] = 1
    return zeros_seq


librispeech_info = pd.read_csv("./data/LibriSpeech/LibriSpeech_label.csv")

speech_timestamps_all = []
for i in tqdm(range(len(librispeech_info))):
    audio, fs = torchaudio.load(librispeech_info["filename"][i])
    filename_info = librispeech_info["filename"][i].split("/")[-1].split(".")[0]
    masked = masking_generator(audio, fs)
    masked = masked.to(dtype=torch.int8)
    after_vad_path = "./data/LibriSpeech/mask"
    if not os.path.exists(after_vad_path):
        os.makedirs(after_vad_path)

    torch.save(
        masked,
        os.path.join(
            after_vad_path,
            filename_info + ".pt",
        ),
    )
