# Data preparation for occupant level estimation
import pandas as pd
import os
import torchaudio
import torch
from src.preprocessing.speech_synthesis import mix_speech
import torchaudio.functional as F
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


# read the RIR manifest
def synth_mixed_speech(
    rir_manifest: pd.DataFrame, speech_manifest: pd.DataFrame, num_occ
):
    rir_path = os.path.join(
        "./data/RIR_aggregated/RIR.data",
        rir_manifest["RIR"],
    )
    rir, fs = torchaudio.load(rir_path)
    mixed_speech, mixed_speech_label, file_info = mix_speech(num_occ, speech_manifest)
    rir = rir / rir.max(dim=-1)[0]
    reverb_mixed_speech = F.fftconvolve(mixed_speech, rir, mode="full")
    len_mixed_speech = mixed_speech.numel()
    reverb_mixed_speech = reverb_mixed_speech[:, 0:len_mixed_speech]
    return (
        reverb_mixed_speech,
        mixed_speech,
        mixed_speech_label,
        file_info,
    )


rir_manifest_ID_resampled = pd.read_csv(
    "./data/RIR_aggregated/RIR.metadata/RIRLabelAugmentV2.csv"
)
librispeech_info = pd.read_csv("./data/LibriSpeech/LibriSpeech_label.csv")
num_occ = pd.read_csv("./data/RIR_aggregated/RIR.metadata/num_occ.csv")


reverb_mixed_speech_manifest = []
mixed_speech_label_manifest = []
mixed_speech_manifest = []


def mixed_speech(i):
    (
        reverb_mixed_speech,
        mixed_speech,
        mixed_speech_label,
        file_info,
    ) = synth_mixed_speech(
        rir_manifest_ID_resampled.iloc[i], librispeech_info, num_occ.iloc[i]
    )

    rir_info = rir_manifest_ID_resampled["RIR"][i]
    numOcc = num_occ["num_occ"][i]
    mixed_speech_savepath = "./data/mixed_speech/mixed_speech.data"

    mixed_speech_info_savepath = "./data/mixed_speech/mixed_speech.metadata"

    if not os.path.exists(os.path.join(mixed_speech_savepath, "reverb_mixed")):
        os.makedirs(os.path.join(mixed_speech_savepath, "reverb_mixed"))
    if not os.path.exists(os.path.join(mixed_speech_savepath, "clean_mixed")):
        os.makedirs(os.path.join(mixed_speech_savepath, "clean_mixed"))
    if not os.path.exists(mixed_speech_info_savepath):
        os.makedirs(mixed_speech_info_savepath)

    with open(
        os.path.join(
            mixed_speech_info_savepath,
            f"reverb_mixed_speech_No{str(i)}_info.txt",
        ),
        "a",
    ) as f:
        for base_name, start_idx, end_idx, dist_speech in file_info:
            print(
                ",".join([base_name, str(start_idx), str(end_idx), str(dist_speech)]),
                file=f,
            )

    torchaudio.save(
        os.path.join(
            mixed_speech_savepath,
            "reverb_mixed",
            f"reverb_mixed_speech_No{str(i)}.wav",
        ),
        reverb_mixed_speech,
        sample_rate=16000,
        bits_per_sample=16,
        encoding="PCM_S",
    )

    torchaudio.save(
        os.path.join(
            mixed_speech_savepath, "clean_mixed", f"mixed_speech_No{str(i)}.wav"
        ),
        mixed_speech,
        sample_rate=16000,
        bits_per_sample=16,
        encoding="PCM_S",
    )

    mixed_speech_label_savepath = "./data/mixed_speech/mixed_speech_label.data"
    if not os.path.exists(mixed_speech_label_savepath):
        os.makedirs(mixed_speech_label_savepath)
    mixed_speech_label = torch.save(
        mixed_speech_label,
        os.path.join(mixed_speech_label_savepath, "mixed_speech_No" + str(i) + ".pt"),
    )

    reverb_mixed_speech_ = [
        "mixed_speech_No" + str(i) + ".wav",
        "mixed_speech_No" + str(i) + ".pt",
        numOcc,
        rir_info,
    ]

    mixed_speech_ = [
        "mixed_speech_No" + str(i) + ".wav",
        "mixed_speech_No" + str(i) + ".pt",
        numOcc,
    ]

    return reverb_mixed_speech_, mixed_speech_


with ThreadPoolExecutor(max_workers=os.cpu_count() - 12) as executor:
    threads = []
    for i in range(len(rir_manifest_ID_resampled)):
        threads.append(executor.submit(mixed_speech, i))

    for t in tqdm(threads):
        reverb_mixed_speech_manifest.append(t.result()[0])
        mixed_speech_manifest.append(t.result()[1])

    reverb_mixed_speech_manifest = pd.DataFrame(
        reverb_mixed_speech_manifest,
        columns=["mixed_speech", "mixed_speech_label", "numOcc", "rir_info"],
    )
    mixed_speech_manifest = pd.DataFrame(
        mixed_speech_manifest, columns=["mixed_speech", "mixed_speech_label", "numOcc"]
    )

    reverb_mixed_speech_manifest.to_csv(
        "./data/mixed_speech/reverb_mixed_speech_manifest.csv",
        index=False,
    )
    mixed_speech_manifest.to_csv(
        "./data/mixed_speech/mixed_speech_manifest.csv",
        index=False,
    )
