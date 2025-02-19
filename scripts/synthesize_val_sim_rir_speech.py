import ast
import os
import random

import numpy as np
import pandas as pd
import pyroomacoustics as pra
import rootutils
import torch
import torch.nn.functional as f
import torchaudio
import torchaudio.functional as F
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.preprocessing.RIRutils import RIRestThTt  # noqa: E402

torch.manual_seed(3046)
random.seed(3046)

speech_long_manifest = pd.read_csv("./data/LibriSpeech/audio_long_manifest_local.csv")
val_sim_rir_manifest = pd.read_csv(
    "/home/lucianius/workspace/BERP/data/noiseReverbSpeech/val_manifest_RIR_sim.csv"
)

val_speech_long_manifest = speech_long_manifest.sample(
    n=len(val_sim_rir_manifest), random_state=3046
)

# ------------------- read the noise -------------------
noisepath_label = "./data/noise_dataset/noise.metadata/noise_label.csv"
noise_label = pd.read_csv(noisepath_label)
noisepath_data = "./data/noise_dataset/noise.data"
noise_database = []
for i in range(0, len(noise_label["filename"])):
    noise, fs = torchaudio.load(
        os.path.join(noisepath_data, noise_label["filename"][i])
    )
    noise_database.append(noise)
# shuffle the noise
random.shuffle(noise_database)


def calculate_angles_arctan(src_pos, mic_pos):
    """
    Calculate azimuth and elevation angles using arctan.

    Args:
        src_pos: Source position [x, y, z]
        mic_pos: Microphone position [x, y, z]

    Returns:
        azimuth: Angle in horizontal plane (degrees)
        elevation: Angle from horizontal plane (degrees)
    """
    # Convert positions to numpy arrays
    src = np.array(src_pos)
    mic = np.array(mic_pos)

    # Calculate vector from source to microphone
    vector = mic - src

    # Calculate distance in horizontal plane
    horizontal_dist = np.sqrt(vector[0] ** 2 + vector[1] ** 2)

    # Calculate azimuth using arctan
    # Need to handle quadrants manually
    if vector[0] != 0:  # Avoid division by zero
        azimuth = np.arctan(vector[1] / vector[0])
    else:  # Special case when x = 0
        if vector[1] > 0:
            azimuth = np.pi / 2
        elif vector[1] < 0:
            azimuth = 3 * np.pi / 2
        else:
            azimuth = 0

    # Calculate elevation using arctan
    if horizontal_dist != 0:  # Avoid division by zero
        elevation = np.arctan(vector[2] / horizontal_dist)
    else:
        elevation = np.pi / 2 if vector[2] > 0 else -np.pi / 2

    return azimuth / np.pi, elevation / np.pi


def rir_synthesis_noise(
    rir_manifest: pd.DataFrame, speech_manifest: pd.DataFrame, noise_database: list
):
    rt60_tgt = rir_manifest["T60"]
    room_dim = ast.literal_eval(rir_manifest["volume_dim"])
    src_pos = ast.literal_eval(rir_manifest["srcPos"])
    mic_pos = ast.literal_eval(rir_manifest["micPos"])

    fs = 16000
    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

    # Create the room
    room = pra.ShoeBox(
        room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
    )
    room.add_source(src_pos)
    room.add_microphone(mic_pos)
    room.compute_rir()

    rir = room.rir[0][0]
    # convert to tensor
    rir = torch.tensor(rir)

    # compute Th, Tt
    Th, Tt, _ = RIRestThTt(rir, fs)
    Th, Tt = round(Th.tolist(), 3), round(Tt.tolist(), 3)

    speech_path = speech_manifest["filename"]
    speech, fs = torchaudio.load(speech_path)

    # rir = rir / rir.max(dim=-1)[0]  # normalize the rir
    reverb_speech = F.fftconvolve(speech, rir.unsqueeze(0))

    # add noise
    noise_idx = torch.randint(len(noise_database), (1,))
    # if len(reverb_speech) > len(noise_database[noise_idx]):
    #     noise_pad = f.pad(
    #         noise_database[noise_idx],
    #         (0, len(reverb_speech) - len(noise_database[noise_idx])),
    #     )
    # elif len(reverb_speech) <= len(noise_database[noise_idx]):
    #     noise_pad = noise_database[noise_idx][: len(reverb_speech)]
    noise_pad = f.interpolate(
        noise_database[noise_idx].unsqueeze(0), size=reverb_speech.size(-1)
    ).squeeze(0)

    snr_list = torch.tensor(
        [0, 5, 10, 15, 20, 100]
    )  # 100 represents without noise influence
    snr_idx = torch.randint(snr_list.numel(), (1,))
    noise_reverb_speech = F.add_noise(reverb_speech, noise_pad, snr=snr_list[snr_idx])

    volume_log10_unitary = rir_manifest["volume_log10_norm"]

    volume = rir_manifest["volume"]
    volume_log10 = (
        np.log10(volume).round(3).item()
    )  # logarithmic volume to torlerate more scale volume
    distRcv = rir_manifest["distRcv"]
    distRcv_norm = rir_manifest["distRcv_norm"]

    azimuth, elevation = calculate_angles_arctan(src_pos, mic_pos)

    snr_dB = snr_list[snr_idx].item()
    return (
        noise_reverb_speech,
        volume_log10_unitary,
        Th,
        Tt,
        volume.tolist(),
        volume_log10,
        distRcv,
        distRcv_norm,
        azimuth,
        elevation,
        snr_dB,
    )


j = 43430  # continue from the last index of the training set
reverbSpeech = []
for i in tqdm(range(len(val_sim_rir_manifest))):
    results = rir_synthesis_noise(
        val_sim_rir_manifest.iloc[i], speech_long_manifest.iloc[i], noise_database
    )
    noise_reverb_speech = results[0]
    volume_log10_unitary = results[1]
    Th = results[2]
    Tt = results[3]
    volume = results[4]
    volume_log10 = results[5]
    distRcv = results[6]
    distRcv_norm = results[7]
    azimuth = results[8]
    elevation = results[9]
    snr_dB = results[10]

    # save the data
    save_path = os.path.join(
        "./data/noiseReverbSpeech",
        "sim_speech.data",
    )
    save_path_label = os.path.join(
        "./data/noiseReverbSpeech",
    )

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torchaudio.save(
        os.path.join(save_path, "reverbSpeech_No" + str(j) + ".wav"),
        noise_reverb_speech,
        sample_rate=16000,
        bits_per_sample=16,
        encoding="PCM_S",
    )

    # save the metadata
    reverbSpeech.append(
        [
            "reverbSpeech_No" + str(j) + ".wav",
            volume_log10_unitary,
            Th,
            Tt,
            volume,
            volume_log10,
            distRcv,
            distRcv_norm,
            azimuth,
            elevation,
            snr_dB,
        ]
    )

    j += 1

reverbSpeech = pd.DataFrame(
    reverbSpeech,
    columns=[
        "reverbSpeech",
        "volume_log10_unitary",
        "Th",
        "Tt",
        "volume",
        "volume_log10",
        "distRcv",
        "distRcv_norm",
        "azimuth",
        "elevation",
        "snr_dB",
    ],
)

# reshuffle the data
reverbSpeech = reverbSpeech.sample(frac=1, random_state=3046)

reverbSpeech.to_csv(os.path.join(save_path_label, "val_sim_manifest.csv"), index=False)
