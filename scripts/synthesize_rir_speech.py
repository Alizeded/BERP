import ast
import os
import random
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import rootutils
import torch
import torch.nn.functional as f
import torchaudio
import torchaudio.functional as F
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

torch.manual_seed(2036)
random.seed(2036)

speech_long_manifest = pd.read_csv("./data/LibriSpeech/audio_long_manifest.csv")
rir_manifest = pd.read_csv("./data/RIR_aggregated/RIR.metadata/RIRLabelAugmentV2.csv")

# choose the same length of speech and rir
speech_long_manifest = speech_long_manifest.sample(
    n=len(rir_manifest), random_state=2036
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


def rir_synthesis_noise(
    rir_manifest: pd.DataFrame, speech_manifest: pd.DataFrame, noise_database: list
):
    rir_subpath = rir_manifest["RIR"]
    rir_path = os.path.join("./data/RIR_aggregated/RIR.data", rir_subpath)
    rir, fs = torchaudio.load(rir_path)
    speech_path = speech_manifest["filename"]
    speech, fs = torchaudio.load(speech_path)

    # rir = rir / rir.max(dim=-1)[0]  # normalize the rir
    reverb_speech = F.fftconvolve(speech, rir)

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

    Th_unitary = rir_manifest["Th_norm"]
    Tt_unitary = rir_manifest["Tt_norm"]
    volume_log10_unitary = rir_manifest["volume_log10_norm"]

    Th = rir_manifest["Th"]
    Tt = rir_manifest["Tt"]
    volume = ast.literal_eval(rir_manifest["volume"])
    volume = torch.tensor(volume).prod().round(decimals=0)
    volume_log10 = round(
        volume.log10().tolist(), 3
    )  # logarithmic volume to torlerate more scale volume
    distRcv = rir_manifest["distRcv"]
    oriSrc = rir_manifest["oriSrc"]

    snr_dB = snr_list[snr_idx].item()
    return (
        noise_reverb_speech,
        reverb_speech,
        Th_unitary,
        Tt_unitary,
        volume_log10_unitary,
        Th,
        Tt,
        volume.tolist(),
        volume_log10,
        distRcv,
        oriSrc,
        snr_dB,
    )


def synthesize_reverb_speech(
    rir_manifest: pd.DataFrame = rir_manifest,
    speech_long_manifest: pd.DataFrame = speech_long_manifest,
    noise_database: list = noise_database,
) -> None:
    """
    Synthesize reverb speech with noise
    Args:
        rir_manifest (pd.DataFrame): RIR manifest
        speech_long_manifest (pd.DataFrame): Speech manifest
        noise_database (list): Noise database
    """
    j = 0
    reverbSpeech = []
    with ThreadPoolExecutor(max_workers=os.cpu_count() - 12) as executor:
        thread = []
        for i in range(len(rir_manifest)):
            thread.append(
                executor.submit(
                    rir_synthesis_noise,
                    rir_manifest.iloc[i],
                    speech_long_manifest.iloc[i],
                    noise_database,
                )
            )
        for t in tqdm(thread):
            noise_reverb_speech = t.result()[0]
            reverb_speech = t.result()[1]
            Th_unitary = t.result()[2]
            Tt_unitary = t.result()[3]
            volume_log10_unitary = t.result()[4]
            Th = t.result()[5]
            Tt = t.result()[6]
            volume = t.result()[7]
            volume_log10 = t.result()[8]
            distRcv = t.result()[9]
            oriSrc = t.result()[10]
            snr_dB = t.result()[11]

            # print(volume_log10)

            # save the data
            save_path = os.path.join(
                "./data/noiseReverbSpeech",
                "speech.data",
            )
            save_path_clean = os.path.join(
                "./data/noiseReverbSpeech",
                "speech_clean.data",
            )
            save_path_label = os.path.join(
                "./data/noiseReverbSpeech",
            )

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if not os.path.exists(save_path_clean):
                os.makedirs(save_path_clean)

            torchaudio.save(
                os.path.join(save_path, "reverbSpeech_No" + str(j) + ".wav"),
                noise_reverb_speech,
                sample_rate=16000,
                bits_per_sample=16,
                encoding="PCM_S",
            )
            torchaudio.save(
                os.path.join(save_path_clean, "clean_No" + str(j) + ".wav"),
                reverb_speech,
                sample_rate=16000,
                bits_per_sample=16,
                encoding="PCM_S",
            )

            # save the metadata
            reverbSpeech.append(
                [
                    "reverbSpeech_No" + str(j) + ".wav",
                    "clean_No" + str(j) + ".wav",
                    Th_unitary,
                    Tt_unitary,
                    volume_log10_unitary,
                    Th,
                    Tt,
                    volume,
                    volume_log10,
                    distRcv,
                    oriSrc,
                    snr_dB,
                ]
            )

            j += 1

    reverbSpeech = pd.DataFrame(
        reverbSpeech,
        columns=[
            "reverbSpeech",
            "cleanSpeech",
            "Th_unitary",
            "Tt_unitary",
            "volume_log10_unitary",
            "Th",
            "Tt",
            "volume",
            "volume_log10",
            "distRcv",
            "oriSrc",
            "snr_dB",
        ],
    )

    reverbSpeech.to_csv(
        os.path.join(save_path_label, "reverbSpeech.metadata.csv"), index=False
    )
