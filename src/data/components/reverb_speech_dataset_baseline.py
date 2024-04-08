from typing import Optional

import torchaudio
import torch
from torch.utils.data import Dataset


class ReverbSpeechDataset(Dataset):
    def __init__(
        self,
        raw_audio: Optional[list],
        clean_audio: Optional[list],
        Th: Optional[list],
        Tt: Optional[list],
        volume: Optional[list],
        dist_src: Optional[list],
        azimuth_src: Optional[list],
        elevation_src: Optional[list],
        azimuth_classif: Optional[list] = None,
        elevation_classif: Optional[list] = None,
    ):
        self.raw_audio = raw_audio
        self.clean_audio = clean_audio
        self.Th = Th
        self.Tt = Tt
        self.volume = volume
        self.dist_src = dist_src
        self.azimuth_src = azimuth_src
        self.elevation_src = elevation_src

    def __len__(self):
        # assert len(self.raw_audio) == len(self.clean_audio)
        return len(self.raw_audio)

    def __getitem__(self, idx):
        raw, fs = torchaudio.load(self.raw_audio[idx])  #
        raw = raw.squeeze()  # raw and clean have shape (80000,)
        Th = torch.tensor(self.Th[idx])
        Tt = torch.tensor(self.Tt[idx])
        volume = torch.tensor(self.volume[idx])
        dist_src = torch.tensor(self.dist_src[idx])
        azimuth_src = torch.tensor(self.azimuth_src[idx])
        elevation_src = torch.tensor(self.elevation_src[idx])

        return {
            "raw": raw,
            "Th": Th,
            "Tt": Tt,
            "volume": volume,
            "dist_src": dist_src,
            "azimuth_src": azimuth_src,
            "elevation_src": elevation_src,
        }
