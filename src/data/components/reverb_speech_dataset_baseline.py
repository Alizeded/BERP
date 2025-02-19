from typing import Optional

import torch
import torchaudio
from torch.utils.data import Dataset


class ReverbSpeechDatasetBaseline(Dataset):

    def __init__(
        self,
        raw_audio: Optional[list],
        Th: Optional[list],
        Tt: Optional[list],
        volume: Optional[list],
        sti: Optional[list],
        alcons: Optional[list],
        t60: Optional[list],
        c80: Optional[list],
        c50: Optional[list],
        volume_ns: Optional[list],
        dist_src: Optional[list],
        edt: Optional[list],
        d50: Optional[list],
        ts: Optional[list],
    ):
        self.raw_audio = raw_audio
        self.Th = Th
        self.Tt = Tt
        self.volume = volume
        self.sti = sti
        self.alcons = alcons
        self.t60 = t60
        self.c80 = c80
        self.c50 = c50
        self.volume_ns = volume_ns
        self.dist_src = dist_src
        self.edt = edt
        self.d50 = d50
        self.ts = ts

    def __len__(self):
        # assert len(self.raw_audio) == len(self.clean_audio)
        return len(self.raw_audio)

    def __getitem__(self, idx):
        raw, fs = torchaudio.load(self.raw_audio[idx])  #
        raw = raw.squeeze()  # raw and clean have shape (80000,)
        Th = torch.tensor(self.Th[idx])
        Tt = torch.tensor(self.Tt[idx])
        volume = torch.tensor(self.volume[idx])
        sti = torch.tensor(self.sti[idx])
        alcons = torch.tensor(self.alcons[idx])
        t60 = torch.tensor(self.t60[idx])
        c80 = torch.tensor(self.c80[idx])
        c50 = torch.tensor(self.c50[idx])
        volume_ns = torch.tensor(self.volume_ns[idx])
        dist_src = torch.tensor(self.dist_src[idx])
        d50 = torch.tensor(self.d50[idx])
        ts = torch.tensor(self.ts[idx])
        edt = torch.tensor(self.edt[idx])
        d50 = torch.tensor(self.d50[idx])
        ts = torch.tensor(self.ts[idx])

        return {
            "raw": raw,
            "Th": Th,
            "Tt": Tt,
            "volume": volume,
            "sti": sti,
            "alcons": alcons,
            "t60": t60,
            "edt": edt,
            "c80": c80,
            "c50": c50,
            "d50": d50,
            "ts": ts,
            "volume_ns": volume_ns,
            "dist_src": dist_src,
        }
