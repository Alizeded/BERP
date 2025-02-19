from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from nnAudio.features.gammatone import Gammatonegram
from nnAudio.features.mel import MelSpectrogram
from nnAudio.features.stft import STFT
from torch.utils.data import Dataset


class ReverbSpeechDataset(Dataset):
    def __init__(
        self,
        feat: Optional[list],
        Th: Optional[list],
        Tt: Optional[list],
        volume: Optional[list],
        dist_src: Optional[list],
        azimuth_src: Optional[list] = None,
        elevation_src: Optional[list] = None,
        azimuth_classif: Optional[list] = None,
        elevation_classif: Optional[list] = None,
        sti: Optional[list] = None,
        alcons: Optional[list] = None,
        t60: Optional[list] = None,
        edt: Optional[list] = None,
        c80: Optional[list] = None,
        c50: Optional[list] = None,
        d50: Optional[list] = None,
        ts: Optional[list] = None,
        feature_extractor: nn.Module = None,
        norm_amplitude: Optional[bool] = False,
        normalization: bool = True,
    ):
        self.feat = feat
        self.Th = Th
        self.Tt = Tt
        self.volume = volume
        self.dist_src = dist_src
        self.azimuth_src = azimuth_src
        self.elevation_src = elevation_src
        self.azimuth_classif = azimuth_classif
        self.elevation_classif = elevation_classif
        self.sti = sti
        self.alcons = alcons
        self.t60 = t60
        self.edt = edt
        self.c80 = c80
        self.c50 = c50
        self.d50 = d50
        self.ts = ts
        self.feature_extractor = feature_extractor
        self.norm_amplitude = norm_amplitude
        self.normalization = normalization

    def post_process(self, feat: torch.Tensor) -> torch.Tensor:
        if self.feature_extractor is not None:
            if (
                isinstance(self.feature_extractor, Gammatonegram)
                or isinstance(self.feature_extractor, MelSpectrogram)
                or isinstance(self.feature_extractor, STFT)
            ):
                feat = 20 * (feat + 1e-8).log10()
        if self.feature_extractor is None and self.normalization:
            feat = F.layer_norm(feat, normalized_shape=feat.shape)

        return feat

    def __len__(self):
        # assert len(self.feat) == len(self.clean_audio)
        return len(self.feat)

    def __getitem__(self, idx):
        raw, fs = torchaudio.load(self.feat[idx])  # raw
        raw = raw.squeeze()  # raw
        Th = torch.tensor(self.Th[idx])
        Tt = torch.tensor(self.Tt[idx])
        volume = torch.tensor(self.volume[idx])
        dist_src = torch.tensor(self.dist_src[idx])
        azimuth_src = (
            torch.tensor(self.azimuth_src[idx])
            if self.azimuth_src is not None
            else None
        )
        elevation_src = (
            torch.tensor(self.elevation_src[idx])
            if self.elevation_src is not None
            else None
        )
        if self.azimuth_classif is not None:
            azimuth_classif = torch.tensor(self.azimuth_classif[idx])
        else:
            azimuth_classif = None
        if self.elevation_classif is not None:
            elevation_classif = torch.tensor(self.elevation_classif[idx])
        else:
            elevation_classif = None

        sti = torch.tensor(self.sti[idx]) if self.sti is not None else None
        alcons = torch.tensor(self.alcons[idx]) if self.alcons is not None else None
        t60 = torch.tensor(self.t60[idx]) if self.t60 is not None else None
        edt = torch.tensor(self.edt[idx]) if self.edt is not None else None
        c80 = torch.tensor(self.c80[idx]) if self.c80 is not None else None
        c50 = torch.tensor(self.c50[idx]) if self.c50 is not None else None
        d50 = torch.tensor(self.d50[idx]) if self.d50 is not None else None
        ts = torch.tensor(self.ts[idx]) if self.ts is not None else None

        # Normalize amplitude
        if self.norm_amplitude:
            raw = raw / raw.abs().max(dim=-1, keepdim=True)[0]

        # Extract features
        if self.feature_extractor is not None:
            feat = self.feature_extractor(raw).squeeze()  # (1, C, T) -> (C, T)
        elif self.feature_extractor is None:
            feat = raw

        # Post-process
        feat = self.post_process(feat)

        return {
            "feat": feat,
            "Th": Th,
            "Tt": Tt,
            "volume": volume,
            "dist_src": dist_src,
            "azimuth_src": azimuth_src,
            "elevation_src": elevation_src,
            "azimuth_classif": azimuth_classif,
            "elevation_classif": elevation_classif,
            "sti": sti,
            "alcons": alcons,
            "t60": t60,
            "edt": edt,
            "c80": c80,
            "c50": c50,
            "d50": d50,
            "ts": ts,
        }
