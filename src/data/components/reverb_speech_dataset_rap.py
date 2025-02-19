from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from nnAudio.features.gammatone import Gammatonegram
from nnAudio.features.mel import MelSpectrogram
from nnAudio.features.stft import STFT
from torch.utils.data import Dataset


class ReverbSpeechDatasetRap(Dataset):
    def __init__(
        self,
        feat: Optional[list],
        sti: Optional[list],
        alcons: Optional[list],
        t60: Optional[list],
        edt: Optional[list],
        c80: Optional[list],
        c50: Optional[list],
        d50: Optional[list],
        ts: Optional[list],
        volume: Optional[list],
        dist_src: Optional[list],
        feature_extractor: nn.Module = None,
        norm_amplitude: Optional[bool] = False,
        normalization: bool = True,
    ):
        self.feat = feat
        self.sti = sti
        self.alcons = alcons
        self.t60 = t60
        self.edt = edt
        self.c80 = c80
        self.c50 = c50
        self.d50 = d50
        self.ts = ts
        self.volume = volume
        self.dist_src = dist_src
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
        sti = torch.tensor(self.sti[idx])
        alcons = torch.tensor(self.alcons[idx])
        t60 = torch.tensor(self.t60[idx])
        edt = torch.tensor(self.edt[idx])
        c80 = torch.tensor(self.c80[idx])
        c50 = torch.tensor(self.c50[idx])
        d50 = torch.tensor(self.d50[idx])
        ts = torch.tensor(self.ts[idx])
        volume = torch.tensor(self.volume[idx])
        dist_src = torch.tensor(self.dist_src[idx])

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
            "sti": sti,
            "alcons": alcons,
            "t60": t60,
            "edt": edt,
            "c80": c80,
            "c50": c50,
            "d50": d50,
            "ts": ts,
            "volume": volume,
            "dist_src": dist_src,
        }
