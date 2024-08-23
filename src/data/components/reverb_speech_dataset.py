from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from nnAudio.features.gammatone import Gammatonegram
from nnAudio.features.mel import MelSpectrogram

from torch.utils.data import Dataset


class ReverbSpeechDataset(Dataset):
    def __init__(
        self,
        feat: Optional[list],
        Th: Optional[list],
        Tt: Optional[list],
        volume: Optional[list],
        dist_src: Optional[list],
        azimuth_src: Optional[list],
        elevation_src: Optional[list],
        azimuth_classif: Optional[list] = None,
        elevation_classif: Optional[list] = None,
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
        self.feature_extractor = feature_extractor
        self.norm_amplitude = norm_amplitude
        self.normalization = normalization

    def post_process(self, feat: torch.Tensor) -> torch.Tensor:
        if self.feature_extractor is not None and (
            isinstance(self.feature_extractor, (Gammatonegram, MelSpectrogram))
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
        azimuth_src = torch.tensor(self.azimuth_src[idx])
        elevation_src = torch.tensor(self.elevation_src[idx])
        if self.azimuth_classif is not None:
            azimuth_classif = torch.tensor(self.azimuth_classif[idx])
        else:
            azimuth_classif = None
        if self.elevation_classif is not None:
            elevation_classif = torch.tensor(self.elevation_classif[idx])
        else:
            elevation_classif = None

        # Normalize amplitude
        if self.norm_amplitude:
            raw = raw / raw.abs().max(dim=-1, keepdim=True)[0]

        # Extract features
        if self.feature_extractor is not None:
            feat = self.feature_extractor(raw).squeeze()  # (1, C, T) -> (C, T)
        else:
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
        }
