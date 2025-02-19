from typing import Optional

import torch
import torch.nn.functional as F
import torchaudio
from nnAudio.features.gammatone import Gammatonegram
from nnAudio.features.mel import MelSpectrogram
from nnAudio.features.stft import STFT
from torch import nn
from torch.utils.data import Dataset


class MixedSpeechDataset(Dataset):
    def __init__(
        self,
        feat: Optional[list],
        label: Optional[list],
        feature_extractor: nn.Module = None,
        normalization: bool = False,
    ):
        self.feat = feat
        self.label = label
        self.feature_extractor = feature_extractor
        self.normalization = normalization

    def post_process(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_extractor is not None:
            if (
                isinstance(self.feature_extractor, Gammatonegram)
                or isinstance(self.feature_extractor, MelSpectrogram)
                or isinstance(self.feature_extractor, STFT)
            ):
                x = 20 * (x + 1e-8).log10()
            if self.feature_extractor is None and self.normalization:
                x = F.layer_norm(x)

        return x

    def __len__(self):
        assert len(self.feat) == len(self.label)
        return len(self.feat)

    def __getitem__(self, idx):
        feat, fs = torchaudio.load(self.feat[idx])
        feat = feat.squeeze()  # (1, T) -> (T,)

        label = torch.load(self.label[idx], weights_only=True)
        label = label.squeeze()  # (1, T) -> (T,)

        feat = feat[: label.shape[-1]]  # Truncate feat to match label

        # Extract features
        if self.feature_extractor is not None:
            feat = self.feature_extractor(feat).squeeze()  # (1, C, T) -> (C, T)

        # Post-process
        feat = self.post_process(feat)

        return {
            "feat": feat,
            "label": label,
        }
