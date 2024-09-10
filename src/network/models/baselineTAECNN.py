from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import Resample

from src.utils.envelope import TemporalEnvelope


def temporal_padding(
    x: torch.Tensor, max_length_sec: Optional[int] = 20, fs: Optional[int] = 16000
):
    """padding zeroes to x so that audio has the same length"""
    len_seq = x.shape[-1]
    if len_seq < max_length_sec * fs:
        len_seq = int(max_length_sec * fs)
        x = F.pad(x, (0, len_seq - x.shape[-1]), mode="constant")
    elif len_seq > max_length_sec * fs:
        len_seq = int(max_length_sec * fs)
        x = x[:, :len_seq]

    return x


class TAECNN(nn.Module):
    def __init__(
        self,
        ch_in: int = 1,
        ch_out: int = 1,
        max_length_sec: Optional[int] = 20,
        fs: Optional[int] = 16000,
        fc: Optional[int] = 20,
        dropout_prob: float = 0.4,
    ):
        super().__init__()

        self.ch_in = ch_in
        self.ch_out = ch_out
        self.dropout_prob = dropout_prob
        self.max_length_sec = max_length_sec
        self.fs = fs
        self.fc = fc

        # temporal amplitude envelope
        self.envelope = TemporalEnvelope(dim=1, fs=fs, fc=fc, mode="TAE")

        self.resample = Resample(orig_freq=fs, new_freq=2 * fc)

        self.conv_block = nn.Sequential(
            nn.Conv1d(  # 1st conv layer
                in_channels=ch_in,
                out_channels=32,
                kernel_size=10,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Conv1d(  # 2nd conv layer
                in_channels=32,
                out_channels=16,
                kernel_size=5,
            ),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Conv1d(  # 3rd conv layer
                in_channels=16,
                out_channels=8,
                kernel_size=5,
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(  # 4th conv layer
                in_channels=8,
                out_channels=4,
                kernel_size=5,
                stride=1,
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.linear = nn.Sequential(
            nn.Linear(in_features=4, out_features=ch_out),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # normalize to unitary amplitude
        x = x / torch.max(torch.abs(x), dim=-1, keepdim=True)[0]  # B x T

        # temporal amplitude envelope extraction
        x = temporal_padding(x, self.max_length_sec, self.fs)  # B x T
        x = self.envelope(x)  # B x T
        # resample to 40 Hz
        x = self.resample(x)  # B x T

        # normalize to unitary amplitude
        x = x / torch.max(torch.abs(x), dim=-1, keepdim=True)[0]  # B x T

        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, T) -> (B, 1, T)

        # TAE-CNN
        x = self.conv_block(x)
        x = x.squeeze()  # [batch, 4, 1] -> [batch, 4]
        x = self.linear(x)  # [batch, 4] -> [batch, ch_out]

        return x.squeeze()  # squeeze the dimension of 1
