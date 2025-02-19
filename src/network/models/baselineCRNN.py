from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from nnAudio.features.mel import MFCC

# def padding(x, n_frame_len):
#     """padding zeroes to x so that denoised audio has the same length,
#     when transforming using Fourier Transform, it will be as the upsampling process in the frequency domain,
#     so that the output has the same length as the input without losing information
#     see: B.P. Lathi, Linear Systems and Signals, 3rd ed., Oxford University Press, 2018, Ch. 7 & 8.
#     """
#     len_seq = x.shape[-1]
#     n_frame = 1997 - 1
#     if len_seq < n_frame_len * n_frame:
#         len_seq = int(n_frame_len * n_frame)
#         x = F.pad(x, (0, len_seq - x.shape[-1]), mode="constant")
#     elif len_seq > n_frame_len * n_frame:
#         len_seq = int(n_frame_len * n_frame)
#         x = x[:, :len_seq]
#     else:
#         len_seq = len_seq
#         x = x
#     return x


class GRUModule(nn.Module):
    def __init__(self, ch_in: int = 128):
        super().__init__()
        self.gru = nn.GRU(
            input_size=ch_in,  # default: 128
            hidden_size=32,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:  # for batch size 1
            x = x.unsqueeze(0)
        x = x.permute(0, 2, 1)  # (batch, n_mfcc, seq_len) -> (batch, seq_len, n_mfcc)
        x, _ = self.gru(x)
        return x


class MaxPool2dAdaptive(nn.Module):
    def __init__(self, output_size: tuple[int, int]):
        super().__init__()
        self.output_size = output_size
        self.adaptive_max_pool2d = nn.AdaptiveMaxPool2d(output_size=output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adaptive_max_pool2d(x)
        return (
            x.squeeze()
        )  # (batch, ch_out, n_mfcc, seq_len) -> (batch, ch_out, seq_len)


class LinearAdaptiveAvgPool1d(nn.Module):
    def __init__(
        self,
        output_size: int,
    ):
        super().__init__()
        self.output_size = output_size
        self.adaptive_avg_pool1d = nn.AdaptiveAvgPool1d(output_size=output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # (batch, seq_len, n_mfcc) -> (batch, n_mfcc, seq_len)
        x = self.adaptive_avg_pool1d(x)
        x = x.permute(0, 2, 1)  # (batch, n_mfcc, seq_len) -> (batch, seq_len, n_mfcc)

        return x


class CRNN(nn.Module):
    def __init__(
        self,
        ch_in: int = 1,
        ch_out: int = 4,
        n_mfcc: Optional[int] = 128,
        n_fft_mel: Optional[int] = 384,
        n_mels: Optional[int] = 128,
        hop_length_mel: Optional[int] = 192,
        dropout_prob: Optional[float] = 0.5,
        dist_src_est: Optional[bool] = False,
    ):
        super().__init__()

        self.ch_in = ch_in
        self.ch_out = ch_out
        self.dropout_prob = dropout_prob
        self.n_mfcc = n_mfcc
        self.n_fft_mel = n_fft_mel
        self.n_mels = n_mels
        self.hop_length_mel = hop_length_mel
        self.dist_src_est = dist_src_est

        self.feature_extractor = MFCC(
            sr=16000,
            n_mfcc=n_mfcc,
            n_fft=n_fft_mel,
            n_mels=n_mels,
            hop_length=hop_length_mel,
            power=1.0,
        )

        self.crnn = nn.ModuleList()
        for layer in range(6):
            if layer == 0:
                self.crnn.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=1,
                            out_channels=64,
                            kernel_size=(3, 3),
                        ),
                        nn.BatchNorm2d(64),
                        nn.ELU(),
                        nn.MaxPool2d(kernel_size=(1, 3)),
                        nn.Dropout(p=dropout_prob),
                    )
                )
            if layer == 1:
                self.crnn.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=64,
                            out_channels=128,
                            kernel_size=(3, 3),
                        ),
                        nn.BatchNorm2d(128),
                        nn.ELU(),
                        nn.MaxPool2d(kernel_size=(1, 3)),
                        nn.Dropout(p=dropout_prob),
                    )
                )
            if layer == 2:
                self.crnn.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=128,
                            out_channels=128,
                            kernel_size=(3, 3),
                        ),
                        nn.BatchNorm2d(128),
                        nn.ELU(),
                        nn.MaxPool2d(kernel_size=(1, 3)),
                        nn.Dropout(p=dropout_prob),
                    )
                )
            if layer == 3:
                self.crnn.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=128,
                            out_channels=128,
                            kernel_size=(3, 3),
                        ),
                        nn.BatchNorm2d(128),
                        nn.ELU(),
                        MaxPool2dAdaptive(output_size=(32, 1)),
                        nn.Dropout(p=dropout_prob),
                    )
                )

            elif layer == 4:
                # GRU layer
                self.crnn.append(GRUModule(ch_in=128))

            elif layer == 5:
                # linear layer
                self.crnn.append(
                    nn.Sequential(
                        nn.Linear(in_features=32, out_features=128),
                        nn.ELU(),
                        nn.Linear(in_features=128, out_features=64),
                        nn.ELU(),
                        nn.Linear(in_features=64, out_features=ch_out),
                        LinearAdaptiveAvgPool1d(output_size=1),
                    )
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            x = x.squeeze(1)  # [batch, 1, seq_len] -> [batch, seq_len]

        # if dist src estimation is enabled, normalize to unitary amplitude
        if self.dist_src_est:
            x = x / x.abs().max(dim=-1, keepdim=True)[0]

        # # padding for MFCC
        # n_frame = self.n_fft_mel - self.hop_length_mel
        # x = padding(x, n_frame_len=n_frame)

        x = self.feature_extractor(x)
        if x.dim() != 4:
            x = x.unsqueeze(1)  # [bz, n_mfcc, L] -> [bz, 1, n_mfcc, L]

        for layer in self.crnn:
            x = layer(x)

        x = x.squeeze()  # [batch, ch_out, 1] -> [batch, ch_out]

        return x
