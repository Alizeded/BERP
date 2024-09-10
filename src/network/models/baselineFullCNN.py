from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from nnAudio.features.gammatone import Gammatonegram


def padding(x, n_frame_len):
    """padding zeroes to x so that denoised audio has the same length"""

    len_seq = x.shape[-1]
    n_frame = 1997 - 1
    if len_seq < n_frame_len * n_frame:
        len_seq = int(n_frame_len * n_frame)
        x = F.pad(x, (0, len_seq - x.shape[-1]), mode="constant")
    elif len_seq > n_frame_len * n_frame:
        len_seq = int(n_frame_len * n_frame)
        x = x[:, :len_seq]
    else:
        len_seq = len_seq
        x = x
    return x


class FullCNN(nn.Module):
    def __init__(
        self,
        ch_in: int = 1,
        ch_out: int = 1,
        n_fft_gammatone: Optional[int] = 384,  # for gammatone prenet
        n_bins_gammatone: Optional[int] = 128,
        hop_length_gammatone: Optional[int] = 192,
        dropout_prob: Optional[float] = 0.5,
        dist_src_est: Optional[bool] = False,
    ):
        super().__init__()

        self.ch_in = ch_in
        self.ch_out = ch_out
        self.dropout_prob = dropout_prob
        self.n_fft_gammatone = n_fft_gammatone
        self.n_bins_gammatone = n_bins_gammatone
        self.hop_length_gammatone = hop_length_gammatone
        self.dist_src_est = dist_src_est

        self.feature_extractor = Gammatonegram(
            sr=16000,
            n_fft=n_fft_gammatone,
            n_bins=n_bins_gammatone,
            hop_length=hop_length_gammatone,
            power=1.0,
        )

        self.full_cnn = nn.ModuleList()
        for layer in range(6):
            if layer == 0:
                self.full_cnn.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=1,
                            out_channels=30,
                            kernel_size=(1, 10),
                            stride=(1, 1),
                        ),
                        nn.ReLU(),
                        nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
                    )
                )
            elif layer == 1:
                self.full_cnn.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=30,
                            out_channels=20,
                            kernel_size=(1, 10),
                            stride=(1, 1),
                        ),
                        nn.ReLU(),
                        nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
                    )
                )
            elif layer == 2:
                self.full_cnn.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=20,
                            out_channels=10,
                            kernel_size=(1, 11),
                            stride=(1, 1),
                        ),
                        nn.ReLU(),
                        nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
                    )
                )
            elif layer == 3:
                self.full_cnn.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=10,
                            out_channels=10,
                            kernel_size=(1, 11),
                            stride=(1, 1),
                        ),
                        nn.ReLU(),
                        nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
                    )
                )
            elif layer == 4:
                self.full_cnn.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=10,
                            out_channels=5,
                            kernel_size=(3, 8),
                            stride=(1, 1),
                        ),
                        nn.ReLU(),
                        nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
                    )
                )
            elif layer == 5:
                self.full_cnn.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=5,
                            out_channels=5,
                            kernel_size=(4, 7),
                            stride=(1, 1),
                        ),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d(output_size=(ch_out, 1)),
                        nn.Dropout(dropout_prob),
                    )
                )

            self.linear = nn.Linear(in_features=5, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            x = x.squeeze(1)  # Bx1xT -> BxT

        # if dist src estimation enabled, normalize to unitary amplitude
        if self.dist_src_est:
            x = x / torch.max(torch.abs(x), dim=-1, keepdim=True)[0]

        # padding for gammatonegram
        n_frame = self.n_fft_gammatone - self.hop_length_gammatone
        x = padding(x, n_frame)

        x = self.feature_extractor(x)
        x = 20 * (x + 1e-8).log10()
        x = x.unsqueeze(1)  # B x C x T -> B x 1 x C x T

        for layer in self.full_cnn:
            x = layer(x)
        # output: [batch, 5, ch_out, 1]

        x = x.permute(0, 3, 2, 1)  # [batch, 5, ch_out, 1] -> [batch, 1, ch_out, 5]
        x = self.linear(x)
        x = x.squeeze()  # [batch, 1, ch_out] -> [batch, ch_out]

        return x
