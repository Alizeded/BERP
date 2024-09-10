import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from nnAudio.features.stft import STFT


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


class MG_TCN(nn.Module):
    def __init__(
        self,
        kernel_size: int = 5,
        dilation: int = 1,
        ch_in: int = 64,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.ch_in = ch_in

        self.shared_unit = nn.Sequential(
            nn.Conv1d(
                in_channels=ch_in,
                out_channels=64,
                kernel_size=1,
                stride=1,
            ),
            nn.PReLU(),
            nn.BatchNorm1d(64),
        )

        self.TGN_unit1 = nn.Conv1d(
            in_channels=64,
            out_channels=64,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
        )

        self.TGN_unit2 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
            ),
            nn.Sigmoid(),
        )

        self.TGN_output = nn.Sequential(
            nn.PReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(
                in_channels=64,
                out_channels=ch_in,
                kernel_size=1,
                stride=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x

        x = self.shared_unit(x)
        z1 = self.TGN_unit1(x)
        z2 = self.TGN_unit2(x)
        x = z1 * z2
        x = self.TGN_output(x)

        x = x + res[..., : x.size(-1)]
        return x


class REnet(nn.Module):
    def __init__(
        self,
        ch_out: int = 1,
        ch_H: int = 64,
        depth: int = 5,
        kernel_size: tuple[int, int] = (3, 2),
        stride: tuple[int, int] = (2, 1),
        n_fft_spectrogram: int = 320,
        hop_length_spectrogram: int = 160,
        freq_bins: int = 192,
        dist_src_est: bool = False,
    ):
        super().__init__()

        self.ch_out = ch_out
        self.ch_H = ch_H
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_fft_spectrogram = n_fft_spectrogram
        self.hop_length_spectrogram = hop_length_spectrogram
        self.freq_bins = freq_bins
        self.dist_src_est = dist_src_est

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.feature_extractor = STFT(
            n_fft=n_fft_spectrogram,
            hop_length=hop_length_spectrogram,
            freq_bins=freq_bins,
            output_format="Magnitude",
        )

        self.input_proj = nn.Linear(
            in_features=1, out_features=n_fft_spectrogram // 2 + 1
        )

        # ----------------- Encoder and decoder -----------------

        for i in range(depth):
            if i == 0:
                encode = nn.Sequential(
                    nn.Conv2d(
                        in_channels=n_fft_spectrogram // 2 + 1,
                        out_channels=ch_H * 2,
                        kernel_size=(5, 2),
                        stride=stride,
                    ),
                    nn.GLU(dim=1),
                )
            else:
                encode = nn.Sequential(
                    nn.Conv2d(
                        in_channels=ch_H,
                        out_channels=ch_H * 2,
                        kernel_size=kernel_size,
                        stride=stride,
                    ),
                    nn.GLU(dim=1),
                )
            self.encoder.append(encode)
        # --------------- output layer -----------------
        self.MG_TCNs = nn.Sequential(  # 6 is the feature dim output from the encoder
            MG_TCN(kernel_size=5, dilation=1, ch_in=ch_H * 4),
            MG_TCN(kernel_size=5, dilation=2, ch_in=ch_H * 4),
            MG_TCN(kernel_size=5, dilation=4, ch_in=ch_H * 4),
            MG_TCN(kernel_size=5, dilation=8, ch_in=ch_H * 4),
            MG_TCN(kernel_size=5, dilation=16, ch_in=ch_H * 4),
        )

        self.output_layer = nn.ModuleList()
        for _ in range(3):  # noqa: B007
            self.output_layer.append(self.MG_TCNs)

        self.linear_proj = nn.Linear(in_features=ch_H * 4, out_features=ch_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            x = x.squeeze()  # Bx1xT -> BxT

        # if dist src estimation is enabled, normalize to unitary amplitude
        if self.dist_src_est:
            x = x / x.abs().max(dim=-1, keepdim=True)[0]
        # padding for spectrogram
        n_frame = self.n_fft_spectrogram - self.hop_length_spectrogram
        x = padding(x, n_frame_len=n_frame)

        # ----------------- Encoder -----------------
        x = self.feature_extractor(x)  # B x C x T
        x = 20 * torch.log10(x + 1e-8)  # B x C x T
        x = rearrange(x, "B C T -> B C T ()")  # B x C x T x 1

        x = self.input_proj(x)  # B x C x T x c, c = n_fft_spectrogram // 2 + 1
        x = rearrange(x, "B C T c -> B c C T")  # B x c x C x T

        for i in range(self.depth):
            x = self.encoder[i](x)  # B x 64 x C x T

        # reshape for MG_TCN
        x = rearrange(x, "B C H T -> B (C H) T")  # B x 64 x (C T)

        # ----------------- output layer -----------------
        for i in range(3):
            x = self.output_layer[i](x)  # B x 64 x (C T)

        x = x.permute(0, 2, 1)  # B x T x 64
        x = self.linear_proj(x)  # B x T x ch_out

        return x.squeeze()  # squeeze all dimensions with size 1
