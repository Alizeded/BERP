import torch
from nnAudio.Spectrogram import STFT
from torch import nn


class ConvLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape)

    def forward(self, x):
        x = x.transpose(1, 2)  # B x C x T -> B x T x C
        x = self.norm(x)
        x = x.transpose(1, 2)  # B x T x C -> B x C x T
        return x


class Model_1(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1d_down_1_depth = nn.Conv1d(
            769, 769, kernel_size=10, stride=1, groups=769, padding=0
        )
        self.conv1d_down_1_point = nn.Conv1d(
            769, 384, kernel_size=1, stride=1, padding=0
        )
        self.ln_1 = ConvLayerNorm(384)

        self.relu = nn.ReLU()

        self.conv1d_down_2_depth = nn.Conv1d(
            384, 384, kernel_size=10, stride=1, groups=384, dilation=2, padding=0
        )
        self.conv1d_down_2_point = nn.Conv1d(384, 192, kernel_size=1, stride=1)
        self.ln_2 = ConvLayerNorm(192)

        self.conv1d_down_3_depth = nn.Conv1d(
            192, 192, kernel_size=2, stride=1, groups=192, dilation=4, padding=0
        )
        self.conv1d_down_3_point = nn.Conv1d(192, 96, kernel_size=1, stride=1)
        self.ln_3 = ConvLayerNorm(96)

        self.drp_1 = nn.Dropout(p=0.2)

        self.drp = nn.Dropout(p=0.5)

    def forward(self, x):
        # x~ ch1, x2~ ch2

        x = self.relu(self.conv1d_down_1_depth(x))

        x = self.ln_1(self.relu(self.conv1d_down_1_point(x)))

        x = self.drp_1(x)

        x = self.relu(self.conv1d_down_2_depth(x))

        x = self.ln_2(self.relu(self.conv1d_down_2_point(x)))

        x = self.drp_1(x)

        x = self.relu(self.conv1d_down_3_depth(x))

        x = self.ln_3(self.relu(self.conv1d_down_3_point(x)))

        return x


class Model_3(torch.nn.Module):
    def __init__(self, ch_out: int = 4):
        super().__init__()
        self.drp = nn.Dropout(p=0.5)
        self.fc_1 = nn.Linear(96, 96)
        self.fc_2 = nn.Linear(96, 48)
        self.fc_3 = nn.Linear(48, ch_out)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.fc_1(x)
        x = self.drp(x)
        x = self.fc_3(self.fc_2(x))

        # split mean and variance
        mean, var = x.chunk(2, dim=-1)

        var = self.softplus(var)

        return mean, var + 1e-6


class EnsembleCNNMLP(nn.Module):

    def __init__(
        self,
        ch_out: int = 4,
        n_fft_spectrogram: int = 1024,
        hop_length_spectrogram: int = 160,
        freq_bins: int = 192,
    ):
        super().__init__()

        self.feature_extractor = STFT(
            n_fft=n_fft_spectrogram,
            hop_length=hop_length_spectrogram,
            freq_bins=freq_bins,
            output_format="Magnitude",
        )

        self.model_a = Model_1()
        self.model_c = Model_3(ch_out=ch_out)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.feature_extractor(x)

        x = self.model_a(x)

        x = self.avg_pool(x).squeeze(-1)  # B x C x T -> B x C

        mean, var = self.model_c(x)

        return mean, var
