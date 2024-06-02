import torch
import torch.nn as nn


# ---------------------------- Convolutional layer norm --------------------


class ConvLayerNorm(nn.Module):
    """
    Layer Normalization Module.
    """

    def __init__(self, ch_out: int):
        super(ConvLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(ch_out, eps=1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # [batch, ch, len] -> [batch, len, ch]
        x = self.layer_norm(x)
        x = x.permute(0, 2, 1)  # [batch, len, ch] -> [batch, ch, len]
        return x


# ----------------------------- Parametric Predictor ----------------------------- #
class ParametricPredictor(nn.Module):
    """
    Room Parametric Predictor Module to output predicted Parametric parameters.

    Parameters:
        in_dim: int
            number of input dimensions
        num_layers: int
            number of layers
        num_channels: int
            number of channels
        kernel_size: int
            kernel size
        dropout_prob: float
    """

    def __init__(
        self,
        in_dim: int,
        num_layers: int = 2,
        num_channels: int = 384,
        kernel_size: int = 3,
        dropout_prob: float = 0.5,
    ):
        super().__init__()
        self.conv = nn.ModuleList()
        for layer in range(num_layers):
            in_channels = in_dim if layer == 0 else num_channels
            if layer not in [0, num_layers - 1]:
                self.conv.append(
                    nn.Sequential(
                        nn.Conv1d(
                            in_channels,
                            num_channels,
                            kernel_size,
                            padding=(kernel_size - 1) // 2,
                        ),
                        nn.ReLU(),
                        ConvLayerNorm(num_channels),
                        nn.Dropout(dropout_prob),
                        nn.Conv1d(
                            num_channels,
                            num_channels,
                            kernel_size * 2 - 1,
                            stride=3,
                        ),
                        nn.ReLU(),
                        ConvLayerNorm(num_channels),
                        nn.Dropout(dropout_prob),
                    )
                )
            else:
                self.conv.append(
                    nn.Sequential(
                        nn.Conv1d(
                            in_channels,
                            num_channels,
                            kernel_size,
                            padding=(kernel_size - 1) // 2,
                        ),
                        nn.ReLU(),
                        ConvLayerNorm(num_channels),
                        nn.Dropout(dropout_prob),
                    )
                )

        self.linear = nn.Linear(num_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            x: [batch, len, in_dim]
        Returns:
            [batch, len]
        """
        x = x.permute(0, 2, 1)  # [batch, len, ch] -> [batch, ch, len]
        for layer in self.conv:
            x = layer(x)  # [batch, ch, len]

        x = self.linear(x.permute(0, 2, 1))  # [batch, ch, len] -> [batch, len, 1]

        return x.squeeze(-1)  # [batch, len]
