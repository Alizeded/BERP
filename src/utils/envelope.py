from typing import Optional

import scipy.signal as signal
import torch
import torch.nn as nn
import torchaudio.functional as F

"""
Envelope extraction using the Hilbert transform and a low-pass filter
"""


class ButterWorthFilter(nn.Module):
    """
    extract the envelope of a signal using the Hilbert transform and a low-pass filter
    x: input signal
    fc: cut-off frequency of the low-pass filter
    fs: sampling frequency
    TE: "TPE" for the temporal power envelope, "TAE" for the temporal amplitude envelope
    """

    def __init__(self, fs: int = 16000, fc: Optional[int] = 128):
        self.fc = fc
        self.fs = fs

    def lpf(self):
        if self.fc <= 200:
            N = 6
        elif self.fc > 200 and self.fc <= 700:
            N = 9
        else:
            N = 12
        Bd, Ad = signal.butter(N, Wn=self.fc, btype="lowpass", fs=self.fs, output="ba")
        Bd, Ad = torch.as_tensor(Bd), torch.as_tensor(Ad)

        return Bd, Ad


class TemporalEnvelope(nn.Module):
    def __init__(
        self, dim: int, fs: int = 16000, mode: str = "envelope", fc: Optional[int] = 128
    ):
        super().__init__()
        self.fc = fc
        self.fs = fs
        self.mode = mode

        self.hilbert = HilbertTransform(axis=dim)
        self.Bd, self.Ad = ButterWorthFilter(fs=self.fs, fc=self.fc).lpf()

    def hilbert_filt(self, x: torch.Tensor):
        """Hilbert transform using a filter"""

        analytic_signal = self.hilbert(x)
        amplitude_envelope = torch.abs(analytic_signal)
        return amplitude_envelope

    def TAE(self, x):
        # temporal amplitude envelope
        x = self.hilbert_filt(x)
        x = F.filtfilt(x, self.Ad.to(x.device), self.Bd.to(x.device))
        return x

    def envelope(self, x):
        return self.hilbert_filt(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "envelope":
            return self.envelope(x).float()
        elif self.mode == "TAE":
            return self.TAE(x).float()
        else:
            raise ValueError("mode must be 'envelope' or 'TAE'.")


class HilbertTransform(nn.Module):
    """
    Determine the analytical signal of a Tensor along a particular axis.

    Args:
        axis: Axis along which to apply Hilbert transform. Default 2 (first spatial dimension).
        n: Number of Fourier components (i.e. FFT size). Default: ``x.shape[axis]``.

    References:
        [1] implementation of Hilbert transform in scipy.signal.hilbert
    """

    def __init__(self, N=None, axis=2) -> None:
        super().__init__()
        self.axis = axis
        self.N = N

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor or array-like to transform. Must be real and in shape ``[Batch, chns, spatial1, spatial2, ...]``.
        Returns:
            torch.Tensor: Analytical signal of ``x``, transformed along axis specified in ``self.axis`` using
            FFT of size ``self.N``. The absolute value of ``x_ht`` relates to the envelope of ``x`` along axis ``self.axis``.
        """
        # Make input a real tensor
        x = torch.as_tensor(x, device=x.device if isinstance(x, torch.Tensor) else None)
        if torch.is_complex(x):
            raise ValueError("x must be real.")

        if (self.axis < 0) or (self.axis > len(x.shape) - 1):
            raise ValueError(
                f"Invalid axis for shape of x, got axis {self.axis} and shape {x.shape}."
            )

        N = x.shape[self.axis] if self.N is None else self.N
        if N <= 0:
            raise ValueError("N must be positive.")

        Xf = torch.fft.fft(x.double(), n=N, dim=self.axis)
        h = torch.zeros(N, dtype=torch.cfloat, device=x.device)
        if N % 2 == 0:
            h[0] = h[N // 2] = 1
            h[1 : N // 2] = 2
        else:
            h[0] = 1
            h[1 : (N + 1) // 2] = 2

        if x.dim() > 1:
            ind = [None] * x.dim()
            ind[self.axis] = slice(None)
            h = h[tuple(ind)]

        x = torch.fft.ifft(Xf * h, dim=self.axis)
        return x
