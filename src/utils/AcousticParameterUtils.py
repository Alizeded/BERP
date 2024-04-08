from typing import Optional, Any
import random

import torch
import torch.nn as nn
from torch import Tensor

import librosa
import numpy as np
from scipy import signal, interpolate

from src.utils.envelope import TemporalEnvelope

torch.manual_seed(3407)
random.seed(3407)


# ================================== SSIR Model ================================== #
class SparseStochasticIRModel(nn.Module):
    def __init__(
        self,
        Ti: Any,
        Td: Any,
        volume: Any,  # volume of the room
        mu: float,
        fs: int = 16000,
        verbose: bool = False,
    ):
        super(SparseStochasticIRModel, self).__init__()

        if torch.is_tensor(Ti) is False:
            Ti = torch.tensor(Ti)
        if torch.is_tensor(Td) is False:
            Td = torch.tensor(Td)
        if torch.is_tensor(volume):
            volume = volume.item()
        self.Ti = torch.round(Ti, decimals=4)
        self.Td = torch.round(Td, decimals=4)
        if isinstance(volume, float):
            self.volume = int(round(volume, 0))
        elif isinstance(volume, str):
            self.volume = int(volume)
        else:
            self.volume = volume
        self.mu = mu  # poisson distribution parameter, the mean of the sample set
        self.fs = fs
        self.verbose = verbose

    def construct_rir(self) -> Tensor:
        # Th: time of early reflection
        Ti = self.Ti
        # Tt: time of late reflection
        Td = self.Td
        # volume: volume of the room
        volume = self.volume

        fs = self.fs
        early_reflection_range = torch.arange(-Ti, -1 / fs, 1 / fs)
        # Tt: time of late reflection
        late_reverberation_range = torch.arange(0, Td, 1 / fs)

        # early reflection energy curve
        early_reflection_part = torch.exp(6.9 * (early_reflection_range / Ti))

        # early reflection fine structure
        early_reflection_carrier = torch.zeros_like(early_reflection_range)

        # set fixed seed for reproducibility
        early_reflection_carrier = np.random.default_rng(3407).poisson(
            self.mu, size=volume
        )
        early_reflection_carrier = torch.from_numpy(early_reflection_carrier).float()

        # adjust the length of early_reflection_carrier to match the length of early_reflection_range
        if len(early_reflection_carrier) > len(early_reflection_range):
            early_reflection_carrier = early_reflection_carrier[
                -len(early_reflection_range) :
            ]
        elif len(early_reflection_carrier) < len(early_reflection_range):
            early_reflection_carrier = torch.cat(
                (
                    torch.zeros(
                        len(early_reflection_range) - len(early_reflection_carrier)
                    ),
                    early_reflection_carrier,
                )
            )
            early_reflection_carrier = early_reflection_carrier[
                -len(early_reflection_range) :
            ]

        # odd
        early_reflection_carrier[::2] = 1 * early_reflection_carrier[::2]
        # even
        early_reflection_carrier[1::2] = -1 * early_reflection_carrier[1::2]

        early_reflection = early_reflection_part * early_reflection_carrier

        # late reflection energy curve
        late_reverberation_part = torch.exp(-6.9 * (late_reverberation_range / Td))
        # late reflection fine structure
        mu_normal, sigma_normal = 0.0, 1.0  # mean and standard deviation
        late_reverberation_carrier = np.random.default_rng(3407).normal(
            mu_normal, sigma_normal, size=len(late_reverberation_range)
        )  # standard Gaussian distribution
        late_reverberation_carrier = torch.from_numpy(
            late_reverberation_carrier
        ).float()

        late_reverberation = late_reverberation_part * late_reverberation_carrier

        # concatenate early and late reflection
        synthesized_rir = torch.cat((early_reflection, late_reverberation), dim=0)

        synthesized_rir_envelope = torch.cat(
            (early_reflection_part, late_reverberation_part), dim=0
        )

        b = (1 / torch.trapz(synthesized_rir_envelope)).sqrt()

        if self.verbose:
            return b * synthesized_rir, b * synthesized_rir_envelope
        else:
            return b * synthesized_rir

    def forward(self) -> Tensor:
        return self.construct_rir()


def octave_seven_band_filter(x: np.ndarray, fs0: int = 16000) -> np.ndarray:
    """Octave band filter bank

    Parameters:
        x (Tensor): input signal
        fs (int): sampling frequency

    Returns:
        Tensor: octave band filtered signal
    """
    if x.ndim != 1:
        raise ValueError("Only 1D Array is supported.")

    if fs0 / 2 < 12000:
        x_upsampled = librosa.resample(x, orig_sr=fs0, target_sr=24000)
        fs = 24000
    else:
        x_upsampled = x
        fs = fs0

    center_frequency = np.asarray([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])

    upper_bond = center_frequency * np.sqrt(2).round(decimals=1)
    lower_bond = center_frequency / np.sqrt(2).round(decimals=1)

    # bandpass filter
    filtered_signal = np.zeros((7, len(x)))
    for ch in range(7):
        # Calculate butterworth filter coefficients
        wp1 = lower_bond[ch]
        wp2 = upper_bond[ch]
        sos_each_band = signal.butter(
            9, Wn=[wp1, wp2], btype="bandpass", fs=fs, output="sos"
        )

        filtered_banded_signal = signal.sosfilt(sos_each_band, x_upsampled)

        if fs0 / 2 < 12000:
            filtered_banded_signal = librosa.resample(
                filtered_banded_signal, orig_sr=24000, target_sr=fs0
            )
            filtered_signal[ch, ...] = filtered_banded_signal[: len(x)]
            #! truncate the signal to the original length to avoid error
        else:
            filtered_signal[ch, ...] = filtered_banded_signal[: len(x)]

    return filtered_signal


# ============================= Room Acoustic Parameter calculation from RIR ============================= #


class RapidSpeechTransmissionIndex(nn.Module):
    """Speech Transmission Index (RASTI) calculation"""

    def __init__(self):
        super(RapidSpeechTransmissionIndex, self).__init__()

    def forward(self, rir: Tensor, fs: int = 16000, snr: int = 100) -> Tensor:
        if rir.dim() != 1:
            raise ValueError("Only 1D Tensor is supported.")

        rir = rir / torch.max(torch.abs(rir))  # normalize to unit
        rir = rir.numpy()  # convert to numpy array
        rir_len = len(rir)
        rir_octaveband = octave_seven_band_filter(rir, fs)  # octave band filter

        fm = [0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5]

        w = [0.129, 0.143, 0.114, 0.114, 0.186, 0.171, 0.143]

        k = 7  # number of frequency bands
        i = 14  # number of modulation bands

        # envelope extraction
        env_rir_octaveband = np.zeros((k, rir_len))
        envelope_extractor = TemporalEnvelope(dim=0, fs=fs, fc=20, mode="TAE")
        for n in range(k):
            env_rir_each_band = envelope_extractor(rir_octaveband[n, ...])
            env_rir_octaveband[n, ...] = env_rir_each_band[:rir_len]
            # ! truncate the signal to the original length to avoid error

        # modulation transfer function calculation
        mtf_raw = np.zeros((k, rir_len))
        for n in range(k):
            nums_MTF = np.abs(np.fft.fft(env_rir_octaveband[n, ...] ** 2))
            dens_MTF = np.trapz(env_rir_octaveband[n, ...] ** 2)
            mtf_raw[n, ...] = (nums_MTF / dens_MTF) * (1 / (1 + 10 ** (-snr / 10)))

        # modulation transfer function interpolation
        # cut off the samples above 14 Hz
        freq_mtf = np.arange(0, rir_len) / rir_len * fs
        freq = freq_mtf[freq_mtf <= 14]

        mtf = np.zeros((k, freq.size))
        for n in range(k):
            mtf[n, ...] = mtf_raw[n, : freq.size]

        mtf_interp = np.zeros((k, len(fm)))
        for n in range(k):
            mtf_interp[n, ...] = interpolate.interp1d(
                freq, mtf[n, ...], fill_value="extrapolate"
            )(fm)
            # ! using fill_value="extrapolate" to avoid error when fm is out of bounds

        # SNR calculation
        SNR_apparent = 10 * np.log10((mtf_interp / (1 - mtf_interp)))

        # transmission index calculation
        ti = np.zeros((k, i))
        for n in range(k):
            for m in range(i):
                if SNR_apparent[n, m] > 15:
                    ti[n, m] = 1
                elif SNR_apparent[n, m] < -15:
                    ti[n, m] = 0
                else:
                    ti[n, m] = (SNR_apparent[n, m] + 15) / 30

        # modulation transmission index calculation
        mti = np.zeros(k)
        for n in range(k):
            mti[n] = np.sum(ti[n, :]) / i  # sum and average over modulation bands

        sti = np.sum(mti * w)

        return torch.tensor(sti, dtype=torch.float32)


def polyval(p, x) -> torch.Tensor:
    p = torch.as_tensor(p)
    if p.ndim == 0 or (len(p) < 1):
        return torch.zeros_like(x)
    y = torch.zeros_like(x)
    for p_i in p:
        y = y * x + p_i
    return y


class PercentageArticulationLoss(nn.Module):
    """Percentage Articulation Loss Consonant (%AL_cons) calculation by using
    the Farrell Becker empirical formula

    """

    def __init__(self):
        super(PercentageArticulationLoss, self).__init__()

    def forward(self, sti: Tensor) -> Tensor:
        al_cons = 170.5045 * torch.exp(-5.419 * sti)
        return al_cons


class EarlyDecayTime(nn.Module):
    """Early Decay Time (EDT) calculation"""

    def __init__(self, verbose: Optional[bool] = False):
        super(EarlyDecayTime, self).__init__()
        self.verbose = verbose

    def forward(self, rir: Tensor, fs: int = 16000) -> Tensor:
        # backward integral of squared impulse response
        x = torch.cumulative_trapezoid(rir.flip(0) ** 2)
        x = x.flip(0)  # backward integration

        edc = 10 * x.log10()  # early decay curve
        edc = edc - torch.max(edc)  # offset to 0 dB
        idx = torch.arange(1, len(edc) + 1)  # time index

        # find decay line
        # find zero-dB point
        xt1 = torch.where(edc <= 0)[0][0]
        if xt1 == []:
            xt1 = 1
        # find -10dB point (T20)
        xt2 = torch.where(edc <= -10)[0][0]
        if xt2 == []:
            xt2 = torch.min(edc)

        # linear fitting
        def linearfit(sta, end, EDC):
            I_xT = torch.arange(sta, end + 1)
            I_xT = I_xT.reshape(I_xT.numel(), 1)
            A = I_xT ** torch.arange(1, -1, -1.0)
            p = torch.linalg.inv(A.T @ A) @ (A.T @ EDC[I_xT])
            fittedline = polyval(p, idx)
            return fittedline

        fitted_line = linearfit(xt1, xt2, edc)
        fitted_line = fitted_line - torch.max(fitted_line)  # offset to 0 dB

        xt_10dB = 6.2 * torch.where(fitted_line <= -9.8)[0][0]
        edt = (xt_10dB / fs).round(decimals=3)

        if self.verbose:
            return edt.float(), edc.float(), fitted_line.float()
        else:
            return edt.float()


class ReverberationTime(nn.Module):
    """Reverberation Time (T30/T20) calculation"""

    def __init__(self, verbose: Optional[bool] = False):
        super(ReverberationTime, self).__init__()
        self.verbose = verbose

    def forward(self, rir: Tensor, fs: int = 16000) -> Tensor:
        # backward integral of squared impulse response
        x = torch.cumulative_trapezoid(rir.flip(0) ** 2)
        x = x.flip(0)  # backward integration

        edc = 10 * x.log10()  # early decay curve
        edc = edc - torch.max(edc)  # offset to 0 dB
        idx = torch.arange(1, len(edc) + 1)  # time index

        # find decay line
        # find zero-dB point
        xt1 = torch.where(edc <= 0)[0][0]
        if xt1 == []:
            xt1 = 1
        # find -10dB point (T20)
        xt2 = torch.where(edc <= -20)[0][0]
        if xt2 == []:
            xt2 = torch.min(edc)

        # linear fitting
        def linearfit(sta, end, EDC):
            I_xT = torch.arange(sta, end + 1)
            I_xT = I_xT.reshape(I_xT.numel(), 1)
            A = I_xT ** torch.arange(1, -1, -1.0)
            p = torch.linalg.inv(A.T @ A) @ (A.T @ EDC[I_xT])
            fittedline = polyval(p, idx)
            return fittedline

        fitted_line = linearfit(xt1, xt2, edc)
        fitted_line = fitted_line - torch.max(fitted_line)  # offset to 0 dB

        try:
            if torch.where(fitted_line <= -18.2)[0].numel() == 0:
                # for the case that -60dB point is not found using T10 fitting, instead using T20 fitting
                xt1 = torch.where(edc <= 0)[0][0]
                xt2 = torch.where(edc <= -10)[0][0]
                fitted_line = linearfit(xt1, xt2, edc)
                fitted_line = fitted_line - torch.max(fitted_line)  # normalize to 0dB
                xt_60dB = 3.3 * torch.where(fitted_line <= -18.2)[0][0]
            else:
                xt_60dB = 3.3 * torch.where(fitted_line <= -18.2)[0][0]
        except:  # noqa: E722
            raise ValueError("TR is not found.")

        rt = xt_60dB / fs
        rt = rt.round(decimals=3)  # round to 3 decimal places

        if self.verbose:
            return rt.float(), edc.float(), fitted_line.float()
        else:
            return rt.float()


class ReverberationTimeT30(nn.Module):
    """Reverb Time (T30) calculation"""

    def __init__(self, verbose: Optional[bool] = False):
        super(ReverberationTimeT30, self).__init__()
        self.verbose = verbose

    def forward(self, rir: Tensor, fs: int = 16000) -> Tensor:
        # backward integral of squared impulse response
        x = torch.cumulative_trapezoid(rir.flip(0) ** 2)
        x = x.flip(0)  # backward integration

        edc = 10 * x.log10()  # early decay curve
        edc = edc - torch.max(edc)  # offset to 0 dB
        idx = torch.arange(1, len(edc) + 1)  # time index

        # find decay line
        # find zero-dB point
        xt1 = torch.where(edc <= 0)[0][0]
        if xt1 == []:
            xt1 = 1
        # find -30dB point (T30)
        xt2 = torch.where(edc <= -30)[0][0]
        if xt2 == []:
            xt2 = torch.min(edc)

        # linear fitting
        def linearfit(sta, end, EDC):
            I_xT = torch.arange(sta, end + 1)
            I_xT = I_xT.reshape(I_xT.numel(), 1)
            A = I_xT ** torch.arange(1, -1, -1.0)
            p = torch.linalg.inv(A.T @ A) @ (A.T @ EDC[I_xT])
            fittedline = polyval(p, idx)
            return fittedline

        fitted_line = linearfit(xt1, xt2, edc)
        fitted_line = fitted_line - torch.max(fitted_line)  # offset to 0 dB

        try:
            if torch.where(fitted_line <= -30)[0].numel() == 0:
                # for the case that -60dB point is not found using T30 fitting, instead using T10 fitting
                xt1 = torch.where(edc <= -0)[0][0]
                xt2 = torch.where(edc <= -20)[0][0]
                fitted_line = linearfit(xt1, xt2, edc)
                fitted_line = fitted_line - torch.max(fitted_line)
                xt_30dB = 3.3 * torch.where(fitted_line <= -18.2)[0][0]
            else:
                xt_30dB = 2.1 * torch.where(fitted_line <= -28.6)[0][0]
        except:  # noqa: E722
            raise ValueError("T30 is not found.")

        rt30 = (xt_30dB / fs).round(decimals=3)

        if self.verbose:
            return rt30.float(), edc.float(), fitted_line.float()
        else:
            return rt30.float()


class Clarity(nn.Module):
    """Clarity (C80/C50) calculation"""

    def __init__(self, clarity_mode: str = "C80"):
        super(Clarity, self).__init__()
        self.clarity_mode = clarity_mode

    def forward(self, rir: Tensor, fs: int = 16000) -> Tensor:
        t_correction = 2.5 / 1000 * fs  # bias correction, 2.5 ms
        if self.clarity_mode == "C80":
            t_duration = int(80 / 1000 * fs)
        elif self.clarity_mode == "C50":
            t_duration = int(50 / 1000 * fs)

        # find the time index of direct sound
        rir = rir / torch.max(torch.abs(rir))  # normalize to 1
        peak = torch.where(rir**2 == torch.max(rir**2))[0]
        t_peak = peak[0]

        # dens and nums of C80 definition
        start_idx = int(max(0, t_peak - t_correction))  # elimation of direct sound
        # early reflection
        nums = torch.trapz(rir[start_idx : t_peak + t_duration] ** 2)
        # late reflection
        dens = torch.trapz(rir[t_peak + t_duration :] ** 2)

        clarity = 10 * (nums / dens).log10()  # dB

        return clarity


class Definition(nn.Module):
    """D50 calculation"""

    def __init__(self):
        super(Definition, self).__init__()

    def forward(self, rir: Tensor, fs: int = 16000) -> Tensor:
        t_correction = int(2.5 / 1000 * fs)  # bias correction, 2.5 ms
        t50 = int(50 / 1000 * fs)  # 50 ms

        # find the time index of direct sound
        rir = rir / torch.max(torch.abs(rir))  # normalize to 1
        peak = torch.where(rir**2 == torch.max(rir**2))[0]
        t_peak = peak[0]

        # dens and nums of D50 definition
        start_idx = int(max(0, t_peak - t_correction))
        nums = torch.trapz(rir[start_idx : t_peak + t50] ** 2)
        dens = torch.trapz(rir[start_idx:] ** 2)

        d50 = (nums / dens) * 100  # percentage

        return d50


class CenterTime(nn.Module):
    """Center Time (Ts) calculation"""

    def __init__(self):
        super(CenterTime, self).__init__()

    def forward(self, rir: Tensor, fs: int = 16000) -> Tensor:
        rir = rir / torch.max(torch.abs(rir))  # normalize to 1
        t = torch.arange(0, len(rir)) / fs
        nums = torch.trapz(t * rir**2)
        dens = torch.trapz(rir**2)

        Ts = nums / dens

        return Ts
