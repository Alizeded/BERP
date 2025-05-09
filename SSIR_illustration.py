import os

import matplotlib.pyplot as plt  # type: ignore
import matplotlib.ticker as ticker  # type: ignore
import numpy as np
import pandas as pd
import rootutils
import torch
import torchaudio

from src.utils import envelope as env  # noqa: E999

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.AcousticParameterUtils import (  # noqa: E402
    EarlyDecayTime,
    ReverberationTime,
    SparseStochasticIRModel,
)

rir_dataset_path = "./data/RIR_aggregated/RIR.data/"
rir_manifest_ID_resampled = pd.read_csv(
    "./data/RIR_aggregated/RIR.metadata/RIRLabelAugmentV2.csv"
)
rir_sample_path = rir_manifest_ID_resampled.iloc[22526]["RIR"]
rir_sample, fs = torchaudio.load(os.path.join(rir_dataset_path, rir_sample_path))
rir_sample = rir_sample.reshape(-1)
mu = rir_manifest_ID_resampled["Th"].mean().round(4)
Th = rir_manifest_ID_resampled.iloc[22526]["Th"]
Tt = rir_manifest_ID_resampled.iloc[22526]["Tt"]
print("Ti: ", Th, "Td: ", Tt, "mu: ", mu)
# Ti:  0.033 Td:  1.506 mu:  0.0399

ssir = SparseStochasticIRModel(Ti=Th, Td=Tt, volume=2000, mu=mu, fs=16000)
rir_syn = ssir()

# extract envelope
envelope_extract = env.TemporalEnvelope(dim=0, fs=16000, fc=20, mode="TAE")
rir_sample_env = envelope_extract(rir_sample) / torch.max(envelope_extract(rir_sample))
rir_syn_env = envelope_extract(rir_syn) / torch.max(envelope_extract(rir_syn))

rmse = (
    torch.mean((rir_sample_env[: len(rir_syn_env)] - rir_syn_env) ** 2).sqrt().tolist()
)
rmse = round(rmse, 3)
# rmse: 0.080


# ! Illustration of SSIR

t_rir = torch.linspace(0, len(rir_sample) / fs, len(rir_sample))
t_synrir = torch.linspace(0, len(rir_syn) / fs, len(rir_syn))
rir_sample_norm = rir_sample / torch.max(rir_sample)

fig, ax = plt.subplots(2, 1)
plt.rcParams["font.family"] = "serif"
ax[0].plot(t_rir, rir_sample_env)
ax[0].plot(t_synrir, rir_syn_env)
ax[0].set_ylabel("Temporal Envelope", fontsize=14, fontname="serif")
ax[0].legend(["Realistic RIR", "Synthetic RIR"], loc="best", fontsize="12")
ax[0].grid(True)
ax[0].set_xlim([0, 1.5])
xticks = np.arange(0, 1.6, 0.25)
yticks_0 = np.arange(0, 1.1, 0.5)
ax[0].set_xticks(xticks)
ax[0].set_yticks(yticks_0)
ax[0].text(0.65, 0.15, f"RMSE: {str(rmse)}", transform=ax[0].transAxes)
ax[1].plot(t_rir, rir_sample_norm, color="k")
ax[1].set_xlabel("Time [s]", fontsize=14, fontname="serif")
ax[1].set_ylabel("Amplitude", fontsize=14, fontname="serif")
ax[1].set_xlim([0, 1.5])
ax[1].set_ylim([-1, 1])
ax[1].set_xticks(xticks)
yticks_1 = np.arange(-1.0, 1.1, 0.5)
ax[1].set_yticks(yticks_1)
fig.savefig(
    "./data/Figure/illustration_SSIR.pdf",
    format="pdf",
)

# ! Illustration of TR estimation

TR_estimator = ReverberationTime(verbose=True)
EDT_estimator = EarlyDecayTime(verbose=True)

TR, EDC, fittedline_TR = TR_estimator(rir_sample, fs)
EDT, _, fittedline_EDT = EDT_estimator(rir_sample, fs)

fig, ax = plt.subplots(1, 1)
plt.rcParams["font.family"] = "serif"
t_rir = torch.linspace(0, len(rir_sample) / fs, len(rir_sample))
t_EDC = torch.linspace(0, len(EDC) / fs, len(EDC))
rir_sqrd = 10 * (rir_sample**2).log10()  # squared rir
rir_sqrd = rir_sqrd - torch.max(rir_sqrd)  # normalize to 0 dB
ax.plot(t_rir, rir_sqrd, color="grey")
ax.plot(t_EDC, EDC, linewidth=1.2)
ax.plot(t_EDC, fittedline_TR, linewidth=1.5, linestyle="--")
ax.plot(t_EDC, fittedline_EDT, linewidth=1.5, linestyle="-.")
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
ax.plot([t_EDC[0], t_EDC[-1]], [-60, -60], color="k", linewidth=1.5)
ax.plot(
    [t_EDC[0], t_EDC[-1]], [-5, -5], color="darkgrey", linewidth=1.5, linestyle="--"
)
ax.plot(
    [t_EDC[0], t_EDC[-1]], [-10, -10], color="darkgrey", linewidth=1.5, linestyle="--"
)
ax.plot(
    [t_EDC[0], t_EDC[-1]], [-20, -20], color="darkgrey", linewidth=1.5, linestyle="--"
)
ax.plot(
    [t_EDC[0], t_EDC[-1]], [-35, -35], color="darkgrey", linewidth=1.5, linestyle="--"
)
ax.grid(False)
ax.legend(
    ["Squared RIR", "Eenergy Decay Curve", "$T_{60}$ Fitted Line", "EDT Fitted Line"],
    loc="best",
    fontsize="12",
)
ax.set_xlabel("Time [s]", fontsize=16, fontname="serif")
ax.set_ylabel("Magnitude [dB]", fontsize=16, fontname="serif")
ax.set_xlim([0, 2])
ax.set_yticks([-70, -60, -50, -40, -35, -30, -20, -10, -5])
ax.set_ylim([-70, 0])
ax.legend(
    ["Squared RIR", "Eenergy Decay Curve", "$T_{60}$ Fitted Line", "EDT Fitted Line"],
    loc="best",
)
fig.savefig(
    "./data/Figure/illustration_TREstimation.pdf",
    format="pdf",
)
