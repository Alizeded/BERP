import torch
import pandas as pd
import seaborn as sns
import rootutils
import ast
import matplotlib.pyplot as plt

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.unpack import agg_rpp_pred  # noqa: E402


def std_error_calculation(x, y):
    return round(torch.std((x - y).abs()).item(), 4)


# load gammatone model predictions
rap_evaluation = torch.load(
    "./logs/evaluation_rap_joint/runs/${time_stamp}/evaluation_rap_joint_mfcc.pt"
)

rpp_prediction = torch.load(
    "./logs/prediction_joint/runs/${time_stamp}/predictions_joint_mfcc.pt"
)

rpp_ground_truth = pd.read_csv("./data/noiseReverbSpeech/test_manifest.csv")

# aggregate predictions for rpp
volume_hat = agg_rpp_pred(rpp_prediction, param="volume_hat")["volume_hat"]
dist_src_hat = agg_rpp_pred(rpp_prediction, param="dist_src_hat")["dist_src_hat"]
azimuth_src_hat = agg_rpp_pred(rpp_prediction, param="azimuth_src_hat")[
    "azimuth_src_hat"
]
elevation_src_hat = agg_rpp_pred(rpp_prediction, param="elevation_src_hat")[
    "elevation_src_hat"
]

volume = torch.tensor(rpp_ground_truth["volume_log10"], dtype=torch.float32)
dist_src = torch.tensor(rpp_ground_truth["distRcv"], dtype=torch.float32)
ori_src = rpp_ground_truth["oriSrc"].apply(lambda x: ast.literal_eval(x))
azimuth_src = torch.tensor(
    ori_src.apply(lambda x: x[0] * torch.pi), dtype=torch.float32
)
elevation_src = torch.tensor(
    ori_src.apply(lambda x: x[1] * torch.pi), dtype=torch.float32
)

# unpack the pairs of rap evaluation
STI_hat, STI = rap_evaluation["STI_pair"]
ALcons_hat, ALcons = rap_evaluation["ALcons_pair"]
TR_hat, TR = rap_evaluation["TR_pair"]
EDT_hat, EDT = rap_evaluation["EDT_pair"]
C80_hat, C80 = rap_evaluation["C80_pair"]
C50_hat, C50 = rap_evaluation["C50_pair"]
D50_hat, D50 = rap_evaluation["D50_pair"]
Ts_hat, Ts = rap_evaluation["Ts_pair"]

# # plot jointplot by seaborn
df_plot = pd.DataFrame(
    {
        "STI": STI,
        "STI_hat": STI_hat,
        "ALcons": ALcons,
        "ALcons_hat": ALcons_hat,
        "TR": TR,
        "TR_hat": TR_hat,
        "EDT": EDT,
        "EDT_hat": EDT_hat,
        "C80": C80,
        "C80_hat": C80_hat,
        "C50": C50,
        "C50_hat": C50_hat,
        "D50": D50,
        "D50_hat": D50_hat,
        "Ts": Ts,
        "Ts_hat": Ts_hat,
        "volume": volume,
        "volume_hat": volume_hat,
        "dist_src": dist_src,
        "dist_src_hat": dist_src_hat,
        "azimuth_src": azimuth_src,
        "azimuth_src_hat": azimuth_src_hat,
        "elevation_src": elevation_src,
        "elevation_src_hat": elevation_src_hat,
    }
)

sns.set_style("white", {"axes.grid": True})
sns.set_style({"font.family": "serif"}, {"font.serif": ["Times New Roman"]})
sns.color_palette("viridis", as_cmap=True)
fig, ax = plt.subplots(3, 4, figsize=(20, 15))

# STI plot
diag_STI = torch.linspace(0, 1, 2000)
ax[0, 0] = sns.scatterplot(
    data=df_plot,
    x=df_plot["STI"].sample(n=200, random_state=2036),
    y=df_plot["STI_hat"].sample(n=200, random_state=2036),
    color="mediumslateblue",
    marker="H",
    ax=ax[0, 0],
)
ax[0, 0].set_xlim(0, 1)
ax[0, 0].set_ylim(0, 1)
ax[0, 0].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax[0, 0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax[0, 0].set_xlabel("Ground-truth STI")
ax[0, 0].set_ylabel("Estimated STI")
ax[0, 0].plot(diag_STI, diag_STI, color="grey", linestyle="--")

# ALcons plot
diag_ALcons = torch.linspace(0, 50, 2000)
ax[0, 1] = sns.scatterplot(
    data=df_plot,
    x=df_plot["ALcons"].sample(n=200, random_state=2036),
    y=df_plot["ALcons_hat"].sample(n=200, random_state=2036),
    color="cornflowerblue",
    marker="H",
    ax=ax[0, 1],
)
ax[0, 1].set_xlim(0, 50)
ax[0, 1].set_ylim(0, 50)
ax[0, 1].set_xticks([0, 10, 20, 30, 40, 50])
ax[0, 1].set_yticks([0, 10, 20, 30, 40, 50])
ax[0, 1].set_xlabel("Ground-truth $\%AL_{cons}$")
ax[0, 1].set_ylabel("Estimated $\%AL_{cons}$")
ax[0, 1].plot(diag_ALcons, diag_ALcons, color="grey", linestyle="--")

# TR plot
diag_TR = torch.linspace(0, 8.0, 2000)
ax[0, 2] = sns.scatterplot(
    data=df_plot,
    x=df_plot["TR"].sample(n=200, random_state=2036),
    y=df_plot["TR_hat"].sample(n=200, random_state=2036),
    color="coral",
    marker="H",
    ax=ax[0, 2],
)
ax[0, 2].set_xlim(0, 8)
ax[0, 2].set_ylim(0, 8)
ax[0, 2].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
ax[0, 2].set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
ax[0, 2].set_xlabel("Ground-truth $T_R$")
ax[0, 2].set_ylabel("Estimated $T_R$")
ax[0, 2].plot(diag_TR, diag_TR, color="grey", linestyle="--")

# EDT plot
diag_EDT = torch.linspace(0, 6.0, 2000)
ax[0, 3] = sns.scatterplot(
    data=df_plot,
    x=df_plot["EDT"].sample(n=200, random_state=2036),
    y=df_plot["EDT_hat"].sample(n=200, random_state=2036),
    color="yellowgreen",
    marker="H",
    ax=ax[0, 3],
)
ax[0, 3].set_xlim(0, 6)
ax[0, 3].set_ylim(0, 6)
ax[0, 3].set_xticks([0, 1, 2, 3, 4, 5, 6])
ax[0, 3].set_yticks([0, 1, 2, 3, 4, 5, 6])
ax[0, 3].set_xlabel("Ground-truth EDT")
ax[0, 3].set_ylabel("Estimated EDT")
ax[0, 3].plot(diag_EDT, diag_EDT, color="grey", linestyle="--")

# C80 plot
diag_C80 = torch.linspace(-10, 30, 2000)
ax[1, 0] = sns.scatterplot(
    data=df_plot,
    x=df_plot["C80"].sample(n=200, random_state=2036),
    y=df_plot["C80_hat"].sample(n=200, random_state=2036),
    color="sandybrown",
    marker="H",
    ax=ax[1, 0],
)
ax[1, 0].set_xlim(-10, 30)
ax[1, 0].set_ylim(-10, 30)
ax[1, 0].set_xticks([-10, -5, 0, 5, 10, 15, 20, 25, 30])
ax[1, 0].set_yticks([-10, -5, 0, 5, 10, 15, 20, 25, 30])
ax[1, 0].set_xlabel("Ground-truth $C_{80}$")
ax[1, 0].set_ylabel("Estimated $C_{80}$")
ax[1, 0].plot(diag_C80, diag_C80, color="grey", linestyle="--")

# C50 plot
diag_C50 = torch.linspace(-10, 25, 2000)
ax[1, 1] = sns.scatterplot(
    data=df_plot,
    x=df_plot["C50"].sample(n=200, random_state=2036),
    y=df_plot["C50_hat"].sample(n=200, random_state=2036),
    alpha=0.5,
    color="darkorange",
    marker="H",
    ax=ax[1, 1],
)
ax[1, 1].set_xlim(-10, 25)
ax[1, 1].set_ylim(-10, 25)
ax[1, 1].set_xticks([-10, -5, 0, 5, 10, 15, 20, 25])
ax[1, 1].set_yticks([-10, -5, 0, 5, 10, 15, 20, 25])
ax[1, 1].set_xlabel("Ground-truth $C_{50}$")
ax[1, 1].set_ylabel("Estimated $C_{50}$")
ax[1, 1].plot(diag_C50, diag_C50, color="grey", linestyle="--")

# D50 plot
diag_D50 = torch.linspace(0, 100, 2000)
ax[1, 2] = sns.scatterplot(
    data=df_plot,
    x=df_plot["D50"].sample(n=200, random_state=2036),
    y=df_plot["D50_hat"].sample(n=200, random_state=2036),
    alpha=0.5,
    color="khaki",
    marker="H",
    ax=ax[1, 2],
)
ax[1, 2].set_xlim(0, 100)
ax[1, 2].set_ylim(0, 100)
ax[1, 2].set_xticks([0, 15, 30, 45, 60, 75, 90, 100])
ax[1, 2].set_yticks([0, 15, 30, 45, 60, 75, 90, 100])
ax[1, 2].set_xlabel("Ground-truth $D_{50}$")
ax[1, 2].set_ylabel("Estimated $D_{50}$")
ax[1, 2].plot(diag_D50, diag_D50, color="grey", linestyle="--")

# Ts plot
diag_Ts = torch.linspace(0, 0.45, 2000)
ax[1, 3] = sns.scatterplot(
    data=df_plot,
    x=df_plot["Ts"].sample(n=200, random_state=2036),
    y=df_plot["Ts_hat"].sample(n=200, random_state=2036),
    color="lightseagreen",
    marker="H",
    ax=ax[1, 3],
)
ax[1, 3].set_xlim(0, 0.45)
ax[1, 3].set_ylim(0, 0.45)
ax[1, 3].set_xticks([0, 0.05, 0.15, 0.25, 0.35, 0.45])
ax[1, 3].set_yticks([0, 0.05, 0.15, 0.25, 0.35, 0.45])
ax[1, 3].set_xlabel("Ground-truth $T_s$")
ax[1, 3].set_ylabel("Estimated $T_s$")
ax[1, 3].plot(diag_Ts, diag_Ts, color="grey", linestyle="--")

# volume plot
diag_volume = torch.linspace(1, 4, 2000)
ax[2, 0] = sns.scatterplot(
    data=df_plot,
    x=df_plot["volume"].sample(n=200, random_state=2036),
    y=df_plot["volume_hat"].sample(n=200, random_state=2036),
    color="deepskyblue",
    marker="H",
    ax=ax[2, 0],
)
ax[2, 0].set_xlim(1, 4)
ax[2, 0].set_ylim(1, 4)
ax[2, 0].set_xticks([1, 1.5, 2, 2.5, 3, 3.5, 4])
ax[2, 0].set_yticks([1, 1.5, 2, 2.5, 3, 3.5, 4])
ax[2, 0].set_xlabel("Ground-truth $V$")
ax[2, 0].set_ylabel("Estimated $V$")
ax[2, 0].plot(diag_volume, diag_volume, color="grey", linestyle="--")

# dist_src plot
diag_dist_src = torch.linspace(0, 30, 2000)
ax[2, 1] = sns.scatterplot(
    data=df_plot,
    x=df_plot["dist_src"].sample(n=200, random_state=2036),
    y=df_plot["dist_src_hat"].sample(n=200, random_state=2036),
    color="mediumseagreen",
    marker="H",
    ax=ax[2, 1],
)
ax[2, 1].set_xlim(0, 30)
ax[2, 1].set_ylim(0, 30)
ax[2, 1].set_xticks([0, 5, 10, 15, 20, 25, 30])
ax[2, 1].set_yticks([0, 5, 10, 15, 20, 25, 30])
ax[2, 1].set_xlabel("Ground-truth $D$")
ax[2, 1].set_ylabel("Estimated $D$")
ax[2, 1].plot(diag_dist_src, diag_dist_src, color="grey", linestyle="--")

# azimuth_src plot
diag_azimuth_src = torch.linspace(-3.14, 3.14, 2000)
ax[2, 2] = sns.scatterplot(
    data=df_plot,
    x=df_plot["azimuth_src"].sample(n=200, random_state=2036),
    y=df_plot["azimuth_src_hat"].sample(n=200, random_state=2036),
    color="mediumorchid",
    marker="H",
    ax=ax[2, 2],
)
ax[2, 2].set_xlim(-3.14, 3.14)
ax[2, 2].set_ylim(-3.14, 3.14)
ax[2, 2].set_xticks([-3.14, -2.36, -1.57, -0.79, 0, 0.79, 1.57, 2.36, 3.14])
ax[2, 2].set_yticks([-3.14, -2.36, -1.57, -0.79, 0, 0.79, 1.57, 2.36, 3.14])
ax[2, 2].set_xlabel("Ground-truth $\\theta$")
ax[2, 2].set_ylabel("Estimated $\\theta$")
ax[2, 2].plot(diag_azimuth_src, diag_azimuth_src, color="grey", linestyle="--")

# elevation_src plot
diag_elevation_src = torch.linspace(-3.14 / 2, 3.14 / 2, 2000)
ax[2, 3] = sns.scatterplot(
    data=df_plot,
    x=df_plot["elevation_src"].sample(n=200, random_state=2036),
    y=df_plot["elevation_src_hat"].sample(n=200, random_state=2036),
    color="crimson",
    marker="H",
    ax=ax[2, 3],
)
ax[2, 3].set_xlim(-3.14 / 2, 3.14 / 2)
ax[2, 3].set_ylim(-3.14 / 2, 3.14 / 2)
ax[2, 3].set_xticks([-3.14 / 2, -3.14 / 4, 0, 3.14 / 4, 3.14 / 2])
ax[2, 3].set_yticks([-3.14 / 2, -3.14 / 4, 0, 3.14 / 4, 3.14 / 2])
ax[2, 3].set_xlabel("Ground-truth $\psi$")
ax[2, 3].set_ylabel("Estimated $\psi$")
ax[2, 3].plot(diag_elevation_src, diag_elevation_src, color="grey", linestyle="--")

# save the plot as pdf
fig.savefig(
    "./data/Figure/roomparam_est_visualization.pdf",
    format="pdf",
)
