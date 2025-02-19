import ast
import os
import random
from pathlib import Path
from re import sub

import numpy as np
import pandas as pd
import rootutils
import scipy as sp
import sklearn.model_selection as skms
import torch
import torchaudio

import src.utils.envelope as env

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.AcousticParameterUtils import (  # noqa: E402
    CenterTime,
    Clarity,
    Definition,
    EarlyDecayTime,
    PercentageArticulationLoss,
    RapidSpeechTransmissionIndex,
)

random.seed(42)


def polyval(p, x) -> torch.Tensor:
    p = torch.as_tensor(p)
    if p.ndim == 0 or (len(p) < 1):
        return torch.zeros_like(x)
    y = torch.zeros_like(x)
    for p_i in p:
        y = y * x + p_i
    return y


# ------------------- T60 estimation ------------------- #
def RIRestT60(h, fs, opMod=False):
    """
    Estimate reverberation time (T60) by Schroeder's back integral method
    via using energy decay curve until -20dB (T20).
    """
    # --------------- Schroeder's backward integration method --------------- #
    x = torch.cumulative_trapezoid(h.flip(0) ** 2).double()
    x = x.flip(0)  # backward integration

    EDC = 10 * torch.log10(x)  # energy decay curve
    EDC = EDC - torch.max(EDC)  # offset to 0dB
    I = torch.arange(1, len(EDC) + 1)  # time vector  # noqa: E741

    # -------------------- find decay line --------------------- #
    # find zero-dB point
    xT1 = torch.where(EDC <= 0)[0][0]
    if xT1 == []:
        xT1 = 1

    # find -20dB point (T20)
    xT2 = torch.where(EDC <= -20)[0][0]
    if xT2 == []:
        xT2 = torch.min(EDC)

    # linear fitting
    def linearfit(sta, end, EDC):
        I_xT = torch.arange(sta, end + 1)
        I_xT = I_xT.reshape(I_xT.numel(), 1)
        A = I_xT ** torch.arange(1, -1, -1.0).double()
        p = torch.linalg.inv(A.T @ A) @ (A.T @ EDC[I_xT])
        fittedline = polyval(p, I)
        return fittedline

    fittedline = linearfit(xT1, xT2, EDC)
    fittedline = fittedline - torch.max(fittedline)  # normalize to 0dB

    # linear extrapolation to -60dB point
    # print(torch.where(fittedline <= -18.2)[0])
    try:
        if torch.where(fittedline <= -18.2)[0].numel() == 0:
            # for the case that -60dB point is not found using T20 fitting, instead using T10 fitting
            xT2 = torch.where(EDC <= -10)[0][0]
            fittedline = linearfit(xT1, xT2, EDC)
            fittedline = fittedline - torch.max(fittedline)  # normalize to 0dB
            xT_60 = 3.3 * torch.where(fittedline <= -18.2)[0][0]
        else:
            xT_60 = 3.3 * torch.where(fittedline <= -18.2)[0][0]
    except:  # noqa: E722
        print("T60 does not exist, the signal is not an RIR.")
    RT = xT_60 / fs
    RT = RT.round(decimals=3)

    if opMod:
        return RT, EDC, fittedline
    else:
        return RT


# ----------------- Th and Tt estimation ----------------- #
def RIRestThTt(h, fs, opMod=False):
    """
    Estimate Th and Tt via exponential fitting and linear fitting
    respectively for extended RIR model
    """
    # ----------------- temporal amplitude envelope ----------------- #
    fc = 20
    temporal_env = env.TemporalEnvelope(dim=0, fc=fc, fs=fs, mode="TAE")
    eh = temporal_env(h)
    torch.arange(0, len(eh)) / fs  # time vector

    # -------------- parameter estimation via fitting -------------- #
    # t0
    Pks = torch.max(eh)
    I_t0 = torch.argmax(eh)
    t0 = I_t0 / fs
    # Th

    riseside = eh[:I_t0]
    xT_Th = torch.arange(0, len(riseside)) / fs  # time vector
    try:
        # construct the objective function
        def objfunc(x, *params):
            return params[0] * torch.exp(6.9 * x / params[1])

        param_init = [0.01, 0.01]
        fit, err = sp.optimize.curve_fit(
            objfunc,
            xT_Th,
            riseside,
            p0=param_init,
            bounds=([0.0, Pks], [0.01, t0]),
            method="trf",
        )
    except:  # noqa: E722
        fit = [0.0000, t0]  # Th does not exist
        # print("Th does not exist. Set to t0.")

    torch.as_tensor(fit[0]).round(decimals=3)
    Th = torch.as_tensor(fit[1]).round(decimals=3)

    # Tt
    Tt = RIRestT60(h, fs)

    # Gain
    a = torch.max(h).double().round(decimals=4)  # float64 dtype

    # eh_fit
    t_Th = torch.arange(-len(riseside), 0) / fs  # time vector
    yf_Th = torch.exp(6.9 * t_Th / Th)
    xT_Tt = torch.arange(0, len(eh) - I_t0) / fs  # time vector
    yf_Tt = torch.exp(-6.9 * xT_Tt / Tt)
    eh_fit = Pks * torch.cat((yf_Th, yf_Tt))

    if opMod:
        return Th, Tt, a, eh, eh_fit
    else:
        return Th, Tt, a


# ----------------- read RIRs ------------------- #


def micPos_Arni(posNo):
    """
    According to the microphone position in Arni RIR database, calculate the
    relative distance between the microphone and the source

    """
    match posNo:
        case 1:
            d = (890 - 240 - 175) ** 2 + (200 - 180) ** 2 + (135 - 143) ** 2
            d = torch.sqrt(torch.as_tensor(d)) / 100  # convert to meter
        case 2:
            d = (890 - 240 - 200) ** 2 + (630 - 149 - 180) ** 2 + (139 - 143) ** 2
            d = torch.sqrt(torch.as_tensor(d)) / 100
        case 3:
            d = (445 - 240) ** 2 + (630 - 250 - 180) ** 2 + (144 - 143) ** 2
            d = torch.sqrt(torch.as_tensor(d)) / 100
        case 4:
            d = (130 - 240) ** 2 + (315 - 180) ** 2 + (154 - 143) ** 2
            d = torch.sqrt(torch.as_tensor(d)) / 100
        case 5:
            d = (381 - 240) ** 2 + (145 - 180) ** 2 + (154 - 143) ** 2
            d = torch.sqrt(torch.as_tensor(d)) / 100

    return d.round(decimals=3)


def foldercheck_Arni(folderNo):
    match folderNo:
        case 1:
            path = "/IR_Arni_upload_numClosed_0-5/"
        case 2:
            path = "/IR_Arni_upload_numClosed_6-15/"
        case 3:
            path = "/IR_Arni_upload_numClosed_16-25/"
        case 4:
            path = "/IR_Arni_upload_numClosed_26-35/"
        case 5:
            path = "/IR_Arni_upload_numClosed_36-45/"
        case 6:
            path = "/IR_Arni_upload_numClosed_46-55/"
    return path


def micOri_Arni(posNo):
    match posNo:
        case 1:
            adjacent_a = torch.tensor(890 - 240 - 175)
            opposite_a = torch.tensor(200 - 180)
            azimuth = (torch.arctan(opposite_a / adjacent_a) / torch.pi).reshape(1)
            adjacent_e = torch.sqrt(
                torch.tensor((890 - 240 - 175) ** 2 + (200 - 180) ** 2)
            )
            opposite_e = torch.tensor(135 - 143)
            elevation = (torch.arctan(opposite_e / adjacent_e) / torch.pi).reshape(1)
            ori = torch.cat((azimuth, elevation))  # radian unit
        case 2:
            adjacent_a = torch.tensor(890 - 240 - 200)
            opposite_a = torch.tensor(630 - 149 - 180)
            azimuth = (torch.arctan(opposite_a / adjacent_a) / torch.pi).reshape(1)
            adjacent_e = torch.sqrt(
                torch.tensor((890 - 240 - 200) ** 2 + (630 - 149 - 180) ** 2)
            )
            opposite_e = torch.tensor(139 - 143)
            elevation = (torch.arctan(opposite_e / adjacent_e) / torch.pi).reshape(1)
            ori = torch.cat((azimuth, elevation))  # radian unit
        case 3:
            adjacent_a = torch.tensor(445 - 240)
            opposite_a = torch.tensor(630 - 250 - 180)
            azimuth = (torch.arctan(opposite_a / adjacent_a) / torch.pi).reshape(1)
            adjacent_e = torch.sqrt(
                torch.tensor((445 - 240) ** 2 + (630 - 250 - 180) ** 2)
            )
            opposite_e = torch.tensor(144 - 143)
            elevation = (torch.arctan(opposite_e / adjacent_e) / torch.pi).reshape(1)
            ori = torch.cat((azimuth, elevation))  # radian unit
        case 4:
            adjacent_a = torch.tensor(130 - 240)
            opposite_a = torch.tensor(315 - 180)
            azimuth = (torch.arctan(opposite_a / adjacent_a) / torch.pi).reshape(1)
            adjacent_e = torch.sqrt(torch.tensor((130 - 240) ** 2 + (315 - 180) ** 2))
            opposite_e = torch.tensor(154 - 143)
            elevation = (torch.arctan(opposite_e / adjacent_e) / torch.pi).reshape(1)
            ori = torch.cat((azimuth, elevation))  # radian unit
        case 5:
            adjacent_a = torch.tensor(381 - 240)
            opposite_a = torch.tensor(145 - 180)
            azimuth = (torch.arctan(opposite_a / adjacent_a) / torch.pi).reshape(1)
            adjacent_e = torch.sqrt(torch.tensor((381 - 240) ** 2 + (145 - 180) ** 2))
            opposite_e = torch.tensor(154 - 143)
            elevation = (torch.arctan(opposite_e / adjacent_e) / torch.pi).reshape(1)
            ori = torch.cat((azimuth, elevation))  # radian unit
    return ori.round(decimals=3)


def volume_Arni():
    return torch.tensor(
        [8.9, 6.3, 3.6]
    )  # volume of the room in m^3, [length, width, height]


def room_ID_Arni():
    return "rmID_1"


def readRIR_Arni(commonpath, savepath, csv_savemode="w"):
    """
    Read RIR from Arni RIR database from No.1 to No.6 folder
    """
    # folder = commonpath + path
    i = 0
    ThTtdistRcvOriSrc_label = []
    for folderNo in range(1, 7):
        path = foldercheck_Arni(folderNo)
        folder = commonpath + path
        for filename in os.listdir(folder):
            if filename.endswith("sweep_3.wav"):
                i += 1
                filepath = os.path.join(folder, filename)
                rir, fs = torchaudio.load(filepath)
                rir = rir.reshape(-1)
                if fs != 16000:
                    rir = torchaudio.transforms.Resample(fs, 16000)(rir)
                    fs = 16000
                # print(filepath) # debug
                Th, Tt, _ = RIRestThTt(rir, fs)
                Th, Tt = Th.tolist(), Tt.tolist()  # convert to list type for csv file
                # if  ((Tt <=3.5) and (Tt >= 0.3)) and (Th <= 0.2) and (Th >= 0.01):
                #     chosen_i.append(i)
                V = volume_Arni()
                V = V.tolist()  # convert to list type for csv file
                micNo = int(
                    os.path.basename(filename).split("mic_", 1)[1].split("_sweep", 1)[0]
                )  # int type
                rcv_dist = micPos_Arni(micNo)
                src_ori = micOri_Arni(micNo)  # polar coordinate in radian
                rcv_dist, src_ori = (
                    rcv_dist.tolist(),
                    src_ori.tolist(),
                )  # convert to list type for csv file
                rm_ID = room_ID_Arni()
                ThTtdistRcvOriSrc_label.append(
                    [
                        "Arni_RIR_no" + str(i) + ".wav",
                        rm_ID,
                        Th,
                        Tt,
                        V,
                        rcv_dist,
                        src_ori,
                    ]
                )
                if not os.path.exists(savepath + "/RIR.data"):
                    os.makedirs(savepath + "/RIR.data")
                torchaudio.save(
                    os.path.join(
                        savepath + "/RIR.data", "Arni_RIR_no" + str(i) + ".wav"
                    ),
                    rir.unsqueeze(0),
                    sample_rate=fs,
                    bits_per_sample=16,
                    encoding="PCM_S",
                )

    ThTtdistMic_label = pd.DataFrame(
        ThTtdistRcvOriSrc_label,
        columns=["RIR", "roomID", "Th", "Tt", "volume", "distRcv", "oriSrc"],
    )

    if not os.path.exists(savepath + "/RIR.metadata"):
        os.makedirs(savepath + "/RIR.metadata")

    if csv_savemode == "w":
        ThTtdistMic_label.to_csv(
            savepath + "/RIR.metadata/ThTtDistRcvOriSrc_label.csv",
            mode=csv_savemode,
            index=False,
        )
    elif csv_savemode == "a":
        ThTtdistRcvOriSrc_label.to_csv(
            savepath + "/RIR.metadata/ThTtDistRcvOriSrc_label.csv",
            mode=csv_savemode,
            index=False,
            header=False,
        )

    return print(
        "RIRs saved in "
        + savepath
        + "/RIR.data"
        + " and metadata saved in "
        + savepath
        + "/RIR.metadata"
    )


def micPos_Motus(posNo):
    rcvPos = torch.tensor([1.8250, 2.0750, 1.4480])
    match posNo:
        case 1:  # 1st loudspeaker
            micPos = torch.tensor([3.4620, 2.0750, 1.4480])
            d = torch.sqrt(
                torch.sum((rcvPos - micPos) ** 2)
            )  # distance between mic and receiver
        case 2:  # 2nd loudspeaker
            micPos = torch.tensor([1.7470, 3.7380, 1.4480])
            d = torch.sqrt(torch.sum((rcvPos - micPos) ** 2))
        case 3:  # 3rd loudspeaker
            micPos = torch.tensor([2.4830, 3.2950, 1.4480])
            d = torch.sqrt(torch.sum((rcvPos - micPos) ** 2))
        case 4:  # 4th loudspeaker
            micPos = torch.tensor([3.8810, 3.4370, 1.4480])
            d = torch.sqrt(torch.sum((rcvPos - micPos) ** 2))
    return d.round(decimals=3)


def source_ori_Motus(posNo):
    match posNo:
        case 1:  # 1st ld
            azimuth = torch.tensor(0.0).reshape(1)
            elevation = torch.tensor(0.0).reshape(1)
            ori = torch.cat((azimuth, elevation))  # radian unit
        case 2:  # 2nd ld
            azimuth = torch.tensor(0.5).reshape(1)
            elevation = torch.tensor(0.0).reshape(1)
            ori = torch.cat((azimuth, elevation))  # radian unit
        case 3:  # 3rd ld
            azimuth = torch.tensor(0.33).reshape(1)
            elevation = torch.tensor(0.0).reshape(1)
            ori = torch.cat((azimuth, elevation))  # radian unit
        case 4:  # 4th ld
            azimuth = torch.tensor(0.25).reshape(1)
            elevation = torch.tensor(0.0).reshape(1)
            ori = torch.cat((azimuth, elevation))  # radian unit
    return ori.round(decimals=3)


def volume_Motus():
    return torch.tensor(
        [4.9, 4.4, 2.9]
    )  # volume of the room in m^3, [length, width, height]


def room_ID_Motus():
    return "rmID_2"


def readRIR_Motus(commonpath, savepath, csv_savemode="w"):
    """
    read RIR from Motus RIR database

    """
    raw = "/raw_rirs"
    folder = commonpath + raw
    i = 0
    ThTtDistRcvOriSrc_label = []
    lst = os.listdir(folder)
    lst.sort()
    for filename in lst:
        if filename.endswith("raw_rirs.wav"):
            i += 1
            filepath = os.path.join(folder, filename)
            rir, fs = torchaudio.load(filepath)
            rir = rir[1, :].reshape(
                -1
            )  # randomly choose one-channel RIR from 32-channel RIRs
            if fs != 16000:
                rir = torchaudio.transforms.Resample(fs, 16000)(rir)
                fs = 16000
            Th, Tt, _ = RIRestThTt(rir, fs)
            Th, Tt = Th.tolist(), Tt.tolist()  # convert to list type for csv file
            # print(filepath) # debug
            V = volume_Motus()
            V = V.tolist()  # convert to list type for csv file
            micNo = int(
                os.path.basename(filename).split("_raw_rirs.wav", 1)[0].split("_", 1)[1]
            )  # int type
            rcv_dist = micPos_Motus(micNo)
            src_ori = source_ori_Motus(micNo)
            rcv_dist, src_ori = (
                rcv_dist.tolist(),
                src_ori.tolist(),
            )  # convert to list type for csv file
            # idx = int(os.path.basename(filename).split('_raw_rirs.wav', 1)[0].split('_', 1)[0])
            rm_ID = room_ID_Motus()
            ThTtDistRcvOriSrc_label.append(
                ["Motus_RIR_no" + str(i) + ".wav", rm_ID, Th, Tt, V, rcv_dist, src_ori]
            )
            if not os.path.exists(savepath + "/RIR.data"):
                os.makedirs(savepath + "/RIR.data")
            torchaudio.save(
                os.path.join(savepath + "/RIR.data", "Motus_RIR_no" + str(i) + ".wav"),
                rir.unsqueeze(0),
                sample_rate=fs,
                bits_per_sample=16,
                encoding="PCM_S",
            )

    ThTtDistOri_label = pd.DataFrame(
        ThTtDistRcvOriSrc_label,
        columns=["RIR", "roomID", "Th", "Tt", "volume", "distRcv", "oriSrc"],
    )
    if not os.path.exists(savepath + "/RIR.metadata"):
        os.makedirs(savepath + "/RIR.metadata")

    if csv_savemode == "w":
        ThTtDistOri_label.to_csv(
            savepath + "/RIR.metadata/ThTtDistRcvOriSrc_label.csv",
            mode=csv_savemode,
            index=False,
        )
    elif csv_savemode == "a":
        ThTtDistOri_label.to_csv(
            savepath + "/RIR.metadata/ThTtDistRcvOriSrc_label.csv",
            mode=csv_savemode,
            index=False,
            header=False,
        )
    return print(
        "RIRs saved in "
        + savepath
        + "/RIR.data"
        + " and metadata saved in "
        + savepath
        + "/RIR.metadata"
    )


def checkfolder_BUT(folderNo):
    match folderNo:
        case 1:
            folder = "/Hotel_SkalskyDvur_ConferenceRoom2/MicID01"
        case 2:
            folder = "/Hotel_SkalskyDvur_Room112/MicID01"
        case 3:
            folder = "/VUT_FIT_C236/MicID01"
        case 4:
            folder = "/VUT_FIT_D105/MicID01"
        case 5:
            folder = "/VUT_FIT_E112/MicID01"
        case 6:
            folder = "/VUT_FIT_L207/MicID01"
        case 7:
            folder = "/VUT_FIT_L212/MicID01"
        case 8:
            folder = "/VUT_FIT_L227/MicID01"
        case 9:
            folder = "/VUT_FIT_Q301/MicID01"
    return folder


def volume_BUT(folderNo, commonpath):
    match folderNo:
        case 1:
            path = commonpath + "/Hotel_SkalskyDvur_ConferenceRoom2/env_meta.txt"
            csv_env = pd.read_csv(path, sep="\t", header=None)
            length = csv_env[csv_env[0] == "$EnvDepth"][1].values[0]
            length = torch.tensor(float(length)).round(decimals=1)
            width = csv_env[csv_env[0] == "$EnvWidth"][1].values[0]
            width = torch.tensor(float(width)).round(decimals=1)
            height = csv_env[csv_env[0] == "$EnvHeight"][1].values[0]
            height = torch.tensor(float(height)).round(decimals=1)
            V = torch.cat((length.reshape(1), width.reshape(1), height.reshape(1)))
        case 2:
            path = commonpath + "/Hotel_SkalskyDvur_Room112/env_meta.txt"
            csv_env = pd.read_csv(path, sep="\t", header=None)
            length = csv_env[csv_env[0] == "$EnvDepth"][1].values[0]
            length = torch.tensor(float(length)).round(decimals=1)
            width = csv_env[csv_env[0] == "$EnvWidth"][1].values[0]
            width = torch.tensor(float(width)).round(decimals=1)
            height = csv_env[csv_env[0] == "$EnvHeight"][1].values[0]
            height = torch.tensor(float(height)).round(decimals=1)
            V = torch.cat((length.reshape(1), width.reshape(1), height.reshape(1)))
        case 3:
            path = commonpath + "/VUT_FIT_C236/env_meta.txt"
            csv_env = pd.read_csv(path, sep="\t", header=None)
            length = csv_env[csv_env[0] == "$EnvDepth"][1].values[0]
            length = torch.tensor(float(length)).round(decimals=1)
            width = csv_env[csv_env[0] == "$EnvWidth"][1].values[0]
            width = torch.tensor(float(width)).round(decimals=1)
            height = csv_env[csv_env[0] == "$EnvHeight"][1].values[0]
            height = torch.tensor(float(height)).round(decimals=1)
            V = torch.cat((length.reshape(1), width.reshape(1), height.reshape(1)))
        case 4:
            path = commonpath + "/VUT_FIT_D105/env_meta.txt"
            csv_env = pd.read_csv(path, sep="\t", header=None)
            length = csv_env[csv_env[0] == "$EnvDepth"][1].values[0]
            length = torch.tensor(float(length)).round(decimals=1)
            width = csv_env[csv_env[0] == "$EnvWidth"][1].values[0]
            width = torch.tensor(float(width)).round(decimals=1)
            height = csv_env[csv_env[0] == "$EnvHeight"][1].values[0]
            height = torch.tensor(float(height)).round(decimals=1)
            V = torch.cat((length.reshape(1), width.reshape(1), height.reshape(1)))
        case 5:
            path = commonpath + "/VUT_FIT_E112/env_meta.txt"
            csv_env = pd.read_csv(path, sep="\t", header=None)
            length = csv_env[csv_env[0] == "$EnvDepth"][1].values[0]
            length = torch.tensor(float(length)).round(decimals=1)
            width = csv_env[csv_env[0] == "$EnvWidth"][1].values[0]
            width = torch.tensor(float(width)).round(decimals=1)
            height = csv_env[csv_env[0] == "$EnvHeight"][1].values[0]
            height = torch.tensor(float(height)).round(decimals=1)
            V = torch.cat((length.reshape(1), width.reshape(1), height.reshape(1)))
        case 6:
            path = commonpath + "/VUT_FIT_L207/env_meta.txt"
            csv_env = pd.read_csv(path, sep="\t", header=None)
            length = csv_env[csv_env[0] == "$EnvDepth"][1].values[0]
            length = torch.tensor(float(length)).round(decimals=1)
            width = csv_env[csv_env[0] == "$EnvWidth"][1].values[0]
            width = torch.tensor(float(width)).round(decimals=1)
            height = csv_env[csv_env[0] == "$EnvHeight"][1].values[0]
            height = torch.tensor(float(height)).round(decimals=1)
            V = torch.cat((length.reshape(1), width.reshape(1), height.reshape(1)))
        case 7:
            path = commonpath + "/VUT_FIT_L212/env_meta.txt"
            csv_env = pd.read_csv(path, sep="\t", header=None)
            length = csv_env[csv_env[0] == "$EnvDepth"][1].values[0]
            length = torch.tensor(float(length)).round(decimals=1)
            width = csv_env[csv_env[0] == "$EnvWidth"][1].values[0]
            width = torch.tensor(float(width)).round(decimals=1)
            height = csv_env[csv_env[0] == "$EnvHeight"][1].values[0]
            height = torch.tensor(float(height)).round(decimals=1)
            V = torch.cat((length.reshape(1), width.reshape(1), height.reshape(1)))
        case 8:
            path = commonpath + "/VUT_FIT_L227/env_meta.txt"
            csv_env = pd.read_csv(path, sep="\t", header=None)
            length = csv_env[csv_env[0] == "$EnvDepth"][1].values[0]
            length = torch.tensor(float(length)).round(decimals=1)
            width = csv_env[csv_env[0] == "$EnvWidth"][1].values[0]
            width = torch.tensor(float(width)).round(decimals=1)
            height = csv_env[csv_env[0] == "$EnvHeight"][1].values[0]
            height = torch.tensor(float(height)).round(decimals=1)
            V = torch.cat((length.reshape(1), width.reshape(1), height.reshape(1)))
        case 9:
            path = commonpath + "/VUT_FIT_Q301/env_meta.txt"
            csv_env = pd.read_csv(path, sep="\t", header=None)
            length = csv_env[csv_env[0] == "$EnvDepth"][1].values[0]
            length = torch.tensor(float(length)).round(decimals=1)
            width = csv_env[csv_env[0] == "$EnvWidth"][1].values[0]
            width = torch.tensor(float(width)).round(decimals=1)
            height = csv_env[csv_env[0] == "$EnvHeight"][1].values[0]
            height = torch.tensor(float(height)).round(decimals=1)
            V = torch.cat((length.reshape(1), width.reshape(1), height.reshape(1)))
    return V


def room_ID_BUT(folderNo):
    match folderNo:
        case 1:
            rm_ID = "rmID_3"
        case 2:
            rm_ID = "rmID_4"
        case 3:
            rm_ID = "rmID_5"
        case 4:
            rm_ID = "rmID_6"
        case 5:
            rm_ID = "rmID_7"
        case 6:
            rm_ID = "rmID_8"
        case 7:
            rm_ID = "rmID_9"
        case 8:
            rm_ID = "rmID_10"
        case 9:
            rm_ID = "rmID_11"
    return rm_ID


def readRIR_BUT(commonpath, savepath, csv_savemode="w"):
    """
    read RIR from BUT RIR database

    """
    j = 0
    ThTtDistRcvOriSrc_label = []
    for folderNo in range(1, 10):
        path = commonpath + checkfolder_BUT(folderNo)
        lst = os.listdir(path)
        lst.sort()
        for foldername in lst:
            if foldername.startswith("SpkID"):
                for i in range(1, 32):
                    j += 1
                    rir_path = os.path.join(path, foldername, str(i).zfill(2), "RIR")
                    rir_path_sub = os.listdir(rir_path)
                    rir, fs = torchaudio.load(os.path.join(rir_path, rir_path_sub[0]))
                    rir = rir.reshape(-1)
                    if fs != 16000:
                        rir = torchaudio.transforms.Resample(fs, 16000)(rir)
                        fs = 16000

                    Th, Tt, _ = RIRestThTt(rir, fs)
                    Th, Tt = (
                        Th.tolist(),
                        Tt.tolist(),
                    )  # convert to list type for csv file

                    V = volume_BUT(folderNo, commonpath)
                    V = V.tolist()  # convert to list type for csv file

                    dstRcvOriSrc_path = os.path.join(
                        path, foldername, str(i).zfill(2), "mic_meta.txt"
                    )
                    rir_metadata = pd.read_csv(dstRcvOriSrc_path, sep="\t", header=None)
                    dist_rcv = rir_metadata[
                        rir_metadata[0] == "$EnvMic" + str(i) + "RelDistance"
                    ][1].values[0]
                    dist_rcv = torch.tensor(float(dist_rcv)).round(decimals=3)
                    azimuth = rir_metadata[
                        rir_metadata[0] == "$EnvMic" + str(i) + "RelAzimuth"
                    ][1].values[0]
                    azimuth = (
                        torch.deg2rad(torch.tensor(float(azimuth))) / torch.pi
                    )  # radian unit
                    elevation = rir_metadata[
                        rir_metadata[0] == "$EnvMic" + str(i) + "RelElevation"
                    ][1].values[0]
                    elevation = (
                        torch.deg2rad(torch.tensor(float(elevation))) / torch.pi
                    )  # radian unit
                    ori_src = torch.cat(
                        (azimuth.reshape(1), elevation.reshape(1))
                    ).round(decimals=3)
                    dist_rcv, ori_src = (
                        dist_rcv.tolist(),
                        ori_src.tolist(),
                    )  # convert to list type for csv file

                    rm_ID = room_ID_BUT(folderNo)

                    if not os.path.exists(savepath + "/RIR.data"):
                        os.makedirs(savepath + "/RIR.data")
                    torchaudio.save(
                        os.path.join(
                            savepath + "/RIR.data", "BUT_RIR_no" + str(j) + ".wav"
                        ),
                        rir.unsqueeze(0),
                        sample_rate=fs,
                        bits_per_sample=16,
                        encoding="PCM_S",
                    )

                    ThTtDistRcvOriSrc_label.append(
                        [
                            "BUT_RIR_no" + str(j) + ".wav",
                            rm_ID,
                            Th,
                            Tt,
                            V,
                            dist_rcv,
                            ori_src,
                        ]
                    )

    ThTtDistRcvOriSrc_label = pd.DataFrame(
        ThTtDistRcvOriSrc_label,
        columns=["RIR", "roomID", "Th", "Tt", "volume", "distRcv", "oriSrc"],
    )
    if not os.path.exists(savepath + "/RIR.metadata"):
        os.makedirs(savepath + "/RIR.metadata")

    if csv_savemode == "w":
        ThTtDistRcvOriSrc_label = ThTtDistRcvOriSrc_label.to_csv(
            savepath + "/RIR.metadata/ThTtDistRcvOriSrc_label.csv",
            mode=csv_savemode,
            index=False,
        )
    elif csv_savemode == "a":
        ThTtDistRcvOriSrc_label = ThTtDistRcvOriSrc_label.to_csv(
            savepath + "/RIR.metadata/ThTtDistRcvOriSrc_label.csv",
            mode=csv_savemode,
            index=False,
            header=False,
        )

    return print(
        "RIRs saved in "
        + savepath
        + "/RIR.data"
        + " and metadata saved in "
        + savepath
        + "/RIR.metadata"
    )


sti_calculator = RapidSpeechTransmissionIndex()
alcons_calculator = PercentageArticulationLoss()
edt_calculator = EarlyDecayTime()
c80_calculator = Clarity(clarity_mode="C80")
c50_calculator = Clarity(clarity_mode="C50")
d50_calculator = Definition()
ts_calculator = CenterTime()


def readRIR_BUT_real_recording(
    path_retrans,
    path_rir,
    savepath,
    num_files: int = 100,
):
    """
    read RIR from BUT RIR database for the retransmission dataset

    """
    j = 0
    n = 0
    folderNos = [4, 6, 7, 8, 9]
    ThTtDistRcvOriSrc_label = []
    random.seed(3407)
    for folderNo in folderNos:
        print("Processing folder " + str(folderNo) + "...")
        path_ = path_retrans + checkfolder_BUT(folderNo)
        lst = os.listdir(path_)
        lst.sort()
        path = path_rir + checkfolder_BUT(folderNo)
        for foldername in lst:
            if foldername.startswith("SpkID"):
                print("Processing speaker " + foldername + "...")
                for i in range(1, 32):
                    retransed_path = os.path.join(
                        path_,
                        foldername,
                        str(i).zfill(2),
                        "english/LibriSpeech/test-clean/",
                    )
                    librispeech_folder = Path(retransed_path)
                    extension = ".wav"
                    matching_files = librispeech_folder.rglob(f"*{extension}")
                    matching_files = [str(x) for x in matching_files]
                    # sort the files
                    matching_files.sort()
                    # randomly choose some audio file
                    # real_audio_paths = random.sample(matching_files, num_files)

                    real_audio_paths = matching_files[
                        j * num_files : (j + 1) * num_files
                    ]
                    if len(real_audio_paths) < num_files:
                        # handle the last batch
                        real_audio_paths = random.sample(matching_files, num_files)
                    j += 1

                    for subpath in real_audio_paths:
                        print("Processing audio file " + subpath + "...")
                        real_audio, fs_audio = torchaudio.load(
                            os.path.join(retransed_path, subpath)
                        )
                        meta_info = subpath.split("/")[-1].split(".")[0]
                        real_audio = real_audio.reshape(-1)  # 1D tensor
                        if fs_audio != 16000:
                            real_audio = torchaudio.transforms.Resample(
                                fs_audio, 16000
                            )(real_audio)
                            fs_audio = 16000

                        if not os.path.exists(savepath + "/real_audio.data"):
                            os.makedirs(savepath + "/real_audio.data")
                        torchaudio.save(
                            os.path.join(
                                savepath + "/real_audio.data",
                                "BUT_realRecording_no" + str(n) + ".wav",
                            ),
                            real_audio.unsqueeze(0),
                            sample_rate=fs_audio,
                            bits_per_sample=16,
                            encoding="PCM_S",
                        )

                        rir_path = os.path.join(
                            path, foldername, str(i).zfill(2), "RIR"
                        )
                        rir_path_sub = os.listdir(rir_path)
                        rir_path_sub = [
                            file for file in rir_path_sub if file.endswith("v00.wav")
                        ]
                        rir, fs = torchaudio.load(
                            os.path.join(rir_path, rir_path_sub[0])
                        )
                        rir = rir.reshape(-1)
                        if fs != 16000:
                            rir = torchaudio.transforms.Resample(fs, 16000)(rir)
                            fs = 16000

                        Th, Tt, _ = RIRestThTt(rir, fs)
                        Th, Tt = Th.numpy().round(decimals=4), Tt.numpy().round(
                            decimals=4
                        )

                        V = volume_BUT(folderNo, path_rir)
                        # compute volume of the room
                        V_log10 = np.log10(torch.prod(V).numpy()).round(decimals=3)
                        # round to 1 decimal place
                        V_ns = torch.prod(V).numpy().round(decimals=0)

                        dstRcvOriSrc_path = os.path.join(
                            path, foldername, str(i).zfill(2), "mic_meta.txt"
                        )
                        rir_metadata = pd.read_csv(
                            dstRcvOriSrc_path, sep="\t", header=None
                        )
                        dist_rcv = rir_metadata[
                            rir_metadata[0] == "$EnvMic" + str(i) + "RelDistance"
                        ][1].values[0]
                        dist_rcv = torch.tensor(float(dist_rcv)).round(decimals=3)
                        azimuth = rir_metadata[
                            rir_metadata[0] == "$EnvMic" + str(i) + "RelAzimuth"
                        ][1].values[0]
                        azimuth = (
                            torch.deg2rad(torch.tensor(float(azimuth))) / torch.pi
                        ).round(
                            decimals=3
                        )  # radian unit
                        elevation = rir_metadata[
                            rir_metadata[0] == "$EnvMic" + str(i) + "RelElevation"
                        ][1].values[0]
                        elevation = (
                            torch.deg2rad(torch.tensor(float(elevation))) / torch.pi
                        ).round(
                            decimals=3
                        )  # radian unit

                        dist_rcv = round(dist_rcv.tolist(), 3)
                        elevation = round(elevation.tolist(), 3)
                        azimuth = round(azimuth.tolist(), 3)

                        rm_ID = room_ID_BUT(folderNo)
                        spk_ID = foldername

                        # RA calculation
                        sti = sti_calculator(rir, fs)
                        alcons = alcons_calculator(sti)
                        t60 = Tt
                        edt = edt_calculator(rir, fs)
                        c80 = c80_calculator(rir, fs)
                        c50 = c50_calculator(rir, fs)
                        d50 = d50_calculator(rir, fs) / 100  # dicard the percentage
                        ts = ts_calculator(rir, fs)

                        sti = round(sti.tolist(), 4)
                        alcons = round(alcons.tolist(), 4)
                        t60 = round(t60.tolist(), 4)
                        edt = round(edt.tolist(), 4)
                        c80 = round(c80.tolist(), 4)
                        c50 = round(c50.tolist(), 4)
                        d50 = round(d50.tolist(), 4)
                        ts = round(ts.tolist(), 4)

                        if not os.path.exists(savepath + "/RIR.data"):
                            os.makedirs(savepath + "/RIR.data")
                        torchaudio.save(
                            os.path.join(
                                savepath + "/RIR.data", "BUT_RIR_no" + str(j) + ".wav"
                            ),
                            rir.unsqueeze(0),
                            sample_rate=fs,
                            bits_per_sample=16,
                            encoding="PCM_S",
                        )

                        ThTtDistRcvOriSrc_label.append(
                            [
                                "BUT_realRecording_no" + str(n) + ".wav",
                                "BUT_RIR_no" + str(j) + ".wav",
                                str(i).zfill(2),
                                meta_info,
                                spk_ID,
                                rm_ID,
                                Th,
                                Tt,
                                V_ns,
                                V_log10,
                                dist_rcv,
                                azimuth,
                                elevation,
                                sti,
                                alcons,
                                t60,
                                edt,
                                c80,
                                c50,
                                d50,
                                ts,
                            ]
                        )

                        n += 1

    ThTtDistRcvOriSrc_label = pd.DataFrame(
        ThTtDistRcvOriSrc_label,
        columns=[
            "realRecording",
            "RIR",
            "No",
            "librispeech_metainfo",
            "spkID",
            "roomID",
            "Th",
            "Tt",
            "volume",
            "volume_log10",
            "distRcv",
            "azimuth",
            "elevation",
            "STI",
            "ALCONS",
            "T60",
            "EDT",
            "C80",
            "C50",
            "D50",
            "TS",
        ],
    )
    # uniformly sample the 279 real recordings from previous 2620 unique-speech real recordings
    train_val_test, test_manifest = skms.train_test_split(
        ThTtDistRcvOriSrc_label,
        test_size=279 * 2,
        random_state=2406,
        stratify=ThTtDistRcvOriSrc_label["RIR"],
    )
    # uniformly sample the 2341 real recordings from previous 2341 unique-speech real recordings
    train_manifest, val_manifest = skms.train_test_split(
        train_val_test,
        test_size=279 * 2,
        random_state=2406,
        stratify=train_val_test["RIR"],
    )
    # reshuflle the dataframe
    train_manifest = train_manifest.sample(frac=1, random_state=2406).reset_index(
        drop=True
    )
    val_manifest = val_manifest.sample(frac=1, random_state=2406).reset_index(drop=True)
    test_manifest = test_manifest.sample(frac=1, random_state=2406).reset_index(
        drop=True
    )
    if not os.path.exists(savepath + "/real_audio.metadata"):
        os.makedirs(savepath + "/real_audio.metadata")

    train_manifest.to_csv(
        savepath + "/real_audio.metadata/train_manifest.csv", index=False
    )
    val_manifest.to_csv(savepath + "/real_audio.metadata/val_manifest.csv", index=False)
    test_manifest.to_csv(
        savepath + "/real_audio.metadata/test_manifest.csv", index=False
    )

    return print(
        "Real recordings saved in "
        + savepath
        + "/real_audio.data"
        + " and metadata saved in "
        + savepath
        + "/real_audio.metadata"
    )


def readRIR_BUT_retrans(
    path_retrans, path_rir, savepath, num_files: int = 1, csv_savemode="w"
):
    """
    read RIR from BUT RIR database for the retransmission dataset

    """
    j = 0
    n = 0
    folderNos = [4, 6, 7, 8, 9]
    ThTtDistRcvOriSrc_label = []
    random.seed(42)
    for folderNo in folderNos:
        print("Processing folder " + str(folderNo) + "...")
        path_ = path_retrans + checkfolder_BUT(folderNo)
        lst = os.listdir(path_)
        lst.sort()
        path = path_rir + checkfolder_BUT(folderNo)
        for foldername in lst:
            if foldername.startswith("SpkID"):
                for i in range(1, 32):
                    j += 1
                    retransed_path = os.path.join(
                        path_,
                        foldername,
                        str(i).zfill(2),
                        "english/LibriSpeech/test-clean/",
                    )
                    librispeech_folder = Path(retransed_path)
                    extension = ".wav"
                    matching_files = librispeech_folder.rglob(f"*{extension}")
                    matching_files = [str(x) for x in matching_files]
                    # randomly choose one audio file
                    retran_audio_paths = random.sample(matching_files, num_files)

                    n += 1
                    retran_audio, fs_audio = torchaudio.load(
                        os.path.join(retransed_path, retran_audio_paths[0])
                    )
                    retran_audio = retran_audio.reshape(-1)  # 1D tensor
                    if fs_audio != 16000:
                        retran_audio = torchaudio.transforms.Resample(fs_audio, 16000)(
                            retran_audio
                        )
                        fs_audio = 16000

                    if not os.path.exists(savepath + "/retrans_audio.data"):
                        os.makedirs(savepath + "/retrans_audio.data")
                    torchaudio.save(
                        os.path.join(
                            savepath + "/retrans_audio.data",
                            "BUT_retrans_speech_no" + str(n) + ".wav",
                        ),
                        retran_audio.unsqueeze(0),
                        sample_rate=fs_audio,
                        bits_per_sample=16,
                        encoding="PCM_S",
                    )

                    rir_path = os.path.join(path, foldername, str(i).zfill(2), "RIR")
                    rir_path_sub = os.listdir(rir_path)
                    rir_path_sub = [
                        file for file in rir_path_sub if file.endswith("v00.wav")
                    ]
                    rir, fs = torchaudio.load(os.path.join(rir_path, rir_path_sub[0]))
                    rir = rir.reshape(-1)
                    if fs != 16000:
                        rir = torchaudio.transforms.Resample(fs, 16000)(rir)
                        fs = 16000

                    Th, Tt, _ = RIRestThTt(rir, fs)
                    Th, Tt = Th.numpy().round(decimals=4), Tt.numpy().round(decimals=4)

                    V = volume_BUT(folderNo, path_rir)
                    # compute volume of the room
                    V_log10 = np.log10(torch.prod(V).numpy()).round(decimals=3)
                    # round to 1 decimal place
                    V_ns = torch.prod(V).numpy().round(decimals=0)

                    dstRcvOriSrc_path = os.path.join(
                        path, foldername, str(i).zfill(2), "mic_meta.txt"
                    )
                    rir_metadata = pd.read_csv(dstRcvOriSrc_path, sep="\t", header=None)
                    dist_rcv = rir_metadata[
                        rir_metadata[0] == "$EnvMic" + str(i) + "RelDistance"
                    ][1].values[0]
                    dist_rcv = torch.tensor(float(dist_rcv)).round(decimals=3)
                    azimuth = rir_metadata[
                        rir_metadata[0] == "$EnvMic" + str(i) + "RelAzimuth"
                    ][1].values[0]
                    azimuth = (
                        torch.deg2rad(torch.tensor(float(azimuth))) / torch.pi
                    ).round(
                        decimals=3
                    )  # radian unit
                    elevation = rir_metadata[
                        rir_metadata[0] == "$EnvMic" + str(i) + "RelElevation"
                    ][1].values[0]
                    elevation = (
                        torch.deg2rad(torch.tensor(float(elevation))) / torch.pi
                    ).round(
                        decimals=3
                    )  # radian unit

                    dist_rcv = round(dist_rcv.tolist(), 3)
                    elevation = round(elevation.tolist(), 3)
                    azimuth = round(azimuth.tolist(), 3)

                    rm_ID = room_ID_BUT(folderNo)
                    spk_ID = foldername

                    if not os.path.exists(savepath + "/RIR.data"):
                        os.makedirs(savepath + "/RIR.data")
                    torchaudio.save(
                        os.path.join(
                            savepath + "/RIR.data", "BUT_RIR_no" + str(j) + ".wav"
                        ),
                        rir.unsqueeze(0),
                        sample_rate=fs,
                        bits_per_sample=16,
                        encoding="PCM_S",
                    )

                    ThTtDistRcvOriSrc_label.append(
                        [
                            "BUT_retrans_speech_no" + str(n) + ".wav",
                            "BUT_RIR_no" + str(j) + ".wav",
                            str(i).zfill(2),
                            spk_ID,
                            rm_ID,
                            Th,
                            Tt,
                            V_ns,
                            V_log10,
                            dist_rcv,
                            azimuth,
                            elevation,
                        ]
                    )

    ThTtDistRcvOriSrc_label = pd.DataFrame(
        ThTtDistRcvOriSrc_label,
        columns=[
            "retrans_speech",
            "RIR",
            "No",
            "spkID",
            "roomID",
            "Th",
            "Tt",
            "volume",
            "volume_log10",
            "distRcv",
            "azimuth",
            "elevation",
        ],
    )
    # reshuflle the dataframe
    ThTtDistRcvOriSrc_label = ThTtDistRcvOriSrc_label.sample(
        frac=1, random_state=42
    ).reset_index(drop=True)
    if not os.path.exists(savepath + "/retrans.metadata"):
        os.makedirs(savepath + "/retrans.metadata")

    if csv_savemode == "w":
        ThTtDistRcvOriSrc_label = ThTtDistRcvOriSrc_label.to_csv(
            savepath + "/retrans.metadata/BUT_retrans.csv",
            mode=csv_savemode,
            index=False,
        )
    elif csv_savemode == "a":
        ThTtDistRcvOriSrc_label = ThTtDistRcvOriSrc_label.to_csv(
            savepath + "/retrans.metadata/BUT_retrans.csv",
            mode=csv_savemode,
            index=False,
            header=False,
        )

    return print(
        "RIRs saved in "
        + savepath
        + "/RIR.data"
        + " and metadata saved in "
        + savepath
        + "/RIR.metadata"
    )


def checkfolder_ACE(folderNo):
    match folderNo:
        case 1:
            folderpath = "/Building_Lobby"
        case 2:
            folderpath = "/Lecture_Room_1"
        case 3:
            folderpath = "/Lecture_Room_2"
        case 4:
            folderpath = "/Meeting_Room_1"
        case 5:
            folderpath = "/Meeting_Room_2"
        case 6:
            folderpath = "/Office_1"
        case 7:
            folderpath = "/Office_2"
    return folderpath


def volume_ACE(folderNo):
    match folderNo:
        case 1:
            V = torch.tensor([5.13, 4.47, 3.2])
        case 2:
            V = torch.tensor([9.73, 6.93, 2.9])
        case 3:
            V = torch.tensor([9.29, 13.556, 2.9])
        case 4:
            V = torch.tensor([5.11, 6.61, 3.0])
        case 5:
            V = torch.tensor([9.07, 10.32, 2.6])
        case 6:
            V = torch.tensor([4.83, 3.32, 2.9])
        case 7:
            V = torch.tensor([5.10, 3.32, 2.9])
    return V.round(decimals=1)


def micPos_ACE(folderNo, micNo):
    match folderNo:
        case 1:
            match micNo:
                case 1:
                    d = (61 - 200) ** 2 + (198 - 233) ** 2
                    d = torch.sqrt(torch.tensor(d)) / 100  # convert to meter unit
                case 2:
                    d = (61 - 349) ** 2 + (198 - 225) ** 2
                    d = torch.sqrt(torch.tensor(d)) / 100
        case 2:
            match micNo:
                case 1:
                    d = (373 - 344) ** 2 + (365 - 280) ** 2
                    d = torch.sqrt(torch.tensor(d)) / 100
                case 2:
                    d = (373 - 352) ** 2 + (365 - 107) ** 2
                    d = torch.sqrt(torch.tensor(d)) / 100
        case 3:
            match micNo:
                case 1:
                    d = (314 - 547) ** 2 + (603 - 509) ** 2
                    d = torch.sqrt(torch.tensor(d)) / 100
                case 2:
                    d = (314 - 547) ** 2 + (603 - 393) ** 2
                    d = torch.sqrt(torch.tensor(d)) / 100
        case 4:
            match micNo:
                case 1:
                    d = (125.9 - 273.6) ** 2 + (139.4 - 82) ** 2
                    d = torch.sqrt(torch.tensor(d)) / 100
                case 2:
                    d = (125.9 - 395.6) ** 2 + (139.4 - 85) ** 2
                    d = torch.sqrt(torch.tensor(d)) / 100
        case 5:
            match micNo:
                case 1:
                    d = (407 - 399) ** 2 + (465 - 300) ** 2
                    d = torch.sqrt(torch.tensor(d)) / 100
                case 2:
                    d = (407 - 399) ** 2 + (465 - 200) ** 2
                    d = torch.sqrt(torch.tensor(d)) / 100
        case 6:
            match micNo:
                case 1:
                    d = (104 - 215) ** 2 + (206 - 229) ** 2
                    d = torch.sqrt(torch.tensor(d)) / 100
                case 2:
                    d = (104 - 369) ** 2 + (206 - 215) ** 2
                    d = torch.sqrt(torch.tensor(d)) / 100
        case 7:
            match micNo:
                case 1:
                    d = (173 - 262) ** 2 + (141 - 125) ** 2
                    d = torch.sqrt(torch.tensor(d)) / 100
                case 2:
                    d = (173 - 416) ** 2 + (141 - 225) ** 2
                    d = torch.sqrt(torch.tensor(d)) / 100
    return d.round(decimals=3)


def micOri_ACE(folderNo, micNo):
    match folderNo:
        case 1:
            match micNo:
                case 1:
                    adjacent_a = torch.tensor(61 - 200)
                    opposite_a = torch.tensor(198 - 233)
                    azimuth = torch.arctan(opposite_a / adjacent_a) / torch.pi
                    ori = torch.cat((azimuth.reshape(1), torch.tensor(0.0).reshape(1)))
                case 2:
                    adjacent_a = torch.tensor(61 - 349)
                    opposite_a = torch.tensor(198 - 225)
                    azimuth = torch.arctan(opposite_a / adjacent_a) / torch.pi
                    ori = torch.cat((azimuth.reshape(1), torch.tensor(0.0).reshape(1)))
        case 2:
            match micNo:
                case 1:
                    adjacent_a = torch.tensor(373 - 344)
                    opposite_a = torch.tensor(365 - 280)
                    azimuth = torch.arctan(opposite_a / adjacent_a) / torch.pi
                    ori = torch.cat((azimuth.reshape(1), torch.tensor(0.0).reshape(1)))
                case 2:
                    adjacent_a = torch.tensor(373 - 352)
                    opposite_a = torch.tensor(365 - 107)
                    azimuth = torch.arctan(opposite_a / adjacent_a) / torch.pi
                    ori = torch.cat((azimuth.reshape(1), torch.tensor(0.0).reshape(1)))
        case 3:
            match micNo:
                case 1:
                    adjacent_a = torch.tensor(314 - 547)
                    opposite_a = torch.tensor(603 - 509)
                    azimuth = torch.arctan(opposite_a / adjacent_a) / torch.pi
                    ori = torch.cat((azimuth.reshape(1), torch.tensor(0.0).reshape(1)))
                case 2:
                    adjacent_a = torch.tensor(314 - 547)
                    opposite_a = torch.tensor(603 - 393)
                    azimuth = torch.arctan(opposite_a / adjacent_a) / torch.pi
                    ori = torch.cat((azimuth.reshape(1), torch.tensor(0.0).reshape(1)))
        case 4:
            match micNo:
                case 1:
                    adjacent_a = torch.tensor(125.9 - 273.6)
                    opposite_a = torch.tensor(139.4 - 82)
                    azimuth = torch.arctan(opposite_a / adjacent_a) / torch.pi
                    ori = torch.cat((azimuth.reshape(1), torch.tensor(0.0).reshape(1)))
                case 2:
                    adjacent_a = torch.tensor(125.9 - 395.6)
                    opposite_a = torch.tensor(139.4 - 85)
                    azimuth = torch.arctan(opposite_a / adjacent_a) / torch.pi
                    ori = torch.cat((azimuth.reshape(1), torch.tensor(0.0).reshape(1)))
        case 5:
            match micNo:
                case 1:
                    adjacent_a = torch.tensor(407 - 399)
                    opposite_a = torch.tensor(465 - 300)
                    azimuth = torch.arctan(opposite_a / adjacent_a) / torch.pi
                    ori = torch.cat((azimuth.reshape(1), torch.tensor(0.0).reshape(1)))
                case 2:
                    adjacent_a = torch.tensor(407 - 399)
                    opposite_a = torch.tensor(465 - 200)
                    azimuth = torch.arctan(opposite_a / adjacent_a) / torch.pi
                    ori = torch.cat((azimuth.reshape(1), torch.tensor(0.0).reshape(1)))
        case 6:
            match micNo:
                case 1:
                    adjacent_a = torch.tensor(104 - 215)
                    opposite_a = torch.tensor(206 - 229)
                    azimuth = torch.arctan(opposite_a / adjacent_a) / torch.pi
                    ori = torch.cat((azimuth.reshape(1), torch.tensor(0.0).reshape(1)))
                case 2:
                    adjacent_a = torch.tensor(104 - 369)
                    opposite_a = torch.tensor(206 - 215)
                    azimuth = torch.arctan(opposite_a / adjacent_a) / torch.pi
                    ori = torch.cat((azimuth.reshape(1), torch.tensor(0.0).reshape(1)))
        case 7:
            match micNo:
                case 1:
                    adjacent_a = torch.tensor(173 - 262)
                    opposite_a = torch.tensor(141 - 125)
                    azimuth = torch.arctan(opposite_a / adjacent_a) / torch.pi
                    ori = torch.cat((azimuth.reshape(1), torch.tensor(0.0).reshape(1)))
                case 2:
                    adjacent_a = torch.tensor(173 - 416)
                    opposite_a = torch.tensor(141 - 225)
                    azimuth = torch.arctan(opposite_a / adjacent_a) / torch.pi
                    ori = torch.cat((azimuth.reshape(1), torch.tensor(0.0).reshape(1)))

    return ori.round(decimals=3)


def rm_ID_ACE(folderNo):
    match folderNo:
        case 1:
            rm_ID = "rmID_12"
        case 2:
            rm_ID = "rmID_13"
        case 3:
            rm_ID = "rmID_14"
        case 4:
            rm_ID = "rmID_15"
        case 5:
            rm_ID = "rmID_16"
        case 6:
            rm_ID = "rmID_17"
        case 7:
            rm_ID = "rmID_18"
    return rm_ID


def readRIR_ACE(commonpath, savepath, csv_savemode="w"):
    """read RIR from ACE RIR database"""
    j = 0
    ThTtDistRcvOriSrc_label = []
    for folderNo in range(1, 8):
        path = commonpath + checkfolder_ACE(folderNo)
        micNo = os.listdir(path)
        micNo.sort()
        if len(micNo) == 3:
            micNo = micNo[1:]
        for No in micNo:
            rir_path = os.path.join(path, No)
            for rir_filename in os.listdir(rir_path):
                if rir_filename.endswith("RIR.wav"):
                    j += 1
                    rir, fs = torchaudio.load(os.path.join(rir_path, rir_filename))
                    rir = rir.reshape(-1)
                    if fs != 16000:
                        rir = torchaudio.transforms.Resample(fs, 16000)(rir)
                        fs = 16000
                    Th, Tt, _ = RIRestThTt(rir, fs)
                    Th, Tt = (
                        Th.tolist(),
                        Tt.tolist(),
                    )  # convert to list type for csv file

                    V = volume_ACE(folderNo)
                    V = V.tolist()  # convert to list type for csv file

                    dist_rcv = micPos_ACE(folderNo, int(No))
                    ori_src = micOri_ACE(folderNo, int(No))
                    dist_rcv, ori_src = (
                        dist_rcv.tolist(),
                        ori_src.tolist(),
                    )  # convert to list type for csv file
                    rm_ID = rm_ID_ACE(folderNo)

                    if not os.path.exists(savepath + "/RIR.data"):
                        os.makedirs(savepath + "/RIR.data")
                    torchaudio.save(
                        os.path.join(
                            savepath + "/RIR.data", "ACE_RIR_no" + str(j) + ".wav"
                        ),
                        rir.unsqueeze(0),
                        sample_rate=fs,
                        bits_per_sample=16,
                        encoding="PCM_S",
                    )
                    ThTtDistRcvOriSrc_label.append(
                        [
                            "ACE_RIR_no" + str(j) + ".wav",
                            rm_ID,
                            Th,
                            Tt,
                            V,
                            dist_rcv,
                            ori_src,
                        ]
                    )

    ThTtDistRcvOriSrc_label = pd.DataFrame(
        ThTtDistRcvOriSrc_label,
        columns=["RIR", "roomID", "Th", "Tt", "volume", "distRcv", "oriSrc"],
    )
    if not os.path.exists(savepath + "/RIR.metadata"):
        os.makedirs(savepath + "/RIR.metadata")

    if csv_savemode == "w":
        ThTtDistRcvOriSrc_label.to_csv(
            savepath + "/RIR.metadata/ThTtDistRcvOriSrc_label.csv",
            mode=csv_savemode,
            index=False,
        )
    if csv_savemode == "a":
        ThTtDistRcvOriSrc_label.to_csv(
            savepath + "/RIR.metadata/ThTtDistRcvOriSrc_label.csv",
            mode=csv_savemode,
            index=False,
            header=False,
        )
    return print(
        "RIRs saved in "
        + savepath
        + "/RIR.data"
        + " and metadata saved in "
        + savepath
        + "/RIR.metadata"
    )


def checkfolder_OpenAIR(path, filename, foldername):
    label_rir = pd.read_csv(path + "/RIR.metadata/ThTtDistRcvOriSrc_label.csv")
    ori_src = label_rir["oriSrc"]
    dist_rcv = label_rir["distRcv"]
    # convert to tensor type
    ori_src = ori_src.apply(lambda x: ast.literal_eval(x))
    ori_src = torch.tensor(ori_src.values.tolist())
    dist_rcv = torch.tensor(dist_rcv.values.tolist())

    # median of src orientation and dist rcv for all RIRs
    mean_ori_src = ori_src.mean(dim=0)  # two elements
    median_dist_rcv = dist_rcv.median()

    if foldername == "arthur-sykes-rymer-auditorium-university-york":
        V = torch.tensor(1560.0)
        if filename == "s1r2.wav":
            dist_rcv = torch.tensor(7.6)
        elif filename == "s1r4.wav":
            dist_rcv = torch.tensor(12.9)
        elif filename == "s1r7.wav":
            dist_rcv = torch.tensor(3.3)
        elif filename == "s2r4.wav":
            dist_rcv = torch.tensor(13.5)
        elif filename == "s2r6.wav":
            dist_rcv = torch.tensor(9.5)
        elif filename == "s2r7.wav":
            dist_rcv = torch.tensor(4.1)
        else:
            raise ValueError("filename not found: " + foldername + "/" + filename)
        ori_src = mean_ori_src
        rm_ID = "rmID_19"

    elif foldername == "central-hall-university-york":
        V = torch.tensor(8000.0)
        dist_rcv = median_dist_rcv
        ori_src = mean_ori_src
        rm_ID = "rmID_20"

    elif foldername == "dixon-studio-theatre-university-york":
        V = torch.tensor(908.23)
        if filename == "r1_rir_bformat.wav":
            dist_rcv = torch.tensor(6.4)
        elif filename == "r2_rir_bformat.wav":
            dist_rcv = torch.tensor(6.1)
        elif filename == "r3_rir_bformat.wav":
            dist_rcv = torch.tensor(5.5)
        elif filename == "r4_rir_bformat.wav":
            dist_rcv = torch.tensor(2.3)
        elif filename == "r5_rir_bformat.wav":
            dist_rcv = torch.tensor(9.2)
        else:
            raise ValueError("filename not found: " + foldername + "/" + filename)
        ori_src = mean_ori_src
        rm_ID = "rmID_21"

    elif foldername == "genesis-6-studio-live-room-drum-set":
        V = torch.tensor(35.2)
        if filename == "kick_ir.wav":
            dist_rcv = torch.tensor(2.46)
        elif filename == "snare_ir.wav":
            dist_rcv = torch.tensor(2.46)
        elif filename == "toma_ir.wav":
            dist_rcv = torch.tensor(2.21)
        elif filename == "tomb_ir.wav":
            dist_rcv = torch.tensor(2.23)
        elif filename == "tomc_ir.wav":
            dist_rcv = torch.tensor(2.41)
        elif filename == "hh_ir.wav":
            dist_rcv = torch.tensor(2.46)
        elif filename == "crash_ir.wav":
            dist_rcv = torch.tensor(2.13)
        elif filename == "ride_ir.wav":
            dist_rcv = torch.tensor(2.17)
        else:
            raise ValueError("filename not found: " + foldername + "/" + filename)
        ori_src = mean_ori_src
        rm_ID = "rmID_22"

    elif foldername == "heslington-church-vaa-group-2":
        V = torch.tensor(2000.0)
        if filename == "impulseresponseheslingtonchurch-001.wav":
            dist_rcv = torch.tensor(4.5)
        elif filename == "impulseresponseheslingtonchurch-002.wav":
            dist_rcv = torch.tensor(5.0)
        elif filename == "impulseresponseheslingtonchurch-003.wav":
            dist_rcv = torch.tensor(4.1)
        elif filename == "impulseresponseheslingtonchurch-004.wav":
            dist_rcv = torch.tensor(4.1)
        elif filename == "impulseresponseheslingtonchurch-005.wav":
            dist_rcv = torch.tensor(4.5)
        elif filename == "impulseresponseheslingtonchurch-006.wav":
            dist_rcv = torch.tensor(11.0)
        elif filename == "impulseresponseheslingtonchurch-007.wav":
            dist_rcv = torch.tensor(12.1)
        else:
            raise ValueError("filename not found: " + foldername + "/" + filename)
        ori_src = mean_ori_src
        rm_ID = "rmID_23"

    elif foldername == "r1-nuclear-reactor-hall":
        V = torch.tensor(3500.0)
        if filename == "r1_bformat.wav":
            dist_rcv = torch.tensor(14.88)
        elif filename == "r1_bformat-48k.wav":
            dist_rcv = torch.tensor(14.88)
        else:
            raise ValueError("filename not found: " + foldername + "/" + filename)
        ori_src = mean_ori_src
        rm_ID = "rmID_24"

    elif foldername == "ron-cooke-hub-university-york":
        V = torch.tensor(18000.0)
        if filename == "fsfr.wav":
            dist_rcv = torch.tensor(12.6)
        elif filename == "fssr.wav":
            dist_rcv = torch.tensor(12.6)
        elif filename == "fstr.wav":
            dist_rcv = torch.tensor(34.7)
        elif filename == "ssfr.wav":
            dist_rcv = torch.tensor(13.8)
        elif filename == "sssr.wav":
            dist_rcv = torch.tensor(15.9)
        elif filename == "sstr.wav":
            dist_rcv = torch.tensor(18.0)
        elif filename == "tsfr.wav":
            dist_rcv = torch.tensor(33.0)
        elif filename == "tsfthr.wav":
            dist_rcv = torch.tensor(11.0)
        elif filename == "tssr.wav":
            dist_rcv = torch.tensor(32.0)
        elif filename == "tstr_ir.wav":
            dist_rcv = torch.tensor(4.5)
        else:
            raise ValueError("filename not found: " + foldername + "/" + filename)
        ori_src = mean_ori_src
        rm_ID = "rmID_25"

    elif foldername == "shrine-and-parish-church-all-saints-north-street-_":
        V = torch.tensor(2398.8)
        if filename == "r1_90_bformat.wav":
            dist_rcv = torch.tensor(12.25)
        elif filename == "r2_90_bformat.wav":
            dist_rcv = torch.tensor(12.28)
        elif filename == "r3_90_bformat.wav":
            dist_rcv = torch.tensor(12.4)
        elif filename == "r4_90_bformat.wav":
            dist_rcv = torch.tensor(13.45)
        elif filename == "r5_90_bformat.wav":
            dist_rcv = torch.tensor(5.2)
        elif filename == "r6_90_bformat.wav":
            dist_rcv = torch.tensor(4.5)
        else:
            raise ValueError("filename not found: " + foldername + "/" + filename)
        ori_src = mean_ori_src
        rm_ID = "rmID_26"

    elif foldername == "sports-centre-university-york":
        V = torch.tensor(9000.0)
        dist_rcv = median_dist_rcv
        ori_src = mean_ori_src
        rm_ID = "rmID_27"

    elif foldername == "spring-lane-building-university-york":
        V = torch.tensor(6000.0)
        if filename == "sp1_mp1_ir_bformat_trimmed.wav":
            dist_rcv = torch.tensor(5.1)
        elif filename == "sp1_mp2_ir_bformat_trimmed.wav":
            dist_rcv = torch.tensor(8.9)
        elif filename == "sp1_mp3_ir_bformat_trimmed.wav":
            dist_rcv = torch.tensor(9.0)
        elif filename == "sp1_mp4_ir_bformat_trimmed.wav":
            dist_rcv = torch.tensor(16.7)
        elif filename == "sp1_mp5_ir_bformat_trimmed.wav":
            dist_rcv = torch.tensor(4.0)
        elif filename == "sp2_mp1_ir_bformat_trimmed.wav":
            dist_rcv = torch.tensor(5.9)
        elif filename == "sp2_mp2_ir_bformat_trimmed.wav":
            dist_rcv = torch.tensor(14.2)
        elif filename == "sp2_mp3_ir_bformat_trimmed.wav":
            dist_rcv = torch.tensor(21.6)
        elif filename == "sp2_mp4_ir_bformat_trimmed.wav":
            dist_rcv = torch.tensor(10.0)
        elif filename == "stairwell_ir_bformat_trimmed.wav":
            dist_rcv = median_dist_rcv
        else:
            raise ValueError("filename not found: " + foldername + "/" + filename)
        ori_src = mean_ori_src
        rm_ID = "rmID_28"

    elif foldername == "st-andrews-church":
        V = torch.tensor(2600.0)
        dist_rcv = torch.tensor(11.5)
        ori_src = mean_ori_src
        rm_ID = "rmID_29"

    elif foldername == "st-margarets-church-national-centre-early-music":
        V = torch.tensor(3600.0)
        if filename == "r1_1st_configuration.wav":
            dist_rcv = torch.tensor(11.0)
        elif filename == "r2_1st_configuration.wav":
            dist_rcv = torch.tensor(9.9)
        elif filename == "r3_1st_configuration.wav":
            dist_rcv = torch.tensor(9.0)
        elif filename == "r4_1st_configuration.wav":
            dist_rcv = torch.tensor(8.5)
        elif filename == "r5_1st_configuration.wav":
            dist_rcv = torch.tensor(8.5)
        elif filename == "r6_1st_configuration.wav":
            dist_rcv = torch.tensor(9.0)
        elif filename == "r7_1st_configuration.wav":
            dist_rcv = torch.tensor(9.9)
        elif filename == "r8_1st_configuration.wav":
            dist_rcv = torch.tensor(11.0)
        elif filename == "r9_1st_configuration.wav":
            dist_rcv = torch.tensor(12.3)
        elif filename == "r10_1st_configuration.wav":
            dist_rcv = torch.tensor(7.8)
        elif filename == "r11_1st_configuration.wav":
            dist_rcv = torch.tensor(6.1)
        elif filename == "r12_1st_configuration.wav":
            dist_rcv = torch.tensor(4.6)
        elif filename == "r13_1st_configuration.wav":
            dist_rcv = torch.tensor(3.6)
        elif filename == "r14_1st_configuration.wav":
            dist_rcv = torch.tensor(3.5)
        elif filename == "r15_1st_configuration.wav":
            dist_rcv = torch.tensor(4.5)
        elif filename == "r16_1st_configuration.wav":
            dist_rcv = torch.tensor(6.0)
        elif filename == "r17_1st_configuration.wav":
            dist_rcv = torch.tensor(7.7)
        elif filename == "r18_1st_configuration.wav":
            dist_rcv = torch.tensor(9.6)
        elif filename == "r19_1st_configuration.wav":
            dist_rcv = torch.tensor(8.6)
        elif filename == "r20_1st_configuration.wav":
            dist_rcv = torch.tensor(6.0)
        elif filename == "r21_1st_configuration.wav":
            dist_rcv = torch.tensor(6.0)
        elif filename == "r22_1st_configuration.wav":
            dist_rcv = torch.tensor(5.7)
        elif filename == "r23_1st_configuration.wav":
            dist_rcv = torch.tensor(6.0)
        elif filename == "r24_1st_configuration.wav":
            dist_rcv = torch.tensor(6.6)
        elif filename == "r25_1st_configuration.wav":
            dist_rcv = torch.tensor(8.8)
        elif filename == "r26_1st_configuration.wav":
            dist_rcv = torch.tensor(9.8)
        elif filename == "r1_2nd_configuration.wav":
            dist_rcv = torch.tensor(11.0)
        elif filename == "r2_2nd_configuration.wav":
            dist_rcv = torch.tensor(9.9)
        elif filename == "r3_2nd_configuration.wav":
            dist_rcv = torch.tensor(9.0)
        elif filename == "r4_2nd_configuration.wav":
            dist_rcv = torch.tensor(8.5)
        elif filename == "r5_2nd_configuration.wav":
            dist_rcv = torch.tensor(8.5)
        elif filename == "r6_2nd_configuration.wav":
            dist_rcv = torch.tensor(9.0)
        elif filename == "r7_2nd_configuration.wav":
            dist_rcv = torch.tensor(9.8)
        elif filename == "r8_2nd_configuration.wav":
            dist_rcv = torch.tensor(11.0)
        elif filename == "r9_2nd_configuration.wav":
            dist_rcv = torch.tensor(12.3)
        elif filename == "r10_2nd_configuration.wav":
            dist_rcv = torch.tensor(7.8)
        elif filename == "r11_2nd_configuration.wav":
            dist_rcv = torch.tensor(6.1)
        elif filename == "r12_2nd_configuration.wav":
            dist_rcv = torch.tensor(4.6)
        elif filename == "r13_2nd_configuration.wav":
            dist_rcv = torch.tensor(3.6)
        elif filename == "r14_2nd_configuration.wav":
            dist_rcv = torch.tensor(3.5)
        elif filename == "r15_2nd_configuration.wav":
            dist_rcv = torch.tensor(4.5)
        elif filename == "r16_2nd_configuration.wav":
            dist_rcv = torch.tensor(6.0)
        elif filename == "r17_2nd_configuration.wav":
            dist_rcv = torch.tensor(7.7)
        elif filename == "r18_2nd_configuration.wav":
            dist_rcv = torch.tensor(9.6)
        elif filename == "r19_2nd_configuration.wav":
            dist_rcv = torch.tensor(8.6)
        elif filename == "r20_2nd_configuration.wav":
            dist_rcv = torch.tensor(7.7)
        elif filename == "r21_2nd_configuration.wav":
            dist_rcv = torch.tensor(6.0)
        elif filename == "r22_2nd_configuration.wav":
            dist_rcv = torch.tensor(5.7)
        elif filename == "r23_2nd_configuration.wav":
            dist_rcv = torch.tensor(6.0)
        elif filename == "r24_2nd_configuration.wav":
            dist_rcv = torch.tensor(6.6)
        elif filename == "r25_2nd_configuration.wav":
            dist_rcv = torch.tensor(8.8)
        elif filename == "r26_2nd_configuration.wav":
            dist_rcv = torch.tensor(9.8)
        elif filename == "r1_3rd_configuration.wav":
            dist_rcv = torch.tensor(11.0)
        elif filename == "r2_3rd_configuration.wav":
            dist_rcv = torch.tensor(9.9)
        elif filename == "r3_3rd_configuration.wav":
            dist_rcv = torch.tensor(9.0)
        elif filename == "r4_3rd_configuration.wav":
            dist_rcv = torch.tensor(8.5)
        elif filename == "r5_3rd_configuration.wav":
            dist_rcv = torch.tensor(8.5)
        elif filename == "r6_3rd_configuration.wav":
            dist_rcv = torch.tensor(9.0)
        elif filename == "r7_3rd_configuration.wav":
            dist_rcv = torch.tensor(9.8)
        elif filename == "r8_3rd_configuration.wav":
            dist_rcv = torch.tensor(11.0)
        elif filename == "r9_3rd_configuration.wav":
            dist_rcv = torch.tensor(12.3)
        elif filename == "r10_3rd_configuration.wav":
            dist_rcv = torch.tensor(7.8)
        elif filename == "r11_3rd_configuration.wav":
            dist_rcv = torch.tensor(6.1)
        elif filename == "r12_3rd_configuration.wav":
            dist_rcv = torch.tensor(4.6)
        elif filename == "r13_3rd_configuration.wav":
            dist_rcv = torch.tensor(3.6)
        elif filename == "r14_3rd_configuration.wav":
            dist_rcv = torch.tensor(3.5)
        elif filename == "r15_3rd_configuration.wav":
            dist_rcv = torch.tensor(4.5)
        elif filename == "r16_3rd_configuration.wav":
            dist_rcv = torch.tensor(6.0)
        elif filename == "r17_3rd_configuration.wav":
            dist_rcv = torch.tensor(7.7)
        elif filename == "r18_3rd_configuration.wav":
            dist_rcv = torch.tensor(9.6)
        elif filename == "r19_3rd_configuration.wav":
            dist_rcv = torch.tensor(8.6)
        elif filename == "r20_3rd_configuration.wav":
            dist_rcv = torch.tensor(7.7)
        elif filename == "r21_3rd_configuration.wav":
            dist_rcv = torch.tensor(6.0)
        elif filename == "r22_3rd_configuration.wav":
            dist_rcv = torch.tensor(5.7)
        elif filename == "r23_3rd_configuration.wav":
            dist_rcv = torch.tensor(6.0)
        elif filename == "r24_3rd_configuration.wav":
            dist_rcv = torch.tensor(6.6)
        elif filename == "r25_3rd_configuration.wav":
            dist_rcv = torch.tensor(8.8)
        elif filename == "r26_3rd_configuration.wav":
            dist_rcv = torch.tensor(9.8)
        else:
            raise ValueError("filename not found: " + foldername + "/" + filename)
        ori_src = mean_ori_src
        rm_ID = "rmID_30"

    elif foldername == "st-margarets-church-ncem-5-piece-band-spatial-measurements":
        V = torch.tensor(1500.0)
        dist_rcv = median_dist_rcv
        ori_src = mean_ori_src
        rm_ID = "rmID_31"

    elif foldername == "st-marys-abbey-reconstruction":
        V = torch.tensor(47000.0)
        if filename == "phase1_bformat.wav":
            dist_rcv = torch.tensor(45.0)
        elif filename == "phase2_bformat.wav":
            dist_rcv = torch.tensor(45.0)
        elif filename == "phase3_bformat.wav":
            dist_rcv = torch.tensor(45.0)
        elif filename == "phase1_bformat_catt.wav":
            dist_rcv = torch.tensor(45.0)
        else:
            raise ValueError("filename not found: " + foldername + "/" + filename)
        ori_src = mean_ori_src
        rm_ID = "rmID_32"

    elif foldername == "st-patricks-church-patrington":
        V = torch.tensor(8000.0)
        if filename == "soundfield_s1r1.wav":
            dist_rcv = torch.tensor(18.5)
        elif filename == "soundfield_s2r2.wav":
            dist_rcv = torch.tensor(8.7)
        elif filename == "soundfield_s3r3.wav":
            dist_rcv = torch.tensor(8.3)
        ori_src = mean_ori_src
        rm_ID = "rmID_33"

    elif foldername == "terrys-factory-warehouse":
        V = torch.tensor(4500.0)
        dist_rcv = torch.tensor(28.35)
        ori_src = mean_ori_src
        rm_ID = "rmID_34"

    elif foldername == "terrys-typing-room":
        V = torch.tensor(3500.0)
        dist_rcv = torch.tensor(8.86)
        ori_src = mean_ori_src
        rm_ID = "rmID_35"

    elif foldername == "usina-del-arte-symphony-hall":
        V = torch.tensor(15700.0)
        dist_rcv = median_dist_rcv
        ori_src = mean_ori_src
        rm_ID = "rmID_36"

    elif foldername == "tvisongur-sound-sculpture-iceland-model":
        V = torch.tensor(100.41)
        if filename == "source1domefareceiver2domelabformat.wav":
            dist_rcv = torch.tensor(3.23)
        elif filename == "source1domefareceiver3domesibformat.wav":
            dist_rcv = torch.tensor(4.85)
        elif filename == "source1domefareceiver4domedobformat.wav":
            dist_rcv = torch.tensor(4.73)
        elif filename == "source1domefareceiver5domemibformat.wav":
            dist_rcv = torch.tensor(2.85)
        elif filename == "source2domelareceiver1domefabformat.wav":
            dist_rcv = torch.tensor(3.23)
        elif filename == "source2domelareceiver3domesibformat.wav":
            dist_rcv = torch.tensor(2.68)
        elif filename == "source2domelareceiver4domedobformat.wav":
            dist_rcv = torch.tensor(4.22)
        elif filename == "source2domelareceiver5domemibformat.wav":
            dist_rcv = torch.tensor(2.37)
        elif filename == "source3domesireceiver1domefabformat.wav":
            dist_rcv = torch.tensor(4.85)
        elif filename == "source3domesireceiver2domelabformat.wav":
            dist_rcv = torch.tensor(2.68)
        elif filename == "source3domesireceiver4domedobformat.wav":
            dist_rcv = torch.tensor(2.43)
        elif filename == "source3domesireceiver5domemibformat.wav":
            dist_rcv = torch.tensor(2.20)
        elif filename == "source4domedoreceiver1domefabformat.wav":
            dist_rcv = torch.tensor(4.73)
        elif filename == "source4domedoreceiver2domelabformat.wav":
            dist_rcv = torch.tensor(4.22)
        elif filename == "source4domedoreceiver3domesibformat.wav":
            dist_rcv = torch.tensor(2.43)
        elif filename == "source4domedoreceiver5domemibformat.wav":
            dist_rcv = torch.tensor(2.12)
        elif filename == "source5domemireceiver1domefabformat.wav":
            dist_rcv = torch.tensor(2.85)
        elif filename == "source5domemireceiver2domelabformat.wav":
            dist_rcv = torch.tensor(2.37)
        elif filename == "source5domemireceiver3domesibformat.wav":
            dist_rcv = torch.tensor(2.20)
        elif filename == "source5domemireceiver4domedobformat.wav":
            dist_rcv = torch.tensor(2.12)
        else:
            raise ValueError("filename not found: " + foldername + "/" + filename)
        ori_src = mean_ori_src
        rm_ID = "rmID_37"

    elif foldername == "tyndall-bruce-monument":
        V = torch.tensor(40.0)
        dist_rcv = torch.tensor(2.3)
        ori_src = mean_ori_src
        rm_ID = "rmID_38"

    elif foldername == "york-guildhall-council-chamber":
        V = torch.tensor(1140.0)
        if filename == "councilchamber_s1_r1_ir_1_96000.wav":
            dist_rcv = torch.tensor(12.0)
        elif filename == "councilchamber_s1_r2_ir_1_96000.wav":
            dist_rcv = torch.tensor(9.0)
        elif filename == "councilchamber_s1_r3_ir_1_96000.wav":
            dist_rcv = torch.tensor(9.0)
        elif filename == "councilchamber_s1_r4_ir_1_96000.wav":
            dist_rcv = torch.tensor(7.0)
        elif filename == "councilchamber_s2_r1_ir_1_96000.wav":
            dist_rcv = torch.tensor(6.0)
        elif filename == "councilchamber_s2_r2_ir_1_96000.wav":
            dist_rcv = torch.tensor(4.5)
        elif filename == "councilchamber_s2_r3_ir_1_96000.wav":
            dist_rcv = torch.tensor(3.0)
        elif filename == "councilchamber_s2_r4_ir_1_96000.wav":
            dist_rcv = torch.tensor(3.0)
        elif filename == "councilchamber_s3_r1_ir_1_96000.wav":
            dist_rcv = torch.tensor(6.0)
        elif filename == "councilchamber_s3_r2_ir_1_96000.wav":
            dist_rcv = torch.tensor(3.0)
        elif filename == "councilchamber_s3_r3_ir_1_96000.wav":
            dist_rcv = torch.tensor(4.5)
        elif filename == "councilchamber_s3_r4_ir_1_96000.wav":
            dist_rcv = torch.tensor(3.0)
        else:
            raise ValueError("filename not found: " + foldername + "/" + filename)
        ori_src = mean_ori_src
        rm_ID = "rmID_39"

    else:
        print("foldername not found: " + foldername)

    return (
        V.round(decimals=1),
        dist_rcv.round(decimals=3),
        ori_src.round(decimals=3),
        rm_ID,
    )


def readRIR_OpenAIR(commonpath, rootpath, savepath, csv_savemode="w"):
    building_foldername = os.listdir(commonpath)
    building_foldername.sort()
    ThTtDictRcvOriSrc_label = []
    j = 0
    for building in building_foldername:
        path = os.path.join(commonpath, building, "b-format")
        lst_files = os.listdir(path)
        lst_files.sort()
        for files in lst_files:
            if files.endswith(".wav"):
                j += 1
                rir, fs = torchaudio.load(os.path.join(path, files))
                if fs != 16000:
                    rir = torchaudio.transforms.Resample(fs, 16000)(rir)
                    fs = 16000
                rir = rir[0, :]  # only use W channel

                Th, Tt, _ = RIRestThTt(rir, fs)
                Th, Tt = Th.tolist(), Tt.tolist()  # convert to list for csv

                V, dist_rcv, ori_src, rm_ID = checkfolder_OpenAIR(
                    rootpath, files, building
                )
                V, dist_rcv, ori_src = (
                    V.tolist(),
                    dist_rcv.tolist(),
                    ori_src.tolist(),
                )  # convert to list for csv

                if not os.path.exists(savepath + "/RIR.data"):
                    os.makedirs(savepath + "/RIR.data")
                torchaudio.save(
                    os.path.join(
                        savepath + "/RIR.data", "OpenAIR_No" + str(j) + ".wav"
                    ),
                    rir.unsqueeze(0),
                    sample_rate=fs,
                    bits_per_sample=16,
                    encoding="PCM_S",
                )
                ThTtDictRcvOriSrc_label.append(
                    [
                        "OpenAIR_No" + str(j) + ".wav",
                        rm_ID,
                        Th,
                        Tt,
                        V,
                        dist_rcv,
                        ori_src,
                    ]
                )
    ThTtDictRcvOriSrc_label = pd.DataFrame(
        ThTtDictRcvOriSrc_label,
        columns=["RIR", "roomID", "Th", "Tt", "volume", "distRcv", "oriSrc"],
    )
    if not os.path.exists(savepath + "/RIR.metadata"):
        os.makedirs(savepath + "/RIR.metadata")

    if csv_savemode == "w":
        ThTtDictRcvOriSrc_label.to_csv(
            savepath + "/RIR.metadata/ThTtDistRcvOriSrc_label.csv",
            mode=csv_savemode,
            index=False,
        )
    elif csv_savemode == "a":
        ThTtDictRcvOriSrc_label.to_csv(
            savepath + "/RIR.metadata/ThTtDistRcvOriSrc_label.csv",
            mode=csv_savemode,
            index=False,
            header=False,
        )
    return print(
        "RIRs saved in "
        + savepath
        + "/RIR.data"
        + " and metadata saved in "
        + savepath
        + "/RIR.metadata"
    )


def readSpeech_ATR(folderpath):
    """
    read speech from folder with VAD

    """
    # load vad model
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=True,
        trust_repo=True,
    )

    (get_speech_timestamps, _, _, _, collect_chunks) = vad_utils

    speech_ATR = []
    for filename in os.listdir(folderpath):
        if filename.endswith(".wav") and filename.startswith("J_"):
            filepath = os.path.join(folderpath, filename)
            # print(filepath) # debug
            speech_raw, fs = torchaudio.load(filepath)
            speech_raw = speech_raw.squeeze(0)
            if fs != 16000:
                speech_raw = torchaudio.transforms.Resample(fs, 16000)(speech_raw)
                fs = 16000
            # get speech timestamps from full audio file
            speech_timestamps = get_speech_timestamps(
                speech_raw, vad_model, sampling_rate=fs
            )
            # merge consecutive speech chunks
            speech = collect_chunks(speech_timestamps, speech_raw)
            speech_ATR.append(speech)
    speech_ATR = pd.DataFrame(zip(speech_ATR, strict=False), columns=["ATR_speech"])
    # if not os.path.exists(folderpath+"/ATR_speech_labeled"):
    #     os.makedirs(folderpath+"/ATR_speech_labeled")
    # path = folderpath+"/ATR_speech_labeled/speech_ATR.pt"
    # torch.save(speech_ATR, path)

    return speech_ATR


def readNoise_FSD18k(path, savepath):
    csv_path = path + "/FSDnoisy18k.meta/train.csv"
    csv = pd.read_csv(csv_path)
    noise_filename = csv[csv["noisy_small"] == 1]
    noise_filename = noise_filename["fname"].tolist()
    for filename in os.listdir(path + "/FSDnoisy18k.audio_train"):
        if filename in noise_filename:
            filepath = os.path.join(path + "/FSDnoisy18k.audio_train", filename)
            noise_raw, fs = torchaudio.load(filepath)
            if fs != 16000:
                noise_raw = torchaudio.transforms.Resample(fs, 16000)(noise_raw)
                fs = 16000
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            torchaudio.save(
                os.path.join(savepath, "FSD18k_" + filename),
                noise_raw,
                sample_rate=fs,
                bits_per_sample=16,
                encoding="PCM_S",
            )


def checkfolder_DEMAND(No):
    match No:
        case 1:
            suffix = "DKITCHEN"
        case 2:
            suffix = "DLIVING"
        case 3:
            suffix = "DWASHING"
        case 4:
            suffix = "NFIELD"
        case 5:
            suffix = "NPARK"
        case 6:
            suffix = "NRIVER"
        case 7:
            suffix = "OHALLWAY"
        case 8:
            suffix = "OMEETING"
        case 9:
            suffix = "OOFFICE"
        case 10:
            suffix = "PCAFETER"
        case 11:
            suffix = "PRESTO"
        case 12:
            suffix = "PSTATION"
        case 13:
            suffix = "SPSQUARE"
        case 14:
            suffix = "STRAFFIC"
        case 15:
            suffix = "SCAFE"
        case 16:
            suffix = "TBUS"
        case 17:
            suffix = "TCAR"
        case 18:
            suffix = "TMETRO"
    return suffix


def readNoise_DEMAND(path, savepath, csv_savemode="w"):
    """
    read noise from DEAMND dataset
    """
    noise_label = []
    no = 0
    for suffixNo in range(1, 19):
        suffix = checkfolder_DEMAND(suffixNo)
        commonpath = os.path.join(path, suffix)
        lst = os.listdir(commonpath)
        lst.sort()
        for filename in lst:
            if filename.endswith(".wav"):
                no += 1
                filepath = os.path.join(commonpath, filename)
                noise_raw, fs = torchaudio.load(filepath)
                if fs != 16000:
                    noise_raw = torchaudio.transforms.Resample(fs, 16000)(noise_raw)
                    fs = 16000
                if not os.path.exists(savepath + "/noise.data"):
                    os.makedirs(savepath + "/noise.data")
                torchaudio.save(
                    os.path.join(
                        savepath + "/noise.data", "DEMAND_No" + str(no) + ".wav"
                    ),
                    noise_raw,
                    sample_rate=fs,
                    bits_per_sample=16,
                    encoding="PCM_S",
                )

                noise_label.append(
                    [
                        "DEMAND_No" + str(no) + ".wav",
                        suffix + "_" + filename.split(".wav")[0],
                    ]
                )
    noise_label = pd.DataFrame(noise_label, columns=["filename", "label"])

    if not os.path.exists(savepath + "/noise.metadata"):
        os.mkdir(savepath + "/noise.metadata")
    if csv_savemode == "w":
        noise_label.to_csv(
            savepath + "/noise.metadata/noise_label.csv", mode=csv_savemode, index=False
        )
    elif csv_savemode == "a":
        noise_label.to_csv(
            savepath + "/noise.metadata/noise_label.csv",
            mode=csv_savemode,
            index=False,
            header=False,
        )

    return None


def readNoise_BUT(commonpath, savepath, csv_savemode="w"):
    """
    read noise from BUT ReverbDB
    """
    noise_label = []
    no = 0
    for folderNo in range(1, 10):
        path = commonpath + checkfolder_BUT(folderNo)
        place_name = path.split("/")[-2]  # get place name
        for foldername in os.listdir(path):
            if foldername.startswith("SpkID"):
                SpkID = foldername.split("_")[0]
                for i in range(1, 32):
                    noise_path = os.path.join(
                        path, foldername, str(i).zfill(2), "silence"
                    )
                    for filename in os.listdir(noise_path):
                        if filename.endswith("v00.wav"):
                            filepath = os.path.join(noise_path, filename)
                            noise_raw, fs = torchaudio.load(filepath)
                            if fs != 16000:
                                noise_raw = torchaudio.transforms.Resample(fs, 16000)(
                                    noise_raw
                                )
                                fs = 16000
                            if not os.path.exists(savepath + "/noise.data"):
                                os.makedirs(savepath + "/noise.data")
                            no += 1
                            torchaudio.save(
                                os.path.join(
                                    savepath + "/noise.data",
                                    "BUT_" + "No" + str(no) + ".wav",
                                ),
                                noise_raw,
                                sample_rate=fs,
                                bits_per_sample=16,
                                encoding="PCM_S",
                            )

                            noise_label.append(
                                [
                                    "BUT_" + "No" + str(no) + ".wav",
                                    place_name + "_" + SpkID + "_" + "No" + str(i),
                                ]
                            )
    noise_label = pd.DataFrame(noise_label, columns=["filename", "label"])
    if not os.path.exists(savepath + "/noise.metadata"):
        os.makedirs(savepath + "/noise.metadata")
    if csv_savemode == "w":
        noise_label.to_csv(
            savepath + "/noise.metadata/noise_label.csv", mode=csv_savemode, index=False
        )
    elif csv_savemode == "a":
        noise_label.to_csv(
            savepath + "/noise.metadata/noise_label.csv",
            mode=csv_savemode,
            index=False,
            header=False,
        )

    return None
