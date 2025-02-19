import os

import pandas as pd
import rootutils
import torchaudio
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.AcousticParameterUtils import (  # noqa: E402
    CenterTime,
    Clarity,
    Definition,
    EarlyDecayTime,
    PercentageArticulationLoss,
    RapidSpeechTransmissionIndex,
)

# * ------------------------------ for training RIR dataset ------------------------------

train_rir_manifest = pd.read_csv("./data/noiseReverbSpeech/train_manifest_RIR.csv")
train_manifest = pd.read_csv("./data/noiseReverbSpeech/train_manifest.csv")

sti_ = []
alcons_ = []
T60_ = []
edt_ = []
c80_ = []
c50_ = []
d50_ = []
ts_ = []


with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TimeRemainingColumn(),
    TaskProgressColumn(),
    MofNCompleteColumn(),
) as progress:
    task = progress.add_task("[green]Processing...", total=len(train_rir_manifest))
    for i in range(len(train_rir_manifest)):
        progress.update(task, advance=1)
        rir_path = train_rir_manifest["RIR"][i]
        data_dir = "./data/RIR_aggregated/RIR.data/"
        rir, fs = torchaudio.load(os.path.join(data_dir, rir_path))

        rir = rir.squeeze()

        # compute the rap
        sti_calculator = RapidSpeechTransmissionIndex()
        sti = sti_calculator(rir, fs)

        alcons_calculator = PercentageArticulationLoss()
        alcons = alcons_calculator(sti)

        T60 = train_rir_manifest.iloc[i]["Tt"]

        edt_calculator = EarlyDecayTime()
        edt = edt_calculator(rir, fs)

        c80_calculator = Clarity(clarity_mode="C80")
        c80 = c80_calculator(rir, fs)

        c50_calculator = Clarity(clarity_mode="C50")
        c50 = c50_calculator(rir, fs)

        d50_calculator = Definition()
        d50 = d50_calculator(rir, fs) / 100  # convert to decimals

        ts_calculator = CenterTime()
        ts = ts_calculator(rir, fs)

        sti_.append(round(sti.item(), 4))
        alcons_.append(round(alcons.item(), 4))
        T60_.append(T60)
        edt_.append(round(edt.item(), 4))
        c80_.append(round(c80.item(), 4))
        c50_.append(round(c50.item(), 4))
        d50_.append(round(d50.item(), 4))
        ts_.append(round(ts.item(), 4))

train_manifest.insert(15, "STI", sti_)
train_manifest.insert(16, "ALCONS", alcons_)
train_manifest.insert(17, "T60", T60_)
train_manifest.insert(18, "EDT", edt_)
train_manifest.insert(19, "C80", c80_)
train_manifest.insert(20, "C50", c50_)
train_manifest.insert(21, "D50", d50_)
train_manifest.insert(22, "TS", ts_)

train_manifest.to_csv(
    "./data/noiseReverbSpeech/train_manifest_alt.csv",
    index=False,
)

# * ------------------------------ for validation and test RIR dataset ------------------------------
test_rir_manifest = pd.read_csv("./data/noiseReverbSpeech/test_manifest_RIR.csv")
val_rir_manifest = pd.read_csv("./data/noiseReverbSpeech/val_manifest_RIR.csv")

sti_test_, sti_val_ = [], []
alcons_test_, alcons_val_ = [], []
T60_test_, T60_val_ = [], []
edt_test_, edt_val_ = [], []
c80_test_, c80_val_ = [], []
c50_test_, c50_val_ = [], []
d50_test_, d50_val_ = [], []
ts_test_, ts_val_ = [], []

assert len(test_rir_manifest) == len(val_rir_manifest)

with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TimeRemainingColumn(),
    TaskProgressColumn(),
    MofNCompleteColumn(),
) as progress:
    task = progress.add_task("[green]Processing...", total=len(test_rir_manifest))
    for i in range(len(test_rir_manifest)):
        progress.update(task, advance=1)
        test_rir_path = test_rir_manifest["RIR"][i]
        val_rir_path = val_rir_manifest["RIR"][i]
        data_dir = "./data/RIR_aggregated/RIR.data/"
        rir_test, fs = torchaudio.load(os.path.join(data_dir, test_rir_path))
        rir_val, fs = torchaudio.load(os.path.join(data_dir, val_rir_path))

        rir_test = rir_test.squeeze()
        rir_val = rir_val.squeeze()

        # compute the rap
        sti_calculator = RapidSpeechTransmissionIndex()
        sti_test = sti_calculator(rir_test, fs)
        sti_val = sti_calculator(rir_val, fs)

        alcons_calculator = PercentageArticulationLoss()
        alcons_test = alcons_calculator(sti_test)
        alcons_val = alcons_calculator(sti_val)

        T60_test = test_rir_manifest.iloc[i]["Tt"]
        T60_val = val_rir_manifest.iloc[i]["Tt"]

        edt_calculator = EarlyDecayTime()
        edt_test = edt_calculator(rir_test, fs)
        edt_val = edt_calculator(rir_val, fs)

        c80_calculator = Clarity(clarity_mode="C80")
        c80_test = c80_calculator(rir_test, fs)
        c80_val = c80_calculator(rir_val, fs)

        c50_calculator = Clarity(clarity_mode="C50")
        c50_test = c50_calculator(rir_test, fs)
        c50_val = c50_calculator(rir_val, fs)

        d50_calculator = Definition()
        d50_test = d50_calculator(rir_test, fs) / 100  # convert to decimals
        d50_val = d50_calculator(rir_val, fs) / 100  # convert to decimals

        ts_calculator = CenterTime()
        ts_test = ts_calculator(rir_test, fs)
        ts_val = ts_calculator(rir_val, fs)

        sti_test_.append(round(sti_test.item(), 4))
        alcons_test_.append(round(alcons_test.item(), 4))
        T60_test_.append(round(T60_test, 4))
        edt_test_.append(round(edt_test.item(), 4))
        c80_test_.append(round(c80_test.item(), 4))
        c50_test_.append(round(c50_test.item(), 4))
        d50_test_.append(round(d50_test.item(), 4))
        ts_test_.append(round(ts_test.item(), 4))

        sti_val_.append(round(sti_val.item(), 4))
        alcons_val_.append(round(alcons_val.item(), 4))
        T60_val_.append(round(T60_val, 4))
        edt_val_.append(round(edt_val.item(), 4))
        c80_val_.append(round(c80_val.item(), 4))
        c50_val_.append(round(c50_val.item(), 4))
        d50_val_.append(round(d50_val.item(), 4))
        ts_val_.append(round(ts_val.item(), 4))

test_manifest = pd.read_csv("./data/noiseReverbSpeech/test_manifest.csv")
val_manifest = pd.read_csv("./data/noiseReverbSpeech/val_manifest.csv")

test_manifest.insert(15, "STI", sti_test_)
test_manifest.insert(16, "ALCONS", alcons_test_)
test_manifest.insert(17, "T60", T60_test_)
test_manifest.insert(18, "EDT", edt_test_)
test_manifest.insert(19, "C80", c80_test_)
test_manifest.insert(20, "C50", c50_test_)
test_manifest.insert(21, "D50", d50_test_)
test_manifest.insert(22, "TS", ts_test_)
test_manifest.to_csv(
    "./data/noiseReverbSpeech/test_manifest_alt.csv",
    index=False,
)

val_manifest.insert(15, "STI", sti_val_)
val_manifest.insert(16, "ALCONS", alcons_val_)
val_manifest.insert(17, "T60", T60_val_)
val_manifest.insert(18, "EDT", edt_val_)
val_manifest.insert(19, "C80", c80_val_)
val_manifest.insert(20, "C50", c50_val_)
val_manifest.insert(21, "D50", d50_val_)
val_manifest.insert(22, "TS", ts_val_)
val_manifest.to_csv(
    "./data/noiseReverbSpeech/val_manifest_alt.csv",
    index=False,
)
