import pandas as pd
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# ---------------------- obtain realistic RIRs of the test set ----------------------
rir_manifest = pd.read_csv("./data/RIR_aggregated/RIR.metadata/RIRLabelAugmentV2.csv")

test_manifest = pd.read_csv("./data/noiseReverbSpeech/test_manifest.csv")

rir_path_list_test = []
for i in range(len(test_manifest)):
    rir_No = int(test_manifest["reverbSpeech"][i].split(".wav")[0].split("_No")[1])
    rir_path = rir_manifest.iloc[rir_No]
    rir_path_list_test.append(rir_path)

rir_path_list_test = pd.DataFrame(rir_path_list_test).reset_index(
    drop=True, inplace=False
)

rir_path_list_test.to_csv(
    "./data/noiseReverbSpeech/test_manifest_RIR.csv",
    index=False,
)

# ---------------------- obtain realistic RIRs of the test set ----------------------

val_manifest = pd.read_csv("./data/noiseReverbSpeech/val_manifest.csv")

rir_path_list_val = []
for i in range(len(val_manifest)):
    rir_No = int(val_manifest["reverbSpeech"][i].split(".wav")[0].split("_No")[1])
    rir_path = rir_manifest.iloc[rir_No]
    rir_path_list_val.append(rir_path)

rir_path_list_val = pd.DataFrame(rir_path_list_val).reset_index(
    drop=True, inplace=False
)

rir_path_list_val.to_csv(
    "./data/noiseReverbSpeech/val_manifest_RIR.csv",
    index=False,
)

# ---------------------- obtain realistic RIRs of the train set ----------------------

train_manifest = pd.read_csv("./data/noiseReverbSpeech/train_manifest.csv")

rir_path_list_train = []
for i in range(len(train_manifest)):
    rir_No = int(train_manifest["reverbSpeech"][i].split(".wav")[0].split("_No")[1])
    rir_path = rir_manifest.iloc[rir_No]
    rir_path_list_train.append(rir_path)

rir_path_list_train = pd.DataFrame(rir_path_list_train).reset_index(
    drop=True, inplace=False
)

rir_path_list_train.to_csv(
    "./data/noiseReverbSpeech/train_manifest_RIR.csv",
    index=False,
)
