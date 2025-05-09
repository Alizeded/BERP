import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore
import seaborn as sns
import torch  # type: ignore

from src.preprocessing.RIRutils import checkfolder_BUT

# Assuming src.utils.unitary_linear_norm and src.preprocessing.RIRutils are in PYTHONPATH
# or in the correct relative path.
from src.utils.unitary_linear_norm import unitary_norm


def main():
    # Normalize the RAPs
    # Read the manifest file for synthetic data
    train_manifest_alt_path = "./data/noiseReverbSpeech/train_manifest_alt.csv"
    test_manifest_alt_path = "./data/noiseReverbSpeech/test_manifest_alt.csv"
    val_manifest_alt_path = "./data/noiseReverbSpeech/val_manifest_alt.csv"

    if not (
        os.path.exists(train_manifest_alt_path)
        and os.path.exists(test_manifest_alt_path)
        and os.path.exists(val_manifest_alt_path)
    ):
        print(
            "Warning: One or more alt manifest files not found. Skipping synthetic data processing part 1."
        )
        manifest = (
            pd.DataFrame()
        )  # Create empty manifest to avoid later errors if files are missing
    else:
        train_manifest = pd.read_csv(train_manifest_alt_path)
        test_manifest = pd.read_csv(test_manifest_alt_path)
        val_manifest = pd.read_csv(val_manifest_alt_path)

        # Concat the dataframes
        manifest = pd.concat([train_manifest, test_manifest, val_manifest])

        # Insert 'real_recording' column with 0 for synthetic data
        # Check if column exists to prevent error on re-runs
        if "real_recording" not in train_manifest.columns:
            train_manifest.insert(33, "real_recording", 0)
        else:
            train_manifest["real_recording"] = 0

        if "real_recording" not in val_manifest.columns:
            val_manifest.insert(33, "real_recording", 0)
        else:
            val_manifest["real_recording"] = 0

        if "real_recording" not in test_manifest.columns:
            test_manifest.insert(33, "real_recording", 0)
        else:
            test_manifest["real_recording"] = 0

        train_manifest.to_csv(train_manifest_alt_path, index=False)
        test_manifest.to_csv(test_manifest_alt_path, index=False)
        val_manifest.to_csv(val_manifest_alt_path, index=False)

    if not manifest.empty:
        # This cell originally just displayed manifest["STI"].to_numpy()
        # In a script, this would print to console if we added print()
        # print(manifest["STI"].to_numpy())

        # Extract and normalize parameters from synthetic manifest
        sti = manifest["STI"].to_numpy()
        alcons = manifest["ALCONS"].to_numpy()
        t60 = manifest["T60"].to_numpy()
        edt = manifest["EDT"].to_numpy()
        c80 = manifest["C80"].to_numpy()
        c50 = manifest["C50"].to_numpy()
        d50 = manifest["D50"].to_numpy().round(decimals=4)
        ts = manifest["TS"].to_numpy()

        volume_log10 = manifest["volume_log10"].to_numpy()
        dist_src = manifest["distRcv"].to_numpy()

        print("Synthetic Data Stats:")
        print("upper bound of sti: ", sti.max(), "lower bound of sti: ", sti.min())
        print(
            "upper bound of alcons: ",
            alcons.max(),
            "lower bound of alcons: ",
            alcons.min(),
        )
        print("upper bound of t60: ", t60.max(), "lower bound of t60: ", t60.min())
        print("upper bound of edt: ", edt.max(), "lower bound of edt: ", edt.min())
        print("upper bound of c80: ", c80.max(), "lower bound of c80: ", c80.min())
        print("upper bound of c50: ", c50.max(), "lower bound of c50: ", c50.min())
        print("upper bound of d50: ", d50.max(), "lower bound of d50: ", d50.min())
        print("upper bound of ts: ", ts.max(), "lower bound of ts: ", ts.min())
        print(
            "upper bound of volume_log10: ",
            volume_log10.max(),
            "lower bound of volume_log10: ",
            volume_log10.min(),
        )
        print(
            "upper bound of dist_src: ",
            dist_src.max(),
            "lower bound of dist_src: ",
            dist_src.min(),
        )

    # Read real recordings manifest
    train_real_manifest_path = (
        "./data/BUT_real_recording_11160samples/real_audio.metadata/train_manifest.csv"
    )
    val_real_manifest_path = (
        "./data/BUT_real_recording_11160samples/real_audio.metadata/val_manifest.csv"
    )
    test_real_manifest_path = (
        "./data/BUT_real_recording_11160samples/real_audio.metadata/test_manifest.csv"
    )

    if not (
        os.path.exists(train_real_manifest_path)
        and os.path.exists(val_real_manifest_path)
        and os.path.exists(test_real_manifest_path)
    ):
        print(
            "Warning: One or more real recording manifest files not found. Skipping real data processing."
        )
        real_manifest = pd.DataFrame()
    else:
        train_real_manifest = pd.read_csv(train_real_manifest_path)
        val_real_manifest = pd.read_csv(val_real_manifest_path)
        test_real_manifest = pd.read_csv(test_real_manifest_path)

        real_manifest = pd.concat(
            [train_real_manifest, val_real_manifest, test_real_manifest]
        )

        sti_real = real_manifest["STI"].to_numpy()
        alcons_real = real_manifest["ALCONS"].to_numpy()
        t60_real = real_manifest["T60"].to_numpy()
        edt_real = real_manifest["EDT"].to_numpy()
        c80_real = real_manifest["C80"].to_numpy()
        c50_real = real_manifest["C50"].to_numpy()
        d50_real = real_manifest["D50"].to_numpy().round(decimals=4)
        ts_real = real_manifest["TS"].to_numpy()
        volume_real = real_manifest["volume_log10"].to_numpy()
        dist_src_real = real_manifest["distRcv"].to_numpy()
        # Th_real = real_manifest["Th"].to_numpy() # Th_real was defined but not used in prints

        print("\nReal Recordings Data Stats:")
        print(
            "upper bound of sti_real: ",
            sti_real.max(),
            "lower bound of sti_real: ",
            sti_real.min(),
        )
        print(
            "upper bound of alcons_real: ",
            alcons_real.max(),
            "lower bound of alcons_real: ",
            alcons_real.min(),
        )
        print(
            "upper bound of t60_real: ",
            t60_real.max(),
            "lower bound of t60_real: ",
            t60_real.min(),
        )
        print(
            "upper bound of edt_real: ",
            edt_real.max(),
            "lower bound of edt_real: ",
            edt_real.min(),
        )
        print(
            "upper bound of c80_real: ",
            c80_real.max(),
            "lower bound of c80_real: ",
            c80_real.min(),
        )
        print(
            "upper bound of c50_real: ",
            c50_real.max(),
            "lower bound of c50_real: ",
            c50_real.min(),
        )
        print(
            "upper bound of d50_real: ",
            d50_real.max(),
            "lower bound of d50_real: ",
            d50_real.min(),
        )
        print(
            "upper bound of ts_real: ",
            ts_real.max(),
            "lower bound of ts_real: ",
            ts_real.min(),
        )
        print(
            "upper bound of volume_real: ",
            volume_real.max(),
            "lower bound of volume_real: ",
            volume_real.min(),
        )
        print(
            "upper bound of dist_src_real: ",
            dist_src_real.max(),
            "lower bound of dist_src_real: ",
            dist_src_real.min(),
        )

        # Insert 'real_recording' column with 1 for real data
        if "real_recording" not in train_real_manifest.columns:
            train_real_manifest.insert("real_recording", 1)
        else:
            train_real_manifest["real_recording"] = 1
        if "real_recording" not in val_real_manifest.columns:
            val_real_manifest.insert("real_recording", 1)
        else:
            val_real_manifest["real_recording"] = 1
        if "real_recording" not in test_real_manifest.columns:
            test_real_manifest.insert("real_recording", 1)
        else:
            test_real_manifest["real_recording"] = 1

        train_real_manifest.to_csv(train_real_manifest_path, index=False)
        val_real_manifest.to_csv(val_real_manifest_path, index=False)
        test_real_manifest.to_csv(test_real_manifest_path, index=False)
        print("Updated and saved real recording manifests.")

    # Plotting histograms for synthetic data parameters (if manifest was loaded)
    if not manifest.empty:
        print("\nPlotting histograms for synthetic data parameters...")
        param_list_to_plot = {
            "STI": sti,
            "ALCONS": alcons,
            "T60": t60,
            "EDT": edt,
            "C80": c80,
            "C50": c50,
            "D50": d50,
            "Ts": ts,
        }
        plt.rcParams["font.family"] = "serif"
        for name, data_array in param_list_to_plot.items():
            fig, ax = plt.subplots()
            sns.histplot(data_array, bins=50, color="blue", ax=ax)
            ax.set_title(f"{name} distribution")
            ax.set_xlabel(name)
            ax.set_ylabel("Number of samples")
            # plt.show() # Uncomment to display plots one by one
            fig.savefig(f"./{name}_distribution.pdf")  # Save plots instead of showing
            plt.close(fig)
            print(f"Saved {name}_distribution.pdf")

        # Split again the manifest into train, test and val
        # Need original lengths if manifest was re-read from individual files
        # This assumes train_manifest, test_manifest, val_manifest were loaded at the start
        # If not, these lengths will be 0 or undefined.
        len_train_orig = 0
        len_test_orig = 0
        if os.path.exists(train_manifest_alt_path):
            len_train_orig = len(pd.read_csv(train_manifest_alt_path))
        if os.path.exists(test_manifest_alt_path):
            len_test_orig = len(pd.read_csv(test_manifest_alt_path))

        if len_train_orig > 0:  # Proceed only if original files were loaded
            train_manifest_ = manifest[:len_train_orig]
            test_manifest_ = manifest[len_train_orig : len_train_orig + len_test_orig]
            val_manifest_ = manifest[len_train_orig + len_test_orig :]

            train_manifest_.to_csv(train_manifest_alt_path, index=False)
            test_manifest_.to_csv(test_manifest_alt_path, index=False)
            val_manifest_.to_csv(val_manifest_alt_path, index=False)
            print("Re-split and saved synthetic manifests.")
        else:
            print(
                "Skipping re-splitting of synthetic manifest as original files were not loaded or empty."
            )

    # Iteration through retransmission data
    print("\nStarting retransmission data iteration (exploratory)...")
    path_retrans_base = (
        "/home/lucianius/Data/Datasets/Librispeech_test_clean_retransmission"
    )
    path_rir_base = "/home/lucianius/Data/Datasets/BUT_ReverbDB"

    if os.path.exists(path_retrans_base) and os.path.exists(path_rir_base):
        num_files_retrans = 10
        file_batch_counter = 0
        folderNos_retrans = [4, 6, 7, 8, 9]
        # ThTtDistRcvOriSrc_label_list = [] # Renamed, was ThTtDistRcvOriSrc_label in notebook
        random.seed(3407)

        for folderNo_item in folderNos_retrans:
            print(f"Processing retrans folder number {folderNo_item}...")
            # checkfolder_BUT might return a path segment like "/D105_binaural"
            # This assumes checkfolder_BUT is available and works as expected
            try:
                path_retrans_current_folder_segment = checkfolder_BUT(folderNo_item)
            except Exception as e:
                print(
                    f"Error with checkfolder_BUT for folderNo {folderNo_item}: {e}. Skipping this folder."
                )
                continue

            path_retrans_current_folder = (
                path_retrans_base + path_retrans_current_folder_segment
            )
            # path_rir_current_folder = path_rir_base + path_rir_current_folder_segment # path_rir_current_folder was 'path' in notebook, unused in loop

            if not os.path.isdir(path_retrans_current_folder):
                print(
                    f"Retransmission sub-path not found or not a directory: {path_retrans_current_folder}. Skipping."
                )
                continue

            speaker_id_folders = os.listdir(path_retrans_current_folder)
            speaker_id_folders.sort()

            for speaker_foldername in speaker_id_folders:
                if speaker_foldername.startswith("SpkID"):
                    print(f"Processing speaker {speaker_foldername}...")
                    for mic_config_no in range(1, 32):  # Corresponds to "01" to "31"
                        retransed_audio_base_path = os.path.join(
                            path_retrans_current_folder,
                            speaker_foldername,
                            str(mic_config_no).zfill(2),
                            "english/LibriSpeech/test-clean/",
                        )
                        if not os.path.isdir(retransed_audio_base_path):
                            # print(f"Audio base path not found: {retransed_audio_base_path}") # Can be verbose
                            continue

                        retrans_librispeech_folder = Path(retransed_audio_base_path)
                        retrans_extension = ".wav"
                        retrans_matching_files = list(
                            retrans_librispeech_folder.rglob(f"*{retrans_extension}")
                        )
                        retrans_matching_files = [
                            str(x) for x in retrans_matching_files
                        ]
                        retrans_matching_files.sort()

                        if not retrans_matching_files:
                            continue

                        selected_audio_paths = retrans_matching_files[
                            file_batch_counter
                            * num_files_retrans : (file_batch_counter + 1)
                            * num_files_retrans
                        ]
                        if (
                            len(selected_audio_paths) < num_files_retrans
                            and retrans_matching_files
                        ):  # if last batch is too small
                            selected_audio_paths = random.sample(
                                retrans_matching_files,
                                min(num_files_retrans, len(retrans_matching_files)),
                            )

                        # print(f"Selected for {speaker_foldername}, mic_config {mic_config_no}: {selected_audio_paths}") # Can be very verbose
                        file_batch_counter += 1
        print("Retransmission data iteration finished.")
    else:
        print(
            "Base paths for retransmission or RIR data not found. Skipping retransmission iteration."
        )


if __name__ == "__main__":
    main()
