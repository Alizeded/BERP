import ast
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.model_selection as ms
import torch
import torchaudio

from scripts.synthesize_rir_speech import synthesize_reverb_speech
from scripts.synthesize_speech_noise import synthesize_mixed_speech
from src.preprocessing import RIRutils as iru

# %% random seed fix
torch.manual_seed(2036)
random.seed(2036)


# %% Function definitions (from various cells)
def volume_distri(manifest: pd.DataFrame, rm_ID_num: int):
    volume_stat = []
    for i in range(1, rm_ID_num + 1):
        volume_rmID = manifest[manifest["roomID"] == "rmID_" + str(i)].iloc[0]
        volume_stat.append(
            [
                torch.tensor(ast.literal_eval(volume_rmID["volume"]))
                .prod()
                .round(decimals=0)
                .tolist(),
                volume_rmID["roomID"],
            ]
        )
    volume_stat = pd.DataFrame(volume_stat, columns=["volume", "roomID"])
    return volume_stat.sort_values("volume", ignore_index=True)


def data_downsample(manifest: pd.DataFrame, N: int):
    """
    downsample the manifest
    """
    N = int(N)
    manifest_downsample = manifest.sample(n=N, random_state=2036)
    return manifest_downsample


def data_upsample(manifest: pd.DataFrame, N: int):
    """
    upsample the manifest
    """
    N = int(N)
    repeated_times = N // len(manifest)
    manifest_upsample = pd.concat([manifest] * repeated_times, ignore_index=True)
    manifest_upsample = pd.concat(
        [
            manifest_upsample,
            manifest.sample(n=N - len(manifest_upsample), random_state=2036),
        ],
        ignore_index=True,
    )
    return manifest_upsample


def audio_len_calc_librispeech(matching_file: str):  # Renamed to avoid conflict
    audio, fs = torchaudio.load(matching_file)
    length = audio.shape[-1]
    return matching_file, length


def audio_len_calc_filtered(matching_files: str):  # Renamed to avoid conflict
    audio, fs = torchaudio.load(matching_files)
    length = audio.shape[-1]
    if length > 200528 and length < 269336:
        return matching_files, length
    return None  # Explicitly return None if condition not met


# %% Main script execution
if __name__ == "__main__":
    # ------------------- prepare the noise -------------------
    # %% These sections create initial noise datasets.
    # DEMAND
    print("Preparing DEMAND noise dataset...")
    savepath_demand = "./data/noise_dataset"
    Path(savepath_demand).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(savepath_demand, "noise.metadata")).mkdir(
        parents=True, exist_ok=True
    )
    Path(os.path.join(savepath_demand, "noise.data")).mkdir(parents=True, exist_ok=True)
    DEMANDnoise_path = os.path.expanduser("~/Data/Datasets/DEMAND_noise")
    if os.path.exists(DEMANDnoise_path):
        iru.readNoise_DEMAND(DEMANDnoise_path, savepath_demand)
    else:
        print(f"DEMAND noise path not found: {DEMANDnoise_path}, skipping.")

    # BUT
    print("Preparing BUT noise dataset...")
    savepath_but_noise = "./data/noise_dataset"  # Same savepath as DEMAND
    Path(savepath_but_noise).mkdir(parents=True, exist_ok=True)  # Ensure creation
    Path(os.path.join(savepath_but_noise, "noise.metadata")).mkdir(
        parents=True, exist_ok=True
    )
    Path(os.path.join(savepath_but_noise, "noise.data")).mkdir(
        parents=True, exist_ok=True
    )
    BUT_path_noise = os.path.expanduser("~/Data/Datasets/BUT_ReverbDB")
    if os.path.exists(BUT_path_noise):
        iru.readNoise_BUT(BUT_path_noise, savepath_but_noise, csv_savemode="a")
    else:
        print(f"BUT ReverbDB path not found for noise: {BUT_path_noise}, skipping.")
    print("Noise preparation potentially completed.")

    # ------------------- read the noise -------------------
    print("Reading noise dataset...")
    noisepath_label = "./data/noise_dataset/noise.metadata/noise_label.csv"
    noise_database = []
    if os.path.exists(noisepath_label):
        noise_label = pd.read_csv(noisepath_label)
        noisepath_data = "./data/noise_dataset/noise.data"
        for i in range(0, len(noise_label["filename"])):
            noise_file_path = os.path.join(noisepath_data, noise_label["filename"][i])
            if os.path.exists(noise_file_path):
                noise, fs = torchaudio.load(noise_file_path)
                noise_database.append(noise)
            else:
                print(f"Noise file not found: {noise_file_path}")
        random.shuffle(noise_database)
        print(f"Loaded {len(noise_database)} noise files.")
    else:
        print(f"Noise label file not found: {noisepath_label}. Skipping noise loading.")
        print("Ensure noise preparation steps were run or noise data exists.")

    # ------------------- RIR Dataset Aggregation -------------------
    # %% These sections read individual RIR datasets and aggregate them with label annotations.
    print("Aggregating RIR datasets (this might take a while)...")
    torch.set_default_dtype(torch.float64)
    savepath_rir_aggregated = "./data/RIR_aggregated"
    Path(savepath_rir_aggregated).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(savepath_rir_aggregated, "RIR.metadata")).mkdir(
        parents=True, exist_ok=True
    )
    Path(os.path.join(savepath_rir_aggregated, "RIR.data")).mkdir(
        parents=True, exist_ok=True
    )

    # %% Arni RIR dataset
    Arni_path = os.path.expanduser("~/Data/Datasets/Arni_RIR_dataset")
    if os.path.exists(Arni_path):
        iru.readRIR_Arni(Arni_path, savepath_rir_aggregated, csv_savemode="w")
    else:
        print(f"Arni RIR path not found: {Arni_path}, skipping.")

    # %% Motus RIR dataset
    Motus_path = os.path.expanduser("~/Data/Datasets/Motus_RIR")
    if os.path.exists(Motus_path):
        iru.readRIR_Motus(Motus_path, savepath_rir_aggregated, csv_savemode="a")
    else:
        print(f"Motus RIR path not found: {Motus_path}, skipping.")

    # %% BUT RIR dataset
    BUT_path_rir = os.path.expanduser("~/Data/Datasets/BUT_ReverbDB")
    if os.path.exists(BUT_path_rir):
        iru.readRIR_BUT(BUT_path_rir, savepath_rir_aggregated, csv_savemode="a")
    else:
        print(f"BUT ReverbDB path not found for RIR: {BUT_path_rir}, skipping.")

    # %% ACE RIR dataset
    ACE_path = os.path.expanduser("~/Data/Datasets/ACE_RIR")
    if os.path.exists(ACE_path):
        iru.readRIR_ACE(ACE_path, savepath_rir_aggregated, csv_savemode="a")
    else:
        print(f"ACE RIR path not found: {ACE_path}, skipping.")

    # %% OpenAIR RIR dataset
    common_path_openair = os.path.expanduser("~/Data/Datasets/OpenAIR")
    # root_path_openair = "./data/RIR_aggregated" # This seems to be the savepath itself in the notebook
    if os.path.exists(common_path_openair):
        iru.readRIR_OpenAIR(
            common_path_openair,
            savepath_rir_aggregated,
            savepath_rir_aggregated,
            csv_savemode="a",
        )
    else:
        print(f"OpenAIR path not found: {common_path_openair}, skipping.")
    print("RIR dataset aggregation potentially completed.")

    # %% Load the aggregated manifest (if the above was run)
    rir_manifest_path = "./data/RIR_aggregated/RIR.metadata/ThTtDistRcvOriSrc_label.csv"
    if os.path.exists(rir_manifest_path):
        rir_manifest = pd.read_csv(rir_manifest_path)
        print("Loaded aggregated RIR manifest.")

        # For first data balancing, we use equal numbers of every room to investigate the volume distribution
        # balance the RIR dataset for equal numbers of rm ID (per 1, for investigation)
        rir_manifest_ID_sum_investigation = pd.DataFrame(
            columns=["RIR", "roomID", "Th", "Tt", "volume", "distRcv", "oriSrc"]
        )
        if not rir_manifest.empty:
            # Determine the max room ID from the manifest dynamically if possible or assume 39
            # This part assumes roomIDs are like "rmID_1", "rmID_2", ..., "rmID_39"
            # A more robust way would be to parse existing roomIDs
            present_room_ids = (
                rir_manifest["roomID"]
                .str.extract(r"rmID_(\d+)")
                .astype(int)
                .max()
                .iloc[0]
            )
            max_room_id = (
                present_room_ids if pd.notna(present_room_ids) else 0
            )  # Fallback if no rooms found

            for id_val in range(
                1, int(max_room_id) + 1
            ):  # Iterate up to actual max_room_id
                rir_manifest_id = rir_manifest[
                    rir_manifest["roomID"] == "rmID_" + str(id_val)
                ]
                if not rir_manifest_id.empty:
                    rir_manifest_ID_sum_investigation = pd.concat(
                        [rir_manifest_ID_sum_investigation, rir_manifest_id.iloc[0:1]],
                        ignore_index=True,
                    )
            print(
                "Initial RIR manifest for volume distribution investigation (1 per room):"
            )
            print(rir_manifest_ID_sum_investigation.head())

            # Volume distribution analysis
            if not rir_manifest_ID_sum_investigation.empty:
                # Recalculate max_room_id for this specific subset if necessary, or use the previous one
                # Assuming rir_manifest_ID_sum_investigation contains samples from all relevant rooms up to 39
                # Or determine max_room_id from this specific dataframe
                investigation_max_room_id = (
                    rir_manifest_ID_sum_investigation["roomID"]
                    .str.extract(r"rmID_(\d+)")
                    .astype(int)
                    .max()
                    .iloc[0]
                )
                investigation_max_room_id = (
                    investigation_max_room_id
                    if pd.notna(investigation_max_room_id)
                    else 0
                )

                volume_stat_investigation = volume_distri(
                    rir_manifest_ID_sum_investigation, int(investigation_max_room_id)
                )
                print("Volume statistics (sorted by volume):")
                print(volume_stat_investigation)
            else:
                print(
                    "rir_manifest_ID_sum_investigation is empty, cannot perform volume_distri."
                )
        else:
            print("rir_manifest is empty, cannot perform balancing for investigation.")

        # NOTE: From this investigation, we determine to augment the data of which volume
        # based on segmented levels of volume to obtain more natural distribution.

        # We processed the data augmentation to make the data better fitted for normal distribution.
        # With regard to volume within 400, we do not do any additional data augmentation,
        # for volume in 400 and 7000, we do additional data augmentation by 2 times.
        # For volume in 7000 and 10000, we do additional data augmentation by 4 times.
        # We assume that the volume beyond 10000 is rare for daily life usage.
        # By using aforementioned data augmentation strategy, we make the whole data follow
        # the normal distribution more and more naturally.

        # %% This section performs the actual resampling and augmentation based on volume.

        print("Performing RIR data resampling and augmentation...")
        rir_manifest_ID_resampled = pd.DataFrame(
            columns=["RIR", "roomID", "Th", "Tt", "volume", "distRcv", "oriSrc"]
        )
        N_base = 560  # Base number of samples per category

        if not rir_manifest.empty:
            # Determine the max room ID from the manifest dynamically
            resample_max_room_id = (
                rir_manifest["roomID"]
                .str.extract(r"rmID_(\d+)")
                .astype(int)
                .max()
                .iloc[0]
            )
            resample_max_room_id = (
                resample_max_room_id if pd.notna(resample_max_room_id) else 0
            )

            for id_val in range(1, int(resample_max_room_id) + 1):
                N = N_base + random.randint(-20, 20)  # Slight randomization for N
                rir_manifest_id_num = rir_manifest[
                    rir_manifest["roomID"] == "rmID_" + str(id_val)
                ]
                if rir_manifest_id_num.empty:
                    continue

                # Calculate volume for the current room ID
                # Ensure 'volume' column exists and contains valid string representations of lists/numbers
                try:
                    rir_volume_str = rir_manifest_id_num["volume"].iloc[0]
                    if isinstance(rir_volume_str, str):
                        rir_volume_val = (
                            torch.tensor(ast.literal_eval(rir_volume_str))
                            .prod()
                            .round(decimals=0)
                            .item()
                        )
                    else:  # Assuming it might already be a number
                        rir_volume_val = (
                            torch.tensor(rir_volume_str).prod().round(decimals=0).item()
                        )
                except Exception as e:
                    print(
                        f"Error processing volume for rmID_{id_val}: {e}. Skipping this room."
                    )
                    continue

                target_N = N
                if rir_volume_val < 400:
                    target_N = N
                elif 400 <= rir_volume_val < 7000:
                    target_N = N * 3.5
                elif 7000 <= rir_volume_val < 10000:
                    target_N = N * 4
                # Commented out further augmentation from notebook as it was also commented there
                # elif 10000 <= rir_volume_val < 20000:
                #     target_N = N * 6
                # elif rir_volume_val >= 20000:
                #     target_N = N * 4
                else:  # For volumes >= 10000 (if not covered by commented sections)
                    # Keep N or apply a default if not specified for >10000 and not in ranges above
                    # Based on notebook logic, these large volumes are considered rare and might not be upsampled
                    target_N = N  # Default to N if not in other categories

                target_N = int(target_N)  # Ensure N is an integer

                if (
                    len(rir_manifest_id_num) == 0
                ):  # Should be caught by earlier check but as safeguard
                    continue
                elif len(rir_manifest_id_num) > target_N:
                    rir_manifest_id_processed = data_downsample(
                        rir_manifest_id_num, target_N
                    )
                elif len(rir_manifest_id_num) < target_N:
                    rir_manifest_id_processed = data_upsample(
                        rir_manifest_id_num, target_N
                    )
                else:  # len(rir_manifest_id_num) == target_N
                    rir_manifest_id_processed = rir_manifest_id_num.copy()

                rir_manifest_ID_resampled = pd.concat(
                    [rir_manifest_ID_resampled, rir_manifest_id_processed],
                    ignore_index=True,
                )
        else:
            print("rir_manifest is empty, cannot perform resampling.")

        if not rir_manifest_ID_resampled.empty:
            output_augmented_csv_path = (
                "./data/RIR_aggregated/RIR.metadata/RIRLabelAugment.csv"
            )
            rir_manifest_ID_resampled.to_csv(output_augmented_csv_path, index=False)
            print(f"Saved resampled RIR manifest to {output_augmented_csv_path}")
        else:
            print("No resampled data to save for RIRLabelAugment.csv.")

    else:
        print(
            f"Aggregated RIR manifest not found: {rir_manifest_path}. Skipping RIR processing."
        )
        print("Ensure RIR aggregation steps were run or the manifest file exists.")

    # %% Load the resampled manifest
    print("Loading augmented RIR manifest (RIRLabelAugmentV2.csv)...")
    resampled_rir_manifest_path = (
        "./data/RIR_aggregated/RIR.metadata/RIRLabelAugmentV2.csv"
    )
    if os.path.exists(resampled_rir_manifest_path):
        rir_manifest_ID_resampled = pd.read_csv(resampled_rir_manifest_path)
        print("Loaded RIRLabelAugmentV2.csv")

        rir_manifest_analysis = rir_manifest_ID_resampled.copy()
        # obtain RIR from which dataset
        rir_manifest_analysis["RIR"] = rir_manifest_analysis["RIR"].apply(
            lambda x: x.split("_")[0]
        )

        # Ensure output directory for figures exists
        figure_path = "./data/Figure"
        Path(figure_path).mkdir(parents=True, exist_ok=True)

        # --- Volume Analysis ---
        print("Performing Volume analysis and plotting...")

        # Calculate actual volume values for plotting
        # Handle cases where 'volume' might already be numeric or needs ast.literal_eval
        def parse_volume(v_str):
            try:
                if isinstance(v_str, int | float):
                    return v_str
                return (
                    torch.tensor(ast.literal_eval(v_str))
                    .prod()
                    .round(decimals=0)
                    .item()
                )
            except Exception:
                return None  # Or some default / error indicator

        rir_manifest_analysis["volume_val"] = rir_manifest_analysis["volume"].apply(
            parse_volume
        )
        rir_manifest_analysis.dropna(
            subset=["volume_val"], inplace=True
        )  # Drop rows where volume couldn't be parsed

        fig_vol, ax_vol = plt.subplots()
        plt.rcParams["font.family"] = "serif"
        sns.histplot(
            x=rir_manifest_analysis["volume_val"],
            hue=rir_manifest_analysis["RIR"],
            multiple="stack",
            bins=50,
            ax=ax_vol,
        )
        ax_vol.set_xlabel("Volume [m$^3$]", fontsize=16, fontname="serif")
        ax_vol.set_ylabel("Number of RIRs", fontsize=16, fontname="serif")
        fig_vol.savefig(os.path.join(figure_path, "hist_volume.pdf"), format="pdf")
        plt.close(fig_vol)

        volume_numeric = torch.tensor(
            rir_manifest_analysis["volume_val"].values
        )  # Ensure float for log
        volume_log10 = torch.log10(volume_numeric)

        fig_logvol, ax_logvol = plt.subplots()
        plt.rcParams["font.family"] = "serif"
        sns.histplot(
            x=volume_log10.numpy(),  # sns typically works well with numpy arrays
            hue=rir_manifest_analysis["RIR"],  # Ensure this aligns with the x data
            multiple="stack",
            bins=50,
            ax=ax_logvol,
        )
        ax_logvol.set_xlabel(
            "log$_{10}$(Volume) [m$^3$]", fontname="serif", fontsize=15
        )
        ax_logvol.set_ylabel("Number of RIRs", fontname="serif", fontsize=15)
        fig_logvol.savefig(
            os.path.join(figure_path, "hist_logVolume.pdf"), format="pdf"
        )
        plt.close(fig_logvol)

        print(
            "The upper bound of the log10(volume) is: ",
            round(volume_log10.max().item(), 4),
        )
        print(
            "The lower bound of the log10(volume) is: ",
            round(volume_log10.min().item(), 4),
        )

        # --- Sound Source distance of source Analysis ---
        print("Performing Sound Source Distance analysis and plotting...")
        if pd.api.types.is_numeric_dtype(rir_manifest_analysis["distRcv"]):
            distSrc = torch.tensor(rir_manifest_analysis["distRcv"].values)

            fig_dist, ax_dist = plt.subplots()
            plt.rcParams["font.family"] = "serif"
            sns.histplot(
                x=rir_manifest_analysis["distRcv"],
                hue=rir_manifest_analysis["RIR"],  # Use the modified manifest for hue
                multiple="stack",
                bins=60,
                ax=ax_dist,
            )
            ax_dist.set_xlabel("Distance of source [m]", fontname="serif", fontsize=15)
            ax_dist.set_ylabel("Number of RIRs", fontname="serif", fontsize=15)
            fig_dist.savefig(
                os.path.join(figure_path, "hist_distSrc.pdf"), format="pdf"
            )
            plt.close(fig_dist)

            print(
                "The upper bound of the distance is: ", round(distSrc.max().item(), 4)
            )
            print(
                "The lower bound of the distance is: ", round(distSrc.min().item(), 4)
            )
        else:
            print("'distRcv' column is not numeric. Skipping distance analysis.")

        # --- T60 parameter Analysis ---
        # Note: Notebook uses 'Tt' for histplot and 'T60' for print stats.
        # If 'T60' is a separate column and intended, change "Tt" to "T60" below.
        t60_col_name = "Tt"
        if "T60" in rir_manifest_analysis.columns and pd.api.types.is_numeric_dtype(
            rir_manifest_analysis["T60"]
        ):
            t60_col_name_stats = "T60"

        if t60_col_name_stats:  # If a suitable column for stats was found
            print(
                f"Performing T60 ({t60_col_name_stats}) parameter analysis and plotting (using {t60_col_name} for plot)..."
            )
            Tt_tensor = torch.tensor(
                rir_manifest_analysis[t60_col_name].values
            )  # For plot
            # Tt_unitary_norm = unitary_norm(Tt_tensor).round(decimals=4).tolist() # If needed

            fig_t60, ax_t60 = plt.subplots()
            plt.rcParams["font.family"] = "serif"
            sns.histplot(
                x=rir_manifest_analysis[t60_col_name],
                hue=rir_manifest_analysis["RIR"],
                multiple="stack",
                bins=60,
                ax=ax_t60,
            )
            ax_t60.set_xlabel(
                f"{t60_col_name} (likely T60) [s]", fontname="serif", fontsize=15
            )  # Clarify label
            ax_t60.set_ylabel("Number of RIRs", fontname="serif", fontsize=15)
            fig_t60.savefig(os.path.join(figure_path, "hist_T60.pdf"), format="pdf")
            plt.close(fig_t60)

            # Stats using the determined column
            Td_stats_tensor = torch.tensor(
                rir_manifest_analysis[t60_col_name_stats].values
            )
            print(
                f"The upper bound of {t60_col_name_stats} is: ",
                round(Td_stats_tensor.max().item(), 5),
            )
            print(
                f"The lower bound of {t60_col_name_stats} is: ",
                round(Td_stats_tensor.min().item(), 5),
            )
            print(
                f"The mean of {t60_col_name_stats} is: ",
                round(Td_stats_tensor.mean().item(), 5),
            )

        # Saving the (potentially modified for analysis) resampled manifest
        # The notebook saves rir_manifest_ID_resampled to a specific path.
        # Here we save the one used for analysis, which might have slight changes (like 'volume_val')
        # Or save the original rir_manifest_ID_resampled if preferred.
        # Notebook cell 36 saves `rir_manifest_ID_resampled`
        # final_save_path = os.path.expanduser("~/workspace/acoustic/data/RIR_aggregated/RIR.metadata/RIRLabelAugmentV2.csv")
        # Path(os.path.dirname(final_save_path)).mkdir(parents=True, exist_ok=True)
        # rir_manifest_ID_resampled.to_csv(final_save_path, index=False)
        # print(f"Saved RIR manifest (original from RIRLabelAugmentV2.csv) to {final_save_path}")

    else:
        print(
            f"Augmented RIR manifest not found: {resampled_rir_manifest_path}. Skipping analysis."
        )
        print(
            "Ensure RIR augmentation steps were run or the RIRLabelAugmentV2.csv file exists."
        )

    # --- LibriSpeech Processing ---
    # %% We randomly sample the speech signals from LibriSpeech dataset to match the count of the RIRs
    # to synthesize the noisy reverberant speech signals.

    print("Processing LibriSpeech dataset to get audio lengths...")
    librispeech_root_path = "./data/LibriSpeech/train-clean-360"
    librispeech_label_csv_path = "./data/LibriSpeech/LibriSpeech_label.csv"
    Path(os.path.dirname(librispeech_label_csv_path)).mkdir(parents=True, exist_ok=True)

    if os.path.exists(librispeech_root_path):
        librispeech_folder = Path(librispeech_root_path)
        extension = ".flac"
        matching_files_librispeech = [
            str(x) for x in librispeech_folder.rglob(f"*{extension}")
        ]

        if matching_files_librispeech:
            librispeech_info_list = []
            # Limiting threads for potentially I/O bound task on many small files.
            num_cpus = os.cpu_count()
            max_workers_libri = min(
                8, num_cpus - 1 if num_cpus > 1 else 1
            )  # Sensible default

            with ThreadPoolExecutor(max_workers=max_workers_libri) as executor:
                results = list(
                    executor.map(audio_len_calc_librispeech, matching_files_librispeech)
                )

            librispeech_info = pd.DataFrame(results, columns=["filename", "length"])
            librispeech_info.to_csv(librispeech_label_csv_path, index=False)
            print(f"Saved LibriSpeech manifest to {librispeech_label_csv_path}")
        else:
            print(f"No .flac files found in {librispeech_root_path}")
    else:
        print(
            f"LibriSpeech path not found: {librispeech_root_path}. Skipping LibriSpeech processing."
        )

    # Plotting histogram of audio lengths from LibriSpeech_label.csv
    if os.path.exists(librispeech_label_csv_path):
        librispeech_info_loaded = pd.read_csv(librispeech_label_csv_path)
        audio_len = librispeech_info_loaded["length"]
        plt.figure()  # Create a new figure for this plot
        plt.hist(audio_len, bins=30, color="tomato", ec="black")
        plt.xlabel("Audio Length (samples)")
        plt.ylabel("Frequency")
        plt.title("Histogram of LibriSpeech Audio Lengths")
        plt.show()  # or savefig

        print(
            len(audio_len[(audio_len > 200528) & (audio_len < 269336)])
        )  # used in the paper
        Path(figure_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(
            os.path.join(figure_path, "hist_librispeech_audio_len.pdf"), format="pdf"
        )
        plt.close()
        print("Plotted histogram of LibriSpeech audio lengths.")
    else:
        print(
            f"Cannot plot LibriSpeech audio lengths, file not found: {librispeech_label_csv_path}"
        )

    # NOTE: After this investigation, we decide to use the sequence length between 200528 and 269336
    # to convolve RIR to augment data since this is the most common length of speech in LibriSpeech dataset.
    # For synthesized mixed speech, we select the speech below 20s

    # %% Dataset preparation For RIR estimation (filtering LibriSpeech files by length)
    print("Filtering LibriSpeech files by length...")
    audio_long_manifest_path = "./data/LibriSpeech/audio_long_manifest.csv"
    if os.path.exists(librispeech_root_path):  # Depends on librispeech_root_path
        # Re-list files or use 'matching_files_librispeech' if still in scope and fresh
        librispeech_folder_filter = Path(librispeech_root_path)
        extension_filter = ".flac"
        matching_files_filter = [
            str(x) for x in librispeech_folder_filter.rglob(f"*{extension_filter}")
        ]

        if matching_files_filter:
            audio_info_filtered_list = []
            num_cpus = os.cpu_count()
            max_workers_filter = min(
                8, num_cpus - 1 if num_cpus > 1 else 1
            )  # Sensible default

            with ThreadPoolExecutor(max_workers=max_workers_filter) as executor:
                results = list(
                    executor.map(audio_len_calc_filtered, matching_files_filter)
                )

            audio_info_filtered = [x for x in results if x is not None]
            if audio_info_filtered:
                audio_long_manifest = pd.DataFrame(
                    audio_info_filtered, columns=["filename", "audioLength"]
                )
                audio_long_manifest.to_csv(audio_long_manifest_path, index=False)
                print(
                    f"Saved filtered LibriSpeech manifest (long audio) to {audio_long_manifest_path}"
                )
            else:
                print("No audio files matched the length criteria.")
        else:
            print(f"No .flac files found in {librispeech_root_path} for filtering.")
    else:
        print(
            f"LibriSpeech path not found: {librispeech_root_path}. Skipping filtering."
        )

    # %% --- Illustration of the mixed speech and its frame-level label ---
    print("Illustrating mixed speech example...")
    reverb_mixed_speech_manifest_path = (
        "./data/mixed_speech/reverb_mixed_speech_manifest.csv"
    )
    mixed_speech_data_dir = "./data/mixed_speech/mixed_speech.data/reverb_mixed"
    mixed_speech_label_dir = "./data/mixed_speech/mixed_speech_label.data"

    if os.path.exists(reverb_mixed_speech_manifest_path):
        reverb_mixed_speech_manifest = pd.read_csv(reverb_mixed_speech_manifest_path)
        if len(reverb_mixed_speech_manifest) > 2011:
            example_idx = 2011  # As per notebook

            mixed_speech_file = os.path.join(
                mixed_speech_data_dir,
                reverb_mixed_speech_manifest["mixed_speech"][example_idx],
            )
            mixed_speech_label_file = os.path.join(
                mixed_speech_label_dir,
                reverb_mixed_speech_manifest["mixed_speech_label"][example_idx],
            )

            if os.path.exists(mixed_speech_file) and os.path.exists(
                mixed_speech_label_file
            ):
                mixed_speech, fs_mixed = torchaudio.load(mixed_speech_file)
                mixed_speech_label = torch.load(mixed_speech_label_file)

                # Plotting
                Path(figure_path).mkdir(
                    parents=True, exist_ok=True
                )  # Ensure figure_path exists
                fig_mixed_ill, (ax1_ill, ax2_ill) = plt.subplots(
                    2, 1, sharex=True
                )  # Share x-axis
                t_mixed = torch.arange(0, mixed_speech.shape[-1]) / fs_mixed

                ax1_ill.plot(t_mixed, mixed_speech.reshape(-1), color="cornflowerblue")
                ax1_ill.set_ylabel("Amplitude", fontsize=16, fontname="serif")

                ax2_ill.plot(t_mixed, mixed_speech_label.reshape(-1), color="orange")
                ax2_ill.set_xlabel("Time [s]", fontsize=16, fontname="serif")
                ax2_ill.set_ylabel("Occupancy level", fontsize=16, fontname="serif")

                fig_mixed_ill.savefig(
                    os.path.join(figure_path, "mixed_speech_illustration.pdf"),
                    format="pdf",
                )
                plt.close(fig_mixed_ill)
                print("Saved mixed speech illustration plot.")
            else:
                print(
                    f"Example mixed speech/label file not found for index {example_idx}."
                )
        else:
            print(
                "Reverberant mixed speech manifest does not have enough entries for example."
            )
    else:
        print(
            f"Reverberant mixed speech manifest not found: {reverb_mixed_speech_manifest_path}"
        )

    # %% --- Synthesis Scripts ---

    print("Synthesis of single-source reverb speech...")
    synthesize_reverb_speech()

    print("Synthesis of reverb mixed speech...")
    synthesize_mixed_speech()
    # %% --- Analysis of downsampled mixed speech labels ---
    print("Analyzing downsampled mixed speech labels...")
    mixed_speech_downsampled_label_path = (
        "./data/mixed_speech/mixed_speech_downsampled_label.data"
    )
    if os.path.exists(mixed_speech_downsampled_label_path):
        mixed_speech_downsampled_label_lst = os.listdir(
            mixed_speech_downsampled_label_path
        )
        mixed_speech_downsampled_label_lst.sort()

        if (
            len(mixed_speech_downsampled_label_lst) > 2000
        ):  # Ensure enough files for sampling
            # Example from notebook
            test_label_sample = random.sample(
                mixed_speech_downsampled_label_lst,
                (
                    4000
                    if len(mixed_speech_downsampled_label_lst) >= 4000
                    else len(mixed_speech_downsampled_label_lst)
                ),
            )
            test_one_sample_path = os.path.join(
                mixed_speech_downsampled_label_path, test_label_sample[2000]
            )
            if os.path.exists(test_one_sample_path):
                test_one_sample = torch.load(test_one_sample_path).float()
                print(
                    f"Number of elements in one sample label: {test_one_sample.numel()}"
                )
            else:
                print(f"Sample label file not found: {test_one_sample_path}")

            # Concatenating all labels (can be memory intensive)
            print(
                "Concatenating all downsampled mixed speech labels (this can be memory intensive)..."
            )
            mixed_speech_num_occ = torch.tensor([])
            for i in range(
                len(mixed_speech_downsampled_label_lst)
            ):  # Limit for testing if needed
                label_file = os.path.join(
                    mixed_speech_downsampled_label_path,
                    mixed_speech_downsampled_label_lst[i],
                )
                if os.path.exists(label_file):
                    mixed_speech_num = torch.load(label_file).float()
                    mixed_speech_num_occ = torch.cat(
                        [mixed_speech_num_occ, mixed_speech_num], dim=1
                    )  # Assuming dim=1 is correct
                else:
                    print(f"Label file not found during concatenation: {label_file}")
            if mixed_speech_num_occ.numel() > 0:
                print(f"Shape of concatenated labels: {mixed_speech_num_occ.shape}")
            else:
                print("No labels were concatenated.")
        else:
            print("Not enough downsampled label files for analysis/sampling.")
    else:
        print(
            f"Downsampled mixed speech label path not found: {mixed_speech_downsampled_label_path}"
        )

    # %% ------------------ Split train, val, and test for noisyReverbSpeech ---------------------
    print("Splitting dataset for noisyReverbSpeech...")
    noisy_clean_manifest_base_path = "./data/noiseReverbSpeech"
    noisy_clean_pair_csv = os.path.join(
        noisy_clean_manifest_base_path, "reverbSpeech.metadata.csv"
    )
    val_size_nc = test_size_nc = 2000

    if os.path.exists(noisy_clean_pair_csv):
        noisyCleanPair = pd.read_csv(noisy_clean_pair_csv)

        if len(noisyCleanPair) > (val_size_nc + test_size_nc):
            train_manifest_nc, test_manifest_nc = ms.train_test_split(
                noisyCleanPair,
                test_size=test_size_nc,  # Corrected: test_size first
                random_state=2036,
                shuffle=True,  # Ensure shuffling before split
            )
            # Calculate remaining for train_val, then split val from it
            train_val_size_nc = len(noisyCleanPair) - test_size_nc

            # Split val_size from the train_manifest_nc (which is currently train+val portion)
            # Ensure train_size for the next split is correctly specified
            if len(train_manifest_nc) > val_size_nc:
                train_manifest_nc, val_manifest_nc = ms.train_test_split(
                    train_manifest_nc,
                    test_size=val_size_nc,  # test_size here refers to the validation set size
                    random_state=2036,  # Use same random state for reproducibility of this specific split
                    shuffle=True,
                )
            else:
                print(
                    "Not enough data to create validation set after test set split for noisyCleanPair."
                )
                val_manifest_nc = pd.DataFrame()  # Empty dataframe

            # Shuffle again (optional, as train_test_split can shuffle, but notebook does it)
            train_manifest_nc = train_manifest_nc.sample(
                frac=1, random_state=2036
            ).reset_index(drop=True)
            if not val_manifest_nc.empty:
                val_manifest_nc = val_manifest_nc.sample(
                    frac=1, random_state=2036
                ).reset_index(drop=True)
            test_manifest_nc = test_manifest_nc.sample(
                frac=1, random_state=2036
            ).reset_index(drop=True)

            Path(noisy_clean_manifest_base_path).mkdir(parents=True, exist_ok=True)
            train_manifest_nc.to_csv(
                os.path.join(noisy_clean_manifest_base_path, "train_manifest.csv"),
                index=False,
            )
            if not val_manifest_nc.empty:
                val_manifest_nc.to_csv(
                    os.path.join(noisy_clean_manifest_base_path, "val_manifest.csv"),
                    index=False,
                )
            test_manifest_nc.to_csv(
                os.path.join(noisy_clean_manifest_base_path, "test_manifest.csv"),
                index=False,
            )
            print("Saved train/val/test manifests for noisyReverbSpeech.")
        else:
            print(
                f"Not enough data in {noisy_clean_pair_csv} to split into train/val/test with desired sizes."
            )
    else:
        print(f"Manifest for noisyReverbSpeech not found: {noisy_clean_pair_csv}")

    # %% Load the reverb speech manifests
    train_manifest_reverb = pd.read_csv(
        os.path.join(noisy_clean_manifest_base_path, "train_manifest.csv")
    )
    val_manifest_reverb = pd.read_csv(
        os.path.join(noisy_clean_manifest_base_path, "val_manifest.csv")
    )
    test_manifest_reverb = pd.read_csv(
        os.path.join(noisy_clean_manifest_base_path, "test_manifest.csv")
    )
    print("Loaded reverb speech manifests.")

    # Modify mixed_speech manifests (add numOcc, remove rir_info)
    print("Modifying mixed speech manifests...")
    mixed_speech_manifest_dir = "./data/mixed_speech"
    # mixed_speech_manifest_dir = "./data/Database/mixed_speech" # As per notebook cell 52

    mixed_clean_manifest_csv = os.path.join(
        mixed_speech_manifest_dir, "mixed_speech_manifest.csv"
    )
    mixed_reverb_manifest_csv = os.path.join(
        mixed_speech_manifest_dir, "reverb_mixed_speech_manifest.csv"
    )

    if os.path.exists(mixed_clean_manifest_csv) and os.path.exists(
        mixed_reverb_manifest_csv
    ):
        mixed_clean_manifest = pd.read_csv(mixed_clean_manifest_csv)
        mixed_reverb_manifest = pd.read_csv(mixed_reverb_manifest_csv)

        if "numOcc" in mixed_reverb_manifest.columns:
            num_occ = mixed_reverb_manifest["numOcc"]
            if (
                "numOcc" not in mixed_clean_manifest.columns
            ):  # Avoid inserting if already exists
                mixed_clean_manifest.insert(
                    2, "numOcc", num_occ
                )  # Ensure index 2 is valid
            else:  # Update existing column
                mixed_clean_manifest["numOcc"] = num_occ
        else:
            print("'numOcc' column not found in reverb_mixed_speech_manifest.")

        if "rir_info" in mixed_reverb_manifest.columns:
            mixed_reverb_manifest = mixed_reverb_manifest.drop(columns=["rir_info"])
        else:
            print(
                "'rir_info' column not found in reverb_mixed_speech_manifest for dropping."
            )

        Path(mixed_speech_manifest_dir).mkdir(parents=True, exist_ok=True)
        mixed_clean_manifest.to_csv(mixed_clean_manifest_csv, index=False)
        mixed_reverb_manifest.to_csv(mixed_reverb_manifest_csv, index=False)
        print("Modified and saved mixed speech manifests.")
    else:
        print(
            f"One or both mixed speech manifest files not found: {mixed_clean_manifest_csv}, {mixed_reverb_manifest_csv}"
        )

    # %% ----------- Split train, val and test dataset for mixed_speech_noise -----------
    print("Splitting dataset for mixed_speech_noise...")
    mixed_speech_noise_manifest_base_path = "./data/mixed_speech_noise"

    reverb_mixed_source_csv_for_split = os.path.join(
        mixed_speech_noise_manifest_base_path, "mixed_speech_manifest.csv"
    )

    val_size_msn = test_size_msn = 2000

    if os.path.exists(reverb_mixed_source_csv_for_split):
        reverb_mixed_speech_manifest_to_split = pd.read_csv(
            reverb_mixed_source_csv_for_split
        )

        if len(reverb_mixed_speech_manifest_to_split) > (val_size_msn + test_size_msn):
            train_manifest_msn, test_manifest_msn = ms.train_test_split(
                reverb_mixed_speech_manifest_to_split,
                test_size=test_size_msn,
                random_state=2036,
                shuffle=True,
            )

            if len(train_manifest_msn) > val_size_msn:
                train_manifest_msn, val_manifest_msn = ms.train_test_split(
                    train_manifest_msn,  # This is now train+val portion
                    test_size=val_size_msn,
                    random_state=2036,
                    shuffle=True,
                )
            else:
                print(
                    "Not enough data to create validation set after test set split for mixed_speech_noise."
                )
                val_manifest_msn = pd.DataFrame()

            train_manifest_msn = train_manifest_msn.sample(
                frac=1, random_state=2036
            ).reset_index(drop=True)
            if not val_manifest_msn.empty:
                val_manifest_msn = val_manifest_msn.sample(
                    frac=1, random_state=2036
                ).reset_index(drop=True)
            test_manifest_msn = test_manifest_msn.sample(
                frac=1, random_state=2036
            ).reset_index(drop=True)

            Path(mixed_speech_noise_manifest_base_path).mkdir(
                parents=True, exist_ok=True
            )
            train_manifest_msn.to_csv(
                os.path.join(
                    mixed_speech_noise_manifest_base_path, "train_manifest.csv"
                ),
                index=False,
            )
            if not val_manifest_msn.empty:
                val_manifest_msn.to_csv(
                    os.path.join(
                        mixed_speech_noise_manifest_base_path, "val_manifest.csv"
                    ),
                    index=False,
                )
            test_manifest_msn.to_csv(
                os.path.join(
                    mixed_speech_noise_manifest_base_path, "test_manifest.csv"
                ),
                index=False,
            )
            print("Saved train/val/test manifests for mixed_speech_noise.")
        else:
            print(
                f"Not enough data in {reverb_mixed_source_csv_for_split} for mixed_speech_noise split."
            )
    else:
        print(
            f"Manifest for mixed_speech_noise not found: {reverb_mixed_source_csv_for_split}"
        )
        print(
            "This file might need to be copied from './data/mixed_speech/mixed_speech_manifest.csv' after its modification."
        )

    print("Script execution finished.")
