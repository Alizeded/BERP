import ast

import numpy as np
import pandas as pd

import src.preprocessing.sim_rir_dict as sim_rir_dict


def match_room_and_distance(dict_data, real_rir_manifest: pd.DataFrame, seed=42):
    """
    Match dict entries with dataframe rows for multiple roomIDs.

    Args:
        dict_data: Dictionary containing multiple roomIDs and their data
        df: DataFrame with roomID and distRcv columns
        seed: Random seed for reproducibility

    Returns:
        DataFrame with matched entries and additional tuples
    """
    manifest = []

    np.random.seed(seed)

    # Process each roomID in the dict
    for i in range(len(real_rir_manifest)):  # noqa: B007
        roomID = real_rir_manifest.iloc[i]["roomID"]
        distRcv = real_rir_manifest.iloc[i]["distRcv"]
        Tt = real_rir_manifest.iloc[i]["Tt"]
        volume_log10_norm = real_rir_manifest.iloc[i]["volume_log10_norm"]
        volume_ns = np.asarray(
            ast.literal_eval(real_rir_manifest.iloc[i]["volume"])
        ).round(0)
        distRcv_norm = real_rir_manifest.iloc[i]["distRcv_norm"]

        matched_room_cases = [case for case in dict_data[roomID].values()]

        distRcvs = [case["distRcv"] for case in matched_room_cases]

        distRcvs = np.asarray(distRcvs)

        matched_idx = np.where(np.isclose(distRcvs, distRcv))[0]
        match_type = "exact"

        if len(matched_idx) == 0:
            # randomly select the closest distance
            matched_idx = np.argmin(np.abs(distRcvs - distRcv))
        elif len(matched_idx) > 1:
            # randomly select one of the matched distances
            matched_idx = np.random.choice(matched_idx, 1)
            match_type = "random"

        matched_index = int(matched_idx.item())

        selected_case = matched_room_cases[matched_index]

        # Append the dict entry to the row of the dataframe
        manifest.append(
            [
                roomID,
                tuple(selected_case["src"]),
                tuple(selected_case["mic"]),
                tuple(selected_case["volume"]),
                Tt,
                selected_case["distRcv"],
                distRcv_norm,
                volume_log10_norm,
                volume_ns,
                match_type,
            ]
        )

    manifest = pd.DataFrame(
        manifest,
        columns=[
            "roomID",
            "srcPos",
            "micPos",
            "volume_dim",
            "T60",
            "distRcv",
            "distRcv_norm",
            "volume_log10_norm",
            "volume",
            "match_type",
        ],
    )

    return manifest


if __name__ == "__main__":

    sim_rir_dict = sim_rir_dict.RoomCases().generate_room_cb()
    train_rir_manifest = pd.read_csv("./data/noiseReverbSpeech/train_manifest_RIR.csv")

    train_rir_manifest = match_room_and_distance(sim_rir_dict, train_rir_manifest)
    train_rir_manifest.to_csv(
        "./data/noiseReverbSpeech/train_manifest_RIR_matched.csv", index=False
    )
