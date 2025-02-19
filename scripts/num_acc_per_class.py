import argparse

import pandas as pd
import torch
from sklearn.metrics import classification_report


def per_class_accuracy(path):
    results = torch.load(path, weights_only=True)
    gts = []
    preds = []
    for r in results:
        pred_label = r["pred_label"]
        groundtruth = r["groundtruth"]
        padding_mask = r["padding_mask"]
        pred_label = torch.masked_select(pred_label, ~padding_mask)
        groundtruth = torch.masked_select(groundtruth, ~padding_mask)
        gts.extend(groundtruth.tolist())
        preds.extend(pred_label.tolist())

    report = classification_report(gts, preds)
    print(report)
    df = pd.DataFrame({"gt": gts, "preds": preds})

    df["distance"] = abs(df["gt"] - df["preds"])

    df_grouped = df.groupby("gt", as_index=False).agg(
        distance_mean=("distance", "mean"), count=("distance", "count")
    )

    print(df_grouped)

    # Macro-averaged MAE
    macro_average_mae = df_grouped["distance_mean"].mean()
    print(macro_average_mae)

    # weighted averaged mae
    weighted_mae = (
        df_grouped["distance_mean"] * df_grouped["count"]
    ).sum() / df_grouped["count"].sum()
    print(weighted_mae)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    per_class_accuracy(args.path)
