import argparse
import os

import librosa
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


def calculate_total_audio_length(dataset_path):
    total_length = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=60),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        SpinnerColumn(),
        MofNCompleteColumn(),
    ) as progress:
        # count the total number of wav files in the dataset
        total_num_files = len(
            [
                file
                for root, _dirs, files in os.walk(dataset_path)
                for file in files
                if file.endswith(".wav")
            ]
        )

        task = progress.add_task(
            "Calculating total audio length", total=total_num_files, filename=""
        )

        for root, _dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(".wav"):
                    progress.update(task, advance=1, filename=file)
                    file_path = os.path.join(root, file)
                    audio, fs = torchaudio.load(file_path)
                    audio_length = librosa.get_duration(y=audio, sr=fs)
                    total_length += audio_length

        total_length_hours = total_length / 3600

    # save the total length in hours to a file in the dataset directory
    with open(os.path.join(dataset_path, "total_audio_length.txt"), "w") as f:
        f.write(f"{total_length_hours:.2f}")
    return total_length_hours


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    args = parser.parse_args()

    total_length_hours = calculate_total_audio_length(args.dataset_path)
    print(f"Total audio length in hours: {total_length_hours:.2f}")
