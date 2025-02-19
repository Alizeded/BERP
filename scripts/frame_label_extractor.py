import os

import rootutils
import torch
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

mixed_speech_label_path = "./data/mixed_speech/mixed_speech_label.data"
mixed_speech_label_list = os.listdir(mixed_speech_label_path)
mixed_speech_label_list.sort()

mixed_speech_downsampled_label_path = (
    "./data/mixed_speech/mixed_speech_downsampled_label.data"
)

if not os.path.exists(mixed_speech_downsampled_label_path):
    os.makedirs(mixed_speech_downsampled_label_path)


def downsample_label(label_path):
    label = torch.load(label_path)
    frame_len = int(0.5 * 16000)  # 0.5
    frame_num = int(label.numel() // frame_len)
    frame_aggregated = torch.zeros(1, frame_num)
    for n in range(frame_num):
        frame = label[:, n * frame_len : (n + 1) * frame_len]
        frame = frame.float()  # from int8 to float32
        frame_num_occ = frame.mean().round().int()
        frame_aggregated[0, n] = frame_num_occ.to(torch.int8)
    return frame_aggregated


for i in tqdm(range(len(mixed_speech_label_list))):
    label_path = mixed_speech_label_path + "/" + mixed_speech_label_list[i]
    downsampled_label = downsample_label(label_path)
    downsampled_label_name = mixed_speech_label_list[i]
    torch.save(
        downsampled_label,
        os.path.join(mixed_speech_downsampled_label_path, downsampled_label_name),
    )
