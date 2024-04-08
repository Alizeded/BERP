import torch
import os
from tqdm import tqdm
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

mixed_speech_label_path = "./data/mixed_speech/mixed_speech_downsampled_label.data"
mixed_speech_label_list = os.listdir(mixed_speech_label_path)
mixed_speech_label_list.sort()

mixed_speech_classified_label_path = "./data/mixed_speech/mixed_speech_class_label.data"

if not os.path.exists(mixed_speech_classified_label_path):
    os.makedirs(mixed_speech_classified_label_path)


def classify_label(label_path):
    label = torch.load(label_path).float()
    label[label < 3] = 1
    label[(label >= 3) & (label < 6)] = 2
    label[(label >= 6) & (label < 9)] = 3
    label[(label >= 9) & (label <= 12)] = 4
    label = label.to(torch.int8)
    return label


for i in tqdm(range(len(mixed_speech_label_list))):
    label_path = mixed_speech_label_path + "/" + mixed_speech_label_list[i]
    classified_label = classify_label(label_path)
    classified_label_name = mixed_speech_label_list[i]
    torch.save(
        classified_label,
        os.path.join(mixed_speech_classified_label_path, classified_label_name),
    )
