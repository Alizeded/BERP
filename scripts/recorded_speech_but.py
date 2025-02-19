import rootutils
import torch

torch.manual_seed(2036)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.preprocessing import RIRutils as iru  # noqa: E402

trans_path = "/home/lucianius/Data/Datasets/Librispeech_test_clean_retransmission"
rir_path = "/home/lucianius/Data/Datasets/BUT_ReverbDB"
# savepath = "./data/BUT_real_recording_5580samples"
savepath = "./data/BUT_real_recording_11160samples"

iru.readRIR_BUT_real_recording(
    path_retrans=trans_path, path_rir=rir_path, num_files=40, savepath=savepath
)
