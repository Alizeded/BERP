import rootutils
import torch

torch.manual_seed(2036)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.preprocessing import RIRutils as iru  # noqa: E402

trans_path = "/home/lucianius/Data/Datasets/Librispeech_test_clean_retransmission"
rir_path = "/home/lucianius/Data/Datasets/BUT_ReverbDB"
savepath = "./data/BUT_retrans"
iru.readRIR_BUT_retrans(
    path_retrans=trans_path, path_rir=rir_path, num_files=1, savepath=savepath
)
