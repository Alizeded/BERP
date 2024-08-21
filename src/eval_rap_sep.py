from typing import Any, Dict, List, Tuple

import hydra
import rootutils
import torch
import torchaudio
import os
import pandas as pd
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    TaskProgressColumn,
    MofNCompleteColumn,
)
from concurrent.futures import ThreadPoolExecutor
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from src.utils.AcousticParameterUtils import (  # noqa: E402
    SparseStochasticIRModel,
    RapidSpeechTransmissionIndex,
    PercentageArticulationLoss,
    EarlyDecayTime,
    Clarity,
    Definition,
    CenterTime,
)

from src.utils.eval_metrics import mae_calculator, corrcoef_calculator  # noqa: E402

from src.utils.unpack import agg_pair  # noqa: E402

from src import utils  # noqa: E402

log = utils.get_pylogger(__name__)


def rir_manifest_segment(cfg: DictConfig) -> List:
    assert cfg.batch_size
    assert cfg.rir_test_path

    rir_segments = []
    segment_size = cfg.batch_size

    rir_test_manifest = pd.read_csv(cfg.rir_test_path)

    for start in range(0, len(rir_test_manifest), segment_size):
        end = start + segment_size
        rir_segment = rir_test_manifest.iloc[start:end].reset_index(
            drop=True, inplace=False
        )
        rir_segments.append(rir_segment)

    return rir_segments


def rap_eval_segment(
    cfg: DictConfig,
    rir_segment: pd.DataFrame,
    TiTd_pred_segment: Dict[str, Any],
    volume_pred_segment: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    segment_size = cfg.batch_size

    # pre-alllocate memory for rap pred and ground truth
    STI_est_seg, STI_gt_seg = torch.zeros(segment_size), torch.zeros(segment_size)
    ALcons_est_seg, ALcons_gt_seg = torch.zeros(segment_size), torch.zeros(segment_size)
    TR_est_seg, TR_gt_seg = torch.zeros(segment_size), torch.zeros(segment_size)
    EDT_est_seg, EDT_gt_seg = torch.zeros(segment_size), torch.zeros(segment_size)
    C80_est_seg, C80_gt_seg = torch.zeros(segment_size), torch.zeros(segment_size)
    C50_est_seg, C50_gt_seg = torch.zeros(segment_size), torch.zeros(segment_size)
    D50_est_seg, D50_gt_seg = torch.zeros(segment_size), torch.zeros(segment_size)
    Ts_est_seg, Ts_gt_seg = torch.zeros(segment_size), torch.zeros(segment_size)

    for n in range(segment_size):
        Ti_hat = TiTd_pred_segment["Th_hat"][n]
        Td_hat = TiTd_pred_segment["Tt_hat"][n]
        volume_log_hat = volume_pred_segment["volume_hat"][n]
        volume_hat = 10**volume_log_hat

        # synthesize the RIR
        rir_synthesizer = SparseStochasticIRModel(
            Ti=Ti_hat, Td=Td_hat, volume=volume_hat, mu=0.0399
        )
        rir_synthesized = rir_synthesizer()

        # obtain the real RIR
        rir_path = rir_segment.iloc[n]["RIR"]
        rir_realistic, fs = torchaudio.load(
            os.path.join(cfg.rir_dataset_path, rir_path)
        )
        rir_realistic = rir_realistic.squeeze()  # squeeze the channel dimension

        # compute the rap metrics
        STI_calculator = RapidSpeechTransmissionIndex()
        STI_est_seg[n] = STI_calculator(rir_synthesized, fs)
        STI_gt_seg[n] = STI_calculator(rir_realistic, fs)

        ALcons_calculator = PercentageArticulationLoss()
        ALcons_est_seg[n] = ALcons_calculator(STI_est_seg[n])
        ALcons_gt_seg[n] = ALcons_calculator(STI_gt_seg[n])

        TR_est_seg[n] = Td_hat
        TR_gt_seg[n] = rir_segment.iloc[n]["Tt"]

        EDT_calculator = EarlyDecayTime()
        EDT_est_seg[n] = EDT_calculator(rir_synthesized, fs)
        EDT_gt_seg[n] = EDT_calculator(rir_realistic, fs)

        C80_calculator = Clarity(clarity_mode="C80")
        C80_est_seg[n] = C80_calculator(rir_synthesized, fs)
        C80_gt_seg[n] = C80_calculator(rir_realistic, fs)

        C50_calculator = Clarity(clarity_mode="C50")
        C50_est_seg[n] = C50_calculator(rir_synthesized, fs)
        C50_gt_seg[n] = C50_calculator(rir_realistic, fs)

        D50_calculator = Definition()
        D50_est_seg[n] = D50_calculator(rir_synthesized, fs)
        D50_gt_seg[n] = D50_calculator(rir_realistic, fs)

        Ts_calculator = CenterTime()
        Ts_est_seg[n] = Ts_calculator(rir_synthesized, fs)
        Ts_gt_seg[n] = Ts_calculator(rir_realistic, fs)

    rap = {"STI_est": STI_est_seg, "STI_gt": STI_gt_seg}
    rap["ALcons_est"], rap["ALcons_gt"] = ALcons_est_seg, ALcons_gt_seg
    rap["TR_est"], rap["TR_gt"] = TR_est_seg, TR_gt_seg
    rap["EDT_est"], rap["EDT_gt"] = EDT_est_seg, EDT_gt_seg
    rap["C80_est"], rap["C80_gt"] = C80_est_seg, C80_gt_seg
    rap["C50_est"], rap["C50_gt"] = C50_est_seg, C50_gt_seg
    rap["D50_est"], rap["D50_gt"] = D50_est_seg, D50_gt_seg
    rap["Ts_est"], rap["Ts_gt"] = Ts_est_seg, Ts_gt_seg

    return rap


def rap_eval(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    assert cfg.TiTd_pred_path
    assert cfg.volume_pred_path
    assert cfg.rir_dataset_path

    rir_segments = rir_manifest_segment(cfg)

    number_of_segments = len(rir_segments)

    log.info(f"Loading Ti and Td prediction from <{cfg.TiTd_pred_path}>")
    TiTd_pred = torch.load(cfg.TiTd_pred_path)

    log.info(f"Loading volume prediction from <{cfg.volume_pred_path}>")
    volume_pred = torch.load(cfg.volume_pred_path)

    rap_pairs = []
    # segment_size = cfg.batch_size

    if cfg.multithreaded:
        with ThreadPoolExecutor(max_workers=12) as executor:
            rap_prediction = []
            for i, rir_segment in enumerate(rir_segments):
                TiTd_pred_segment = TiTd_pred[i]
                volume_pred_segment = volume_pred[i]
                rap_prediction.append(
                    executor.submit(
                        rap_eval_segment,
                        cfg,
                        rir_segment,
                        TiTd_pred_segment,
                        volume_pred_segment,
                    )
                )
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=60),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            SpinnerColumn(),
            MofNCompleteColumn(),
        ) as progress:
            task = progress.add_task("[green]Processing...", total=len(rap_prediction))
            for rap_pred in rap_prediction:
                progress.update(task, advance=1)
                rap_paired = rap_pred.result()
                rap_pairs.append(rap_paired)
    else:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=60),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            SpinnerColumn(),
            MofNCompleteColumn(),
        ) as progress:
            task = progress.add_task("[green]Processing...", total=number_of_segments)
            for i, rir_segment in enumerate(rir_segments):
                # progress.console.print(f"Processing segment {i+1}/{number_of_segments}")
                progress.update(task, advance=1)
                TiTd_pred_segment = TiTd_pred[i]
                volume_pred_segment = volume_pred[i]
                rap_paired = rap_eval_segment(
                    cfg, rir_segment, TiTd_pred_segment, volume_pred_segment
                )
                rap_pairs.append(rap_paired)

    log.info("RAP evaluation progress completed!")
    log.info(f"Saving RAP evaluation to <{cfg.rap_eval_path}>")

    object_dict = {
        "cfg": cfg,
    }

    return rap_pairs, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval_rap.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    rap_eval_results, _ = rap_eval(cfg)

    # aggregate the results
    rap_eval_results = agg_pair(rap_eval_results)

    # unpack the results
    STI_hat, STI = rap_eval_results["STI_pair"]
    ALcons_hat, ALcons = rap_eval_results["ALcons_pair"]
    TR_hat, TR = rap_eval_results["TR_pair"]
    EDT_hat, EDT = rap_eval_results["EDT_pair"]
    C80_hat, C80 = rap_eval_results["C80_pair"]
    C50_hat, C50 = rap_eval_results["C50_pair"]
    D50_hat, D50 = rap_eval_results["D50_pair"]
    Ts_hat, Ts = rap_eval_results["Ts_pair"]

    # calculate the evaluation metrics
    STI_mae = mae_calculator(STI_hat, STI)
    ALcons_mae = mae_calculator(ALcons_hat, ALcons)
    TR_mae = mae_calculator(TR_hat, TR)
    EDT_mae = mae_calculator(EDT_hat, EDT)
    C80_mae = mae_calculator(C80_hat, C80)
    C50_mae = mae_calculator(C50_hat, C50)
    D50_mae = mae_calculator(D50_hat, D50)
    Ts_mae = mae_calculator(Ts_hat, Ts)

    STI_corrcoef = corrcoef_calculator(STI_hat, STI)
    ALcons_corrcoef = corrcoef_calculator(ALcons_hat, ALcons)
    TR_corrcoef = corrcoef_calculator(TR_hat, TR)
    EDT_corrcoef = corrcoef_calculator(EDT_hat, EDT)
    C80_corrcoef = corrcoef_calculator(C80_hat, C80)
    C50_corrcoef = corrcoef_calculator(C50_hat, C50)
    D50_corrcoef = corrcoef_calculator(D50_hat, D50)
    Ts_corrcoef = corrcoef_calculator(Ts_hat, Ts)

    eval_metric_collect = dict(
        {
            "STI": [STI_mae, STI_corrcoef],
            "ALcons": [ALcons_mae, ALcons_corrcoef],
            "TR": [TR_mae, TR_corrcoef],
            "EDT": [EDT_mae, EDT_corrcoef],
            "C80": [C80_mae, C80_corrcoef],
            "C50": [C50_mae, C50_corrcoef],
            "D50": [D50_mae, D50_corrcoef],
            "Ts": [Ts_mae, Ts_corrcoef],
        }
    )

    eval_metric_collect = pd.DataFrame(eval_metric_collect, index=["MAE", "Corrcoef"])

    # save evaluation results
    torch.save(rap_eval_results, cfg.rap_eval_path)
    eval_metric_collect.to_csv(cfg.rap_eval_metric_path)

    log.info(f"RAP evaluation results saved to <{cfg.rap_eval_path}>")
    log.info(f"RAP evaluation metrics saved to <{cfg.rap_eval_metric_path}>")


if __name__ == "__main__":
    main()
