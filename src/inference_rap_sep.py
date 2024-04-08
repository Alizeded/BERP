from typing import Any, Dict, Tuple

import hydra
import pandas as pd
import rootutils
import torch
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

from src.utils.unpack import agg_rap_pred  # noqa: E402

from src import utils  # noqa: E402

log = utils.get_pylogger(__name__)


def rap_predict_segment(
    cfg: DictConfig,
    TiTd_pred_segment: Dict[str, Any],
    volume_pred_segment: Dict[str, Any],
    fs: int = 16000,  # sample rate, default to 16kHz
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    segment_size = cfg.batch_size

    # pre-alllocate memory for rap_pred
    STI_est_seg = torch.zeros(segment_size)
    ALcons_est_seg = torch.zeros(segment_size)
    TR_est_seg = torch.zeros(segment_size)
    EDT_est_seg = torch.zeros(segment_size)
    C80_est_seg = torch.zeros(segment_size)
    C50_est_seg = torch.zeros(segment_size)
    D50_est_seg = torch.zeros(segment_size)
    Ts_est_seg = torch.zeros(segment_size)

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

        # compute the rap metrics
        STI_calculator = RapidSpeechTransmissionIndex()
        STI_est_seg[n] = STI_calculator(rir_synthesized, fs)

        ALcons_calculator = PercentageArticulationLoss()
        ALcons_est_seg[n] = ALcons_calculator(STI_est_seg[n])

        TR_est_seg[n] = Td_hat

        EDT_calculator = EarlyDecayTime()
        EDT_est_seg[n] = EDT_calculator(rir_synthesized, fs)

        C80_calculator = Clarity(clarity_mode="C80")
        C80_est_seg[n] = C80_calculator(rir_synthesized, fs)

        C50_calculator = Clarity(clarity_mode="C50")
        C50_est_seg[n] = C50_calculator(rir_synthesized, fs)

        D50_calculator = Definition()
        D50_est_seg[n] = D50_calculator(rir_synthesized, fs)

        Ts_calculator = CenterTime()
        Ts_est_seg[n] = Ts_calculator(rir_synthesized, fs)

    # round to 4 decimal places
    STI_est_seg = list(map(lambda x: round(x.item(), 4), STI_est_seg))
    ALcons_est_seg = list(map(lambda x: round(x.item(), 4), ALcons_est_seg))
    TR_est_seg = list(map(lambda x: round(x.item(), 4), TR_est_seg))
    EDT_est_seg = list(map(lambda x: round(x.item(), 4), EDT_est_seg))
    C80_est_seg = list(map(lambda x: round(x.item(), 4), C80_est_seg))
    C50_est_seg = list(map(lambda x: round(x.item(), 4), C50_est_seg))
    D50_est_seg = list(map(lambda x: round(x.item(), 4), D50_est_seg))
    Ts_est_seg = list(map(lambda x: round(x.item(), 4), Ts_est_seg))

    rap = {"STI_est": STI_est_seg}
    rap["ALcons_est"] = ALcons_est_seg
    rap["TR_est"] = TR_est_seg
    rap["EDT_est"] = EDT_est_seg
    rap["C80_est"] = C80_est_seg
    rap["C50_est"] = C50_est_seg
    rap["D50_est"] = D50_est_seg
    rap["Ts_est"] = Ts_est_seg

    return rap


def rap_predict(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    assert cfg.TiTd_pred_path
    assert cfg.volume_pred_path

    log.info(f"Loading Ti and Td prediction from <{cfg.TiTd_pred_path}>")
    TiTd_pred = torch.load(cfg.TiTd_pred_path)

    log.info(f"Loading volume prediction from <{cfg.volume_pred_path}>")
    volume_pred = torch.load(cfg.volume_pred_path)

    number_of_segments = len(TiTd_pred)

    # segment_size = cfg.batch_size

    if cfg.multithreaded:
        with ThreadPoolExecutor(max_workers=12) as executor:
            rap_prediction = []
            for i in range(number_of_segments):
                TiTd_pred_segment = TiTd_pred[i]
                volume_pred_segment = volume_pred[i]
                rap_prediction.append(
                    executor.submit(
                        rap_predict_segment,
                        cfg,
                        TiTd_pred_segment,
                        volume_pred_segment,
                        fs=cfg.fs,
                    )
                )
        rap_predictions = []
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=60),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            SpinnerColumn(),
            MofNCompleteColumn(),
        ) as progress:
            task = progress.add_task("[green]Processing...", total=len(rap_prediction))
            for i, rap_pred in enumerate(rap_prediction):
                progress.update(task, advance=1)
                rap_predicted = rap_pred.result()
                rap_predictions.append(rap_predicted)
    else:
        rap_predictions = []
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=60),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            SpinnerColumn(),
            MofNCompleteColumn(),
        ) as progress:
            task = progress.add_task("[green]Processing...", total=number_of_segments)
            for i in range(number_of_segments):
                # progress.console.print(f"Processing segment {i+1}/{number_of_segments}")
                progress.update(task, advance=1)
                TiTd_pred_segment = TiTd_pred[i]
                volume_pred_segment = volume_pred[i]
                rap_predicted = rap_predict_segment(
                    cfg, TiTd_pred_segment, volume_pred_segment, fs=cfg.fs
                )
                rap_predictions.append(rap_predicted)

    log.info("RAP predictions progress completed!")

    object_dict = {
        "cfg": cfg,
    }

    return rap_predictions, object_dict


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="inference_rap.yaml"
)
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    rap_prediction, _ = rap_predict(cfg)

    # aggregate rap predictions
    rap_prediction = agg_rap_pred(rap_prediction, batch_size=cfg.batch_size)

    # convert to dataframe
    rap_prediction_df = pd.DataFrame(rap_prediction)

    # save predictions
    rap_prediction_df.to_csv(cfg.rap_pred_path)
    log.info(f"RAP predictions saved to <{cfg.rap_pred_path}>")


if __name__ == "__main__":
    main()
