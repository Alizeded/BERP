from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import L1Loss, SmoothL1Loss
from torchmetrics import PearsonCorrCoef

from src.utils.AcousticParameterUtils import (  # noqa: E402
    CenterTime,
    Clarity,
    Definition,
    EarlyDecayTime,
    PercentageArticulationLoss,
    RapidSpeechTransmissionIndex,
    ReverberationTime,
    SparseStochasticIRModel,
)
from src.utils.unitary_linear_norm import unitary_norm_inv

SAMPLE_RATE = 16000


def filter_by_quartiles(data, multiplier=1.5):
    """
    Filter data based on quartiles (IQR method)

    Args:
        data: numpy array of values
        multiplier: IQR multiplier (default 1.5 for standard outlier detection)
                   use 3.0 for extreme outlier detection
    Returns:
        filtered_data: data with outliers removed
        mask: boolean mask indicating valid values
    """
    # Calculate quartiles
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 100)  # 100th percentile is the max value

    # Calculate IQR (Interquartile Range)
    IQR = Q3 - Q1

    # Define bounds
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    # Create mask for valid values
    mask = (data >= lower_bound) & (data <= upper_bound)

    # Filter data
    filtered_data = data[mask]

    return filtered_data, mask


# Example usage in your context:
def calculate_filtered_mean(metrics, print_stats=True):
    """
    Calculate filtered mean of metrics removing outliers

    Args:
        metrics: numpy array of metrics (e.g., sti_hats, alcons_hats)
        print_stats: whether to print statistics
    Returns:
        filtered_mean: mean after removing outliers
    """
    # Filter outliers
    filtered_metrics, mask = filter_by_quartiles(metrics)

    if print_stats:
        print(f"Removed {np.sum(~mask)} outliers")
        print(f"Original mean: {metrics.mean():.4f}")
        print(f"Filtered mean: {filtered_metrics.mean():.4f}")

    return filtered_metrics.mean()


def process_single_rir(
    params: tuple,
) -> tuple[float, float, float, float, float, float, float, float]:
    """
    Process a single RIR with given parameters

    Args:
        params: Tuple containing (Ti_hat, Td_hat, volume_hat, seed)

    Returns:
        Tuple of computed metrics
    """
    Ti_hat, Td_hat, volume_hat, seed = params

    # Create RIR synthesizer
    rir_synthesizer = SparseStochasticIRModel(
        Ti=Ti_hat,
        Td=Td_hat,
        volume=volume_hat,
        mu_Th=0.0399,
        fs=SAMPLE_RATE,
        seed=seed,
    )
    rir_synthesized = rir_synthesizer()

    # Initialize calculators
    STI_calculator = RapidSpeechTransmissionIndex()
    ALcons_calculator = PercentageArticulationLoss()
    EDT_calculator = EarlyDecayTime()
    TR_calculator = ReverberationTime()
    C80_calculator = Clarity(clarity_mode="C80")
    C50_calculator = Clarity(clarity_mode="C50")
    D50_calculator = Definition()
    Ts_calculator = CenterTime()

    # Compute metrics
    sti = STI_calculator(rir_synthesized, SAMPLE_RATE)
    alcons = ALcons_calculator(sti)
    tr = TR_calculator(rir_synthesized, SAMPLE_RATE)
    edt = EDT_calculator(rir_synthesized, SAMPLE_RATE)
    c80 = C80_calculator(rir_synthesized, SAMPLE_RATE)
    c50 = C50_calculator(rir_synthesized, SAMPLE_RATE)
    d50 = D50_calculator(rir_synthesized, SAMPLE_RATE)
    ts = Ts_calculator(rir_synthesized, SAMPLE_RATE)

    return sti, alcons, tr, edt, c80, c50, d50, ts


class JointEstimationEvaluation(nn.Module):

    def __init__(self, iter_times: int = 20, max_workers: int = 8):
        super().__init__()

        self.criterion = SmoothL1Loss()
        self.l1_loss = L1Loss()
        self.pearson_corr_coef = PearsonCorrCoef()
        self.iter_times = iter_times
        self.max_workers = max_workers

    def _get_param_pred_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride, padding):
            return torch.floor(
                (input_length + 2 * padding - (kernel_size - 1) - 1) / stride + 1
            )

        room_parametric_predictor_layers = [
            (384, 3, 1, 1),
            (384, 3, 1, 1),
            (384, 5, 3, 0),
            (384, 3, 1, 1),
        ]

        for i in range(len(room_parametric_predictor_layers)):
            input_lengths = _conv_out_length(
                input_lengths,
                room_parametric_predictor_layers[i][1],
                room_parametric_predictor_layers[i][2],
                room_parametric_predictor_layers[i][3],
            )

        return input_lengths.to(torch.long)

    def forward(
        self,
        param_hat: dict[str, torch.Tensor],
        param_groundtruth: dict[str, torch.Tensor],
    ):

        # extract predicted parameters
        # force cast to float to avoid mismatch in data type
        Th_hat = param_hat["Th_hat"]
        Tt_hat = param_hat["Tt_hat"]
        volume_hat = param_hat["volume_hat"]
        dist_src_hat = param_hat["dist_src_hat"]
        padding_mask = param_hat["padding_mask"]

        # extract groundtruth parameters
        # force cast to float to avoid mismatch in data type
        sti = param_groundtruth["sti"]
        alcons = param_groundtruth["alcons"]
        tr = param_groundtruth["t60"]
        edt = param_groundtruth["edt"]
        c80 = param_groundtruth["c80"]
        c50 = param_groundtruth["c50"]
        d50 = param_groundtruth["d50"]
        ts = param_groundtruth["ts"]
        volume = param_groundtruth["volume"]
        dist_src = param_groundtruth["dist_src"]

        assert Th_hat.shape == Tt_hat.shape == volume_hat.shape == dist_src_hat.shape

        # ----------------- obtain padding mask ----------------- #
        if padding_mask is not None and padding_mask.any():
            # B x T
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self._get_param_pred_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                Tt_hat.shape[:2], dtype=Tt_hat.dtype, device=Tt_hat.device
            )  # B x T

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[
                (
                    torch.arange(padding_mask.shape[0], device=padding_mask.device),
                    output_lengths - 1,
                )
            ] = 1
            padding_mask = (
                1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])
            ).bool()  # padded value is True, else False

            # reverse the padding mask, non-padded value is True, else False
            reverse_padding_mask = padding_mask.logical_not()

        #! ------------------- compute evaluation metrics ------------------- #
        # ------------------------- polynomial loss ------------------------- #
        # collapse all the predicted parameters to batch size 1D tensor
        if padding_mask is not None and padding_mask.any():
            # ---------------------- padding mask handling ---------------------- #s

            Th_hat = (Th_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)

            Tt_hat = (Tt_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)

            volume_hat = (volume_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)

            dist_src_hat = (dist_src_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)

        else:
            # ---------------------- no padding mask handling ---------------------- #
            Th_hat = Th_hat.mean(dim=1)
            Tt_hat = Tt_hat.mean(dim=1)
            volume_hat = volume_hat.mean(dim=1)
            dist_src_hat = dist_src_hat.mean(dim=1)

            # ------------------- inverse unitary normalization -------------------
            # Th_hat = unitary_norm_inv(Th_hat, lb=0.005, ub=0.276)
            # volume_hat = unitary_norm_inv(volume_hat, lb=1.5051, ub=3.9542)
            # dist_src_hat = unitary_norm_inv(dist_src_hat, lb=0.191, ub=28.350)

        # ------------------- compute evaluation metrics -------------------

        # MAE metric for all the predicted parameters
        sti_hat = torch.zeros_like(Th_hat, device=sti.device)
        alcons_hat = torch.zeros_like(Th_hat, device=alcons.device)
        tr_hat = torch.zeros_like(Th_hat, device=tr.device)
        edt_hat = torch.zeros_like(Th_hat, device=edt.device)
        c80_hat = torch.zeros_like(Th_hat, device=c80.device)
        c50_hat = torch.zeros_like(Th_hat, device=c50.device)
        d50_hat = torch.zeros_like(Th_hat, device=d50.device)
        ts_hat = torch.zeros_like(Th_hat, device=ts.device)
        volume_hat = volume_hat.to(volume.device)
        dist_src_hat = dist_src_hat.to(dist_src.device)
        for n in range(len(Th_hat)):
            # Generate seeds
            seeds = torch.randint(3200, 4000, (self.iter_times,))

            # Prepare parameters for parallel processing
            params = [
                (Th_hat[n], Tt_hat[n], 10 ** volume_hat[n], seed.item())
                for seed in seeds
            ]

            # Process RIRs in parallel
            metrics = [torch.zeros(self.iter_times) for _ in range(8)]
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = list(executor.map(process_single_rir, params))
                # Collect results
                for i, (sti_, alcons_, tr_, edt_, c80_, c50_, d50_, ts_) in enumerate(
                    futures
                ):
                    metrics[0][i] = sti_
                    metrics[1][i] = alcons_
                    metrics[2][i] = tr_
                    metrics[3][i] = edt_
                    metrics[4][i] = c80_
                    metrics[5][i] = c50_
                    metrics[6][i] = d50_
                    metrics[7][i] = ts_

                # Average metrics and round
            # remove quantile and nan
            metrics = [m[~torch.isnan(m)] for m in metrics]

            # calculate filtered mean
            filtered_means = [
                calculate_filtered_mean(m.numpy(), print_stats=False) for m in metrics
            ]

            estimates = [m.mean().round(4) for m in filtered_means]

            # convert to tensor
            estimates = [torch.tensor(e).to(sti.device) for e in estimates]

            sti_hat[n] = estimates[0]
            alcons_hat[n] = estimates[1]
            tr_hat[n] = estimates[2]
            edt_hat[n] = estimates[3]
            c80_hat[n] = estimates[4]
            c50_hat[n] = estimates[5]
            d50_hat[n] = estimates[6] / 100  # remove percentage
            ts_hat[n] = estimates[7]

        loss_sti = self.l1_loss(sti_hat, sti)
        loss_alcons = self.l1_loss(alcons_hat, alcons)
        loss_tr = self.l1_loss(tr_hat, tr)
        loss_edt = self.l1_loss(edt_hat, edt)
        loss_c80 = self.l1_loss(c80_hat, c80)
        loss_c50 = self.l1_loss(c50_hat, c50)
        loss_d50 = self.l1_loss(d50_hat, d50)
        loss_ts = self.l1_loss(ts_hat, ts)
        loss_volume = self.l1_loss(volume_hat, volume)
        loss_dist_src = self.l1_loss(dist_src_hat, dist_src)

        # Pearson correlation coefficient for all the predicted parameters
        corr_sti = self.pearson_corr_coef(sti_hat, sti).abs()
        corr_alcons = self.pearson_corr_coef(alcons_hat, alcons).abs()
        corr_tr = self.pearson_corr_coef(tr_hat, tr).abs()
        corr_edt = self.pearson_corr_coef(edt_hat, edt).abs()
        corr_c80 = self.pearson_corr_coef(c80_hat, c80).abs()
        corr_c50 = self.pearson_corr_coef(c50_hat, c50).abs()
        corr_d50 = self.pearson_corr_coef(d50_hat, d50).abs()
        corr_ts = self.pearson_corr_coef(ts_hat, ts).abs()
        corr_volume = self.pearson_corr_coef(volume_hat, volume).abs()
        corr_dist_src = self.pearson_corr_coef(dist_src_hat, dist_src).abs()

        return {
            "loss_sti": loss_sti,
            "loss_alcons": loss_alcons,
            "loss_tr": loss_tr,
            "loss_edt": loss_edt,
            "loss_c80": loss_c80,
            "loss_c50": loss_c50,
            "loss_d50": loss_d50,
            "loss_ts": loss_ts,
            "loss_volume": loss_volume,
            "loss_dist_src": loss_dist_src,
            "corr_sti": corr_sti,
            "corr_alcons": corr_alcons,
            "corr_tr": corr_tr,
            "corr_edt": corr_edt,
            "corr_c80": corr_c80,
            "corr_c50": corr_c50,
            "corr_d50": corr_d50,
            "corr_ts": corr_ts,
            "corr_volume": corr_volume,
            "corr_dist_src": corr_dist_src,
        }
