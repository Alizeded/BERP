from typing import Dict, Tuple
import torch


def agg_pair(
    rap_pairs: Tuple[Dict[str, torch.Tensor]],
    batch_size: int = 16,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    raps_size = len(rap_pairs) * batch_size
    STI_hat, STI = torch.zeros(raps_size), torch.zeros(raps_size)
    ALcons_hat, ALcons = torch.zeros(raps_size), torch.zeros(raps_size)
    TR_hat, TR = torch.zeros(raps_size), torch.zeros(raps_size)
    EDT_hat, EDT = torch.zeros(raps_size), torch.zeros(raps_size)
    C80_hat, C80 = torch.zeros(raps_size), torch.zeros(raps_size)
    C50_hat, C50 = torch.zeros(raps_size), torch.zeros(raps_size)
    D50_hat, D50 = torch.zeros(raps_size), torch.zeros(raps_size)
    Ts_hat, Ts = torch.zeros(raps_size), torch.zeros(raps_size)

    j = 0
    for k, batch_rap in enumerate(rap_pairs):
        for i in range(batch_size):
            STI_hat[j] = batch_rap["STI_est"][i]
            STI[j] = batch_rap["STI_gt"][i]
            ALcons_hat[j] = batch_rap["ALcons_est"][i]
            ALcons[j] = batch_rap["ALcons_gt"][i]
            TR_hat[j] = batch_rap["TR_est"][i]
            TR[j] = batch_rap["TR_gt"][i]
            EDT_hat[j] = batch_rap["EDT_est"][i]
            EDT[j] = batch_rap["EDT_gt"][i]
            C80_hat[j] = batch_rap["C80_est"][i]
            C80[j] = batch_rap["C80_gt"][i]
            C50_hat[j] = batch_rap["C50_est"][i]
            C50[j] = batch_rap["C50_gt"][i]
            D50_hat[j] = batch_rap["D50_est"][i]
            D50[j] = batch_rap["D50_gt"][i]
            Ts_hat[j] = batch_rap["Ts_est"][i]
            Ts[j] = batch_rap["Ts_gt"][i]
            j += 1

        output = {
            "STI_pair": (STI_hat, STI),
        }
        output["ALcons_pair"] = (ALcons_hat, ALcons)
        output["TR_pair"] = (TR_hat, TR)
        output["EDT_pair"] = (EDT_hat, EDT)
        output["C80_pair"] = (C80_hat, C80)
        output["C50_pair"] = (C50_hat, C50)
        output["D50_pair"] = (D50_hat, D50)
        output["Ts_pair"] = (Ts_hat, Ts)

    return output


def agg_rap_pred(
    rap_pred: Tuple[Dict[str, torch.Tensor]],
    batch_size: int = 16,
) -> Dict[str, torch.Tensor]:

    rap_size = len(rap_pred) * batch_size
    STI = torch.zeros(rap_size)
    ALcons = torch.zeros(rap_size)
    TR = torch.zeros(rap_size)
    EDT = torch.zeros(rap_size)
    C80 = torch.zeros(rap_size)
    C50 = torch.zeros(rap_size)
    D50 = torch.zeros(rap_size)
    Ts = torch.zeros(rap_size)

    j = 0
    for k, batch_rap in enumerate(rap_pred):
        for i in range(batch_size):
            STI[j] = batch_rap["STI_est"][i]
            ALcons[j] = batch_rap["ALcons_est"][i]
            TR[j] = batch_rap["TR_est"][i]
            EDT[j] = batch_rap["EDT_est"][i]
            C80[j] = batch_rap["C80_est"][i]
            C50[j] = batch_rap["C50_est"][i]
            D50[j] = batch_rap["D50_est"][i]
            Ts[j] = batch_rap["Ts_est"][i]
            j += 1

    output = {
        "STI": STI,
        "ALcons": ALcons,
        "TR": TR,
        "EDT": EDT,
        "C80": C80,
        "C50": C50,
        "D50": D50,
        "Ts": Ts,
    }

    return output


def agg_rpp_pred(
    rpp_pred: Tuple[Dict[str, torch.Tensor]],
    param: str,
    batch_size: int = 16,
) -> Dict[str, torch.Tensor]:

    parmeters_size = len(rpp_pred) * batch_size

    params = torch.zeros(parmeters_size)

    j = 0
    for k, batch_rpp in enumerate(rpp_pred):
        for i in range(batch_size):
            params[j] = batch_rpp[param][i]
            j += 1

    output = {
        param: params,
    }

    return output
