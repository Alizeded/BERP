# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import L1Loss
from torch.nn.modules.loss import _Loss


def label_smoothed_nll_loss(lprob, target, epsilon, reduce=True):
    if target.dim() == lprob.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprob.gather(dim=-1, index=target).squeeze(-1)
    smooth_loss = -lprob.sum(dim=-1, keepdim=True).squeeze(-1)
    if reduce:
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()
    eps_i = epsilon / (lprob.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


class LabelSmoothedCrossEntropyCriterion(_Loss):
    def __init__(
        self,
        label_smoothing,
    ):
        super().__init__()
        self.eps = label_smoothing
        self.l1_dist = L1Loss()

    def forward(
        self,
        lprob: torch.Tensor,
        target: torch.Tensor,
        padding_mask: torch.Tensor = None,
        target_padding_mask: torch.Tensor = None,
        reduce=True,
    ):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        # if padding_mask is None:
        padding_mask = (
            (
                torch.BoolTensor(lprob.shape[0], lprob.shape[1])
                .fill_(False)
                .to(lprob.device)
            )
            if padding_mask is None
            else padding_mask
        )

        target_padding_mask = (
            (torch.BoolTensor(target.shape).fill_(False).to(target.device))
            if target_padding_mask is None
            else target_padding_mask
        )

        reverse_padding_mask = ~padding_mask
        reverse_target_padding_mask = ~target_padding_mask

        # padding_mask_length = reverse_padding_mask.size(1)
        # target_padding_mask_length = reverse_target_padding_mask.size(1)

        seq_len = min(reverse_padding_mask.size(1), reverse_target_padding_mask.size(1))

        reverse_padding_mask_ = reverse_padding_mask[:, :seq_len]
        reverse_target_padding_mask_ = reverse_target_padding_mask[:, :seq_len]
        final_reverse_padding_mask = (
            reverse_padding_mask_ & reverse_target_padding_mask_
        )

        lprob = lprob[:, :seq_len, :]
        target = target[:, :seq_len]

        lprob = torch.stack(
            [
                lprob[..., i].masked_select(final_reverse_padding_mask)
                for i in range(lprob.size(-1))
            ],
            dim=-1,
        )
        target = target.masked_select(final_reverse_padding_mask)

        loss, nll_loss = label_smoothed_nll_loss(
            lprob,
            target,
            self.eps,
            reduce=reduce,
        )

        pred = lprob.argmax(dim=1)

        accu = f1_score(target.cpu(), pred.cpu(), average="weighted")

        f1 = accuracy_score(target.cpu(), pred.cpu())

        accu = round(accu * 100, 2)

        f1 = round(f1 * 100, 2)

        l1 = self.l1_dist(pred.float(), target.float())

        return loss, l1, accu, f1
