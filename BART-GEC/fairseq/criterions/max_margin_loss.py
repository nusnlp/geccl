# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion

def distance(x1, x2):
    # print("The summation of value: {}".format(x2.sum()))
    return x2.sum() - x1.sum()

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        if pad_mask.any():
            nll_loss.masked_fill_(pad_mask, 0.)
            smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def max_margin_loss(pos_ori_lprobs, pos_ori_target, neg_ori_lprobs, neg_ori_target, sample_size, margin, ignore_index=None):
    '''
    ori_lprobs: the log likelihood of P(si|ti) with shape [N, M, V]
    ori_target: the log likelihood of P(si|ti) with shape [N, M]
    ori_lprobs_other：the log likelihood of P(si'|si) with shape [N, M, V]
    ori_target_other：the log likelihood of P(si'|si) with shape [N, M]
    nll_matrix: p(ti|si)
    nll_other_matrix: p(si'|si)
    '''
    a, b = pos_ori_target.shape
    target = pos_ori_target.view(a, b, 1)
    nll_matrix = torch.gather(-pos_ori_lprobs, 2, target)
    a, b = neg_ori_target.shape
    target_other = neg_ori_target.view(a, b, 1)
    nll_other_matrix = torch.gather(-neg_ori_lprobs, 2, target_other)

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        pad_mask_other = target_other.eq(ignore_index)
        if pad_mask.any():
            nll_matrix.masked_fill_(pad_mask, 0.)
            nll_other_matrix.masked_fill_(pad_mask_other, 0.)

    a, b, _ = nll_matrix.shape
    new_nll_matrix = nll_matrix.view(a, b)
    a, b, _ = nll_other_matrix.shape
    new_nll_other_matrix = nll_other_matrix.view(a, b)
    triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=distance, reduction='sum', margin=margin)
    zero_anchor = torch.zeros(a, b)
    final_mml = 0
    assert nll_other_matrix.shape[0] == nll_matrix.shape[0], "nll_other_matrix.shape[0]: {} and nll_matrix.shape[0]: {}".format(nll_other_matrix.shape[0], nll_matrix.shape[0])
    for i in range(a):
        local_mml = triplet_loss(-zero_anchor[i], -new_nll_other_matrix[i], -new_nll_matrix[i])
        final_mml += local_mml
    return final_mml


def mml_label_smoothed_nll_loss(pos_lprobs, pos_target, pos_ori_lprobs, pos_ori_target, neg_ori_lprobs, neg_ori_target,
                                sample_size, margin, trade_off, epsilon, ignore_index=None, reduce=True):
    if pos_target.dim() == pos_lprobs.dim() - 1:
        pos_target = pos_target.unsqueeze(-1)

    nll_loss = -pos_lprobs.gather(dim=-1, index=pos_target)
    smooth_loss = -pos_lprobs.sum(dim=-1, keepdim=True)

    if ignore_index is not None:
        pad_mask = pos_target.eq(ignore_index)

        if pad_mask.any():
            nll_loss.masked_fill_(pad_mask, 0.)
            smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / pos_lprobs.size(-1)
    mml = max_margin_loss(pos_ori_lprobs, pos_ori_target, neg_ori_lprobs, neg_ori_target, sample_size, margin, ignore_index)
    ce_loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    loss = (1 - trade_off) * ce_loss + trade_off * mml
    # print("MML loss before upsampling: {}".format(mml))
    # print("Final loss: {}".format(loss))
    return loss, nll_loss


@register_criterion('max_margin_loss')
class Max_Margin_Criterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.margin = args.margin
        self.trade_off = args.trade_off

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--margin', default=1.0, type=float, metavar='D',
                           help='margin value for the Max Margin Function')
        parser.add_argument('--trade_off', default=0.5, type=float, metavar='D',
                            help='margin value for the Max Margin Function')
        # fmt: on

    def forward(self, model, sample, valid=False, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if 'valid' in sample.keys():
            net_output = model(**sample['net_input'])
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
            sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        else:
            import copy
            pos_sample = copy.deepcopy(sample)
            tmp_sample = copy.deepcopy(sample)
            tmp_sample['net_input']['prev_output_tokens'] = sample['net_input']['prev_output_tokens_neg']
            tmp_sample['target'] = sample['neg_target']
            neg_sample = copy.deepcopy(tmp_sample)
            pos_net_output = model(**pos_sample['net_input'])
            neg_net_output = model(**neg_sample['net_input'])
            sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']

            loss, nll_loss = self.mml_compute_loss(model, pos_net_output, neg_net_output, pos_sample, neg_sample,
                                                   sample_size, reduce=reduce)

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

    def mml_compute_loss(self, model, pos_net_output, neg_net_output, pos_sample, neg_sample, sample_size, reduce=True):
        # positive (si, ti) pair's probability and its corresponding target
        pos_ori_lprobs = model.get_normalized_probs(pos_net_output, log_probs=True)
        pos_lprobs = pos_ori_lprobs.view(-1, pos_ori_lprobs.size(-1))
        pos_ori_target = model.get_targets(pos_sample, pos_net_output)
        pos_target = pos_ori_target.view(-1, 1)



        # negative (si, si') pair's probability and its corresponding target
        neg_ori_lprobs = model.get_normalized_probs(neg_net_output, log_probs=True)
        neg_ori_target = model.get_targets(neg_sample, neg_net_output)
        # neg_lprobs = neg_ori_lprobs.view(-1, neg_ori_lprobs.size(-1))
        # neg_target = neg_ori_target.view(-1, 1)

        loss, nll_loss = mml_label_smoothed_nll_loss(pos_lprobs, pos_target, pos_ori_lprobs, pos_ori_target,
                                                     neg_ori_lprobs, neg_ori_target, sample_size, self.margin,
                                                     self.trade_off, self.eps, ignore_index=self.padding_idx,
                                                     reduce=reduce)

        return loss, nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: round(2**meters['nll_loss'].avg, 3))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
