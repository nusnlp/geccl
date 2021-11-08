import torch
import numpy as np
from . import data_utils, LanguagePairDataset


def collate(
        samples, num_sources, sep_idx, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
        remove_eos_from_target=False, input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    target = None
    neg_target = None
    n_list = []
    prev_output_tokens = None
    prev_output_tokens_neg = None
    if samples[0].get('target', None) is not None:
        target_rows = [s['target'] for s in samples]
        target_list = [[] for _i in range(num_sources)]
        target_lengths = [[] for _i in range(num_sources)]
        prev_output_tokens_list = [[] for _i in range(num_sources)]
        for row in target_rows:
            if not remove_eos_from_target and row[-1] != sep_idx:
                row = torch.cat([row, torch.Tensor([sep_idx])])
            c = row == sep_idx
            eos_pos = torch.nonzero(c, as_tuple=False).squeeze()
            assert eos_pos.dim() == 1, "eos_pos dim not 1"
            eos_pos = eos_pos.tolist()
            last_eos = -1
            num_segments = 0
            for eos in eos_pos:
                if last_eos >= eos - 1:  # skip empty segment from consecutive eos, if any
                    print("Found empty segment, skipping: ")
                    continue
                if remove_eos_from_target:
                    segment_tgt = row[last_eos + 1:eos]
                else:
                    segment_tgt = row[last_eos + 1:eos + 1]

                target_lengths[num_segments].append(segment_tgt.numel())
                target_list[num_segments].append(segment_tgt)
                last_eos = eos
                num_segments += 1
            # print("Content for target list: {}".format(target_list))
            # print("Length of the target list: {}".format(len(target_list)))
            # print("Value of num_segments: {}".format(num_segments))
            assert num_segments == num_sources, \
                "Source input segments ({}) is not the same as the number of encoders ({})" \
                    .format(num_segments, num_sources)
        new_target_list = [[] for _i in range(num_sources)]
        for i in range(num_sources - 1):
            new_target_list[i] = data_utils.collate_tokens(target_list[i], pad_idx, eos_idx, left_pad_target, False)
        ntokens = sum(target_lengths[0])

        if input_feeding:
            for i in range(num_sources - 1):
                prev_output_tokens_list[i] = data_utils.collate_tokens(target_list[i], pad_idx, eos_idx,
                                                                  left_pad_target, True)
        target = new_target_list[0]
        neg_target = new_target_list[1:num_sources-1]

        encoded_index = target_list[-1]
        n_dict = {6925: 1, 447: 2, 4369: 3, 3714: 4, 4168: 5}
        for i in range(len(encoded_index)):
            n_list.append(n_dict[int(encoded_index[i][0])])

        prev_output_tokens = prev_output_tokens_list[0]
        prev_output_tokens_neg = prev_output_tokens_list[1:num_sources-1]
        # print("Content for target: {}".format(target))
        # print("Length for neg_target: {}".format(len(neg_target)))
        # print("Content for n_list: {}".format(n_list))
        # print("Content for prev_output_tokens: {}".format(prev_output_tokens))
        # print("Length for prev_output_tokens_neg: {}".format(len(prev_output_tokens_neg)))
    else:
        ntokens = sum(len(s['source']) for s in samples)


    # if samples[0].get('target', None) is not None:
    #     target = merge('target', left_pad=left_pad_target)
    #     target = target.index_select(0, sort_order)
    #     ntokens = sum(len(s['target']) for s in samples)
    #
    #     if input_feeding:
    #         # we create a shifted version of targets for feeding the
    #         # previous output token(s) into the next decoder step
    #         prev_output_tokens = merge(
    #             'target',
    #             left_pad=left_pad_target,
    #             move_eos_to_beginning=True,
    #         )
    #         prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    # else:
    #     ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
        'neg_target': neg_target,
        'neg_length': n_list,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
        batch['net_input']['prev_output_tokens_neg'] = prev_output_tokens_neg
    return batch


class MultiTargetPairDataset(LanguagePairDataset):

    def __init__(
            self, num_sources, sep_idx, src, src_sizes, src_dict,
            tgt=None, tgt_sizes=None, tgt_dict=None,
            left_pad_source=True, left_pad_target=False,
            max_source_positions=1024, max_target_positions=1024,
            shuffle=True, input_feeding=True, remove_eos_from_target=False, append_eos_to_target=False,
    ):
        super().__init__(
            src, src_sizes, src_dict,
            tgt, tgt_sizes, tgt_dict,
            left_pad_source, left_pad_target,
            max_source_positions, max_target_positions,
            shuffle, input_feeding, remove_eos_from_target, append_eos_to_target,
        )
        self.remove_eos_from_target = remove_eos_from_target
        self.num_sources = num_sources
        self.sep_idx = sep_idx

    def collater(self, samples):
        """
        Merge a list of samples to form a mini-batch.
        Modified from Language Pair Dataset to return list of src_tokens instead
        """
        return collate(
            samples, num_sources=self.num_sources, sep_idx=self.sep_idx, pad_idx=self.src_dict.pad(),
            eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            remove_eos_from_target=self.remove_eos_from_target, input_feeding=self.input_feeding,
        )
