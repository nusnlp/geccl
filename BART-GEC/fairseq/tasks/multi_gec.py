import os

from fairseq import options, utils
from .translation import TranslationTask, load_langpair_dataset
from . import register_task
from fairseq.data import MultiTargetPairDataset, Dictionary, data_utils
import logging

logger = logging.getLogger(__name__)

@register_task('multi_gec')
class MultiGECTask(TranslationTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        parser.add_argument('--num-sources', default=2, type=int, metavar='N',
                            help='Number of sources in the combination model')
        parser.add_argument('--add-mask-token', action='store_true', default=False,
                            help='add mask token to dictionary')

    def __init__(self, args, src_dict, tgt_dict):
        self.sep_idx = src_dict.eos_index  # add_symbol('<sep>')
        self.num_sources = args.num_sources

        super().__init__(args, src_dict, tgt_dict)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).
        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        if getattr(args, 'raw_text', False):
            utils.deprecation_warning('--raw-text is deprecated, please use --dataset-impl=raw')
            args.dataset_impl = 'raw'
        elif getattr(args, 'lazy_load', False):
            utils.deprecation_warning('--lazy-load is deprecated, please use --dataset-impl=lazy')
            args.dataset_impl = 'lazy'

        paths = args.data.split(':')
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info('[{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        logger.info('[{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename
        Args:
            filename (str): the filename
        """
        logger.info('Loaded the dictionary')
        return Dictionary.load(filename)
        # dictionary = Dictionary.load(filename)
        # if args.add_mask_token or 'bart' in getattr(args, 'arch', {}):
        #     dictionary.add_symbol('<mask>')
        # return dictionary

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        lp = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
        )
        self.datasets[split] = MultiTargetPairDataset(
            self.args.num_sources, self.sep_idx,
            lp.src, lp.src_sizes, self.src_dict,
            lp.tgt, lp.tgt_sizes, self.tgt_dict,
            left_pad_source=lp.left_pad_source,
            left_pad_target=lp.left_pad_target,
            max_source_positions=lp.max_source_positions,
            max_target_positions=lp.max_target_positions,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return MultiTargetPairDataset(self.num_sources, self.sep_idx, src_tokens, src_lengths, self.source_dictionary)