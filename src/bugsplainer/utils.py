from argparse import Namespace
from multiprocessing import Pool

from torch.utils.data import TensorDataset
import numpy as np
import logging
import os
import random
import torch
import time
from tqdm import tqdm
from transformers import BertTokenizer

from ._utils import *

logger = logging.getLogger(__name__)


def load_and_cache_gen_data(
        args: Namespace,
        filename: str,
        pool: Pool,
        tokenizer: BertTokenizer,
        split_tag: str,
        only_src=False,
        is_sample=False,
):
    # cache the data into args.cache_path except it is sampled
    # only_src: control whether to return only source ids for bleu evaluating (dev/test)
    # return: examples (Example object), data (TensorDataset)
    data_tag = '_all' if args.data_num == -1 else '_%d' % args.data_num
    cache_fn = '{}/{}.pt'.format(args.cache_path, split_tag + ('_src' if only_src else '') + data_tag)

    examples = read_examples(filename, args.data_num, args.task, args.sub_task)

    if is_sample:
        examples = random.sample(examples, min(5000, len(examples)))
    if split_tag == 'train':
        calc_stats(examples, tokenizer, is_tokenize=True)
    else:
        calc_stats(examples)
    if os.path.exists(cache_fn) and not is_sample:
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info("Sample 5k data for computing bleu from %s", filename)
        else:
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args, split_tag) for idx, example in enumerate(examples)]
        features = pool.map(
            convert_examples_to_features,
            tqdm(tuple_examples, total=len(tuple_examples), desc='Converting examples to features')
        )
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        if split_tag == 'test' or only_src:
            data = TensorDataset(all_source_ids)
        else:
            all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
            data = TensorDataset(all_source_ids, all_target_ids)
        if not is_sample:
            torch.save(data, cache_fn)
    return examples, data


def get_filenames(data_root, task, sub_task, split=''):
    if task in ['explain', 'finetune']:
        valid_sub_tasks = ['patch', 'sbt-random', 'sbt-time', 'sbt-project']
        assert sub_task in valid_sub_tasks, f'Invalid sub_task {sub_task} to explain. Options are {valid_sub_tasks}'
        if sub_task == 'patch':
            train_fn = '{}/train-sbt-random-{}.csv'.format(data_root, task)
            dev_fn = '{}/valid-sbt-random-{}.csv'.format(data_root, task)
            test_fn = '{}/test-sbt-random-{}.csv'.format(data_root, task)
        else:
            train_fn = '{}/train-{}-{}.csv'.format(data_root, sub_task, task)
            dev_fn = '{}/valid-{}-{}.csv'.format(data_root, sub_task, task)
            test_fn = '{}/test-{}-{}.csv'.format(data_root, sub_task, task)
    else:
        raise KeyError(f'`{task}` is not a valid task name')
    if split == 'train':
        return train_fn
    elif split == 'dev':
        return dev_fn
    elif split == 'test':
        return test_fn
    else:
        return train_fn, dev_fn, test_fn


def read_examples(filename, data_num, task, sub_task):
    # if sub_task is 'patch', the input_col is 'patch'
    # if sub_task is 'sbt-random', the input col is 'sbt'
    read_explain_examples = create_read_explain_examples(input_col=sub_task.split('-')[0])
    return read_explain_examples(filename, data_num)


def calc_stats(examples, tokenizer=None, is_tokenize=False):
    percentile_positions = [50, 75, 80, 85, 90, 95]
    src_len = list(map(lambda ex: len(ex.source.split()), examples))
    trg_len = list(map(lambda ex: len(ex.target.split()), examples))

    logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                len(examples), np.mean(src_len), np.mean(trg_len), max(src_len), max(trg_len))
    logger.info("percentiles at %s: src %s, trg %s",
                ', '.join(map(str, percentile_positions)),
                ', '.join(map(str, np.percentile(src_len, percentile_positions))),
                ', '.join(map(str, np.percentile(trg_len, percentile_positions))))
    if is_tokenize:
        src_len_tokenize = list(map(len, tokenizer(list(map(lambda ex: ex.source, examples)))['input_ids']))
        trg_len_tokenize = list(map(len, tokenizer(list(map(lambda ex: ex.target, examples)))['input_ids']))
        logger.info("[TOKENIZE] avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    np.mean(src_len_tokenize), np.mean(trg_len_tokenize), max(src_len_tokenize),
                    max(trg_len_tokenize))
        logger.info("[TOKENIZE] percentiles at %s: src %s, trg %s",
                    ', '.join(map(str, percentile_positions)),
                    ', '.join(map(str, np.percentile(src_len_tokenize, percentile_positions))),
                    ', '.join(map(str, np.percentile(trg_len_tokenize, percentile_positions))))


def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)
