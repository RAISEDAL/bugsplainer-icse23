import os.path
import random
from argparse import ArgumentParser

import torch
import logging
import multiprocessing
import numpy as np

logger = logging.getLogger(__name__)


def add_args(parser: ArgumentParser):
    # Required parameters
    parser.add_argument("--model_name_or_path", required=True, type=str,
                        help="Path to pre-trained model")
    parser.add_argument("--config_name", default="", type=str, required=True,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="roberta-base", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # General parameters
    parser.add_argument("--task", type=str, required=True, choices=['explain', 'finetune'])
    parser.add_argument("--sub_task", type=str, required=True, choices=[
      'patch', 'sbt-random', 'sbt-project',
    ])
    parser.add_argument("--lang", type=str, default='py')
    parser.add_argument("--desc", type=str, required=True)
    parser.add_argument("--model_type", default="codet5")
    parser.add_argument("--add_lang_ids", action='store_true', default=True)
    parser.add_argument("--data_num", default=-1, type=int)
    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--res_dir", type=str, required=True)
    parser.add_argument("--add_task_prefix", action='store_true', default=True)
    parser.add_argument("--do_eval_bleu", action='store_true', default=True)
    parser.add_argument("--eval_batch_size", type=int, required=True)

    parser.add_argument("--max_source_length", default=512, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=64, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")

    parser.add_argument('--seed', type=int, default=1234,
                        help="random seed for initialization")

    # Programmatically set parameters
    parser.add_argument("--train_filename", default=None, type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")

    args = parser.parse_args()

    args.desc = os.path.join('runs', args.desc)
    args.output_dir = os.path.join(args.desc, args.output_dir)
    args.res_dir = os.path.join(args.desc, args.res_dir)
    for dir_path in [args.desc, args.output_dir, args.res_dir]:
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
    data_dir = args.data_dir.split("/")[-1]
    cache_dir = (f'{data_dir}-{args.task}-{args.sub_task}-{args.lang}-{args.max_source_length}'
                 f'-{args.max_target_length}-{args.eval_batch_size}')
    args.cache_path = os.path.join(args.cache_path, cache_dir)
    if not os.path.exists(args.cache_path):
        os.mkdir(args.cache_path)

    return args


def set_dist(args):
    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    cpu_cont = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()

    logger.warning("device: %s, n_gpu: %s, cpu count: %d", device, args.n_gpu, cpu_cont)
    args.device = device
    args.cpu_cont = min(cpu_cont, 8)


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
