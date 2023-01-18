import argparse
import logging
import multiprocessing
import os
import time

import numpy as np
import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from . import smooth_bleu
from .configs import add_args, set_seed, set_dist
from .models import build_or_load_gen_model
from .utils import get_filenames, get_elapse_time, load_and_cache_gen_data

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, split_tag, criteria):
    logger.info("  ***** Running bleu evaluation on {} data*****".format(split_tag))
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_data)
    if args.data_num == -1:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     num_workers=args.cpu_cont, pin_memory=True)
    else:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    pred_ids = []
    bleu, codebleu = 0.0, 0.0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu for {} set".format(split_tag)):
        source_ids = batch[0].to(args.device)
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            if isinstance(model, (DataParallel, DistributedDataParallel)):
                generate_func = model.module.generate
            else:
                generate_func = model.generate
            preds = generate_func(
                source_ids, attention_mask=source_mask, use_cache=True, num_beams=args.beam_size,
                early_stopping=args.task in ['explain', 'finetune'],
                max_length=args.max_target_length,
            )
            top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)

    pred_nls = [
      tokenizer.decode(pred_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
      for pred_id in pred_ids
    ]

    output_fn = os.path.join(args.res_dir, "test_{}.output".format(criteria))
    gold_fn = os.path.join(args.res_dir, "test_{}.gold".format(criteria))
    src_fn = os.path.join(args.res_dir, "test_{}.src".format(criteria))

    if args.task in ['defect']:
        target_dict = {0: 'false', 1: 'true'}
        golds = [target_dict[ex.target] for ex in eval_examples]
        eval_acc = np.mean([int(p == g) for p, g in zip(pred_nls, golds)])
        result = {'em': eval_acc * 100, 'bleu': 0, 'codebleu': 0}

        with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
            for pred_nl, gold in zip(pred_nls, eval_examples):
                f.write(pred_nl.strip() + '\n')
                f1.write(target_dict[gold.target] + '\n')
                f2.write(gold.source.strip() + '\n')
            logger.info("Save the predictions into %s", output_fn)
    else:
        dev_accs, predictions = [], []
        with open(output_fn, 'w', encoding='utf-8') as f, open(gold_fn, 'w', encoding='utf-8') as f1, open(src_fn, 'w', encoding='utf-8') as f2:
            for pred_nl, gold in zip(pred_nls, eval_examples):
                dev_accs.append(pred_nl.strip() == gold.target.strip())
                if args.task in ['summarize', 'explain', 'pretrain', 'finetune']:
                    # for smooth-bleu4 evaluation
                    predictions.append(str(gold.idx) + '\t' + pred_nl)
                    f.write(str(gold.idx) + '\t' + pred_nl.strip() + '\n')
                    f1.write(str(gold.idx) + '\t' + gold.target.strip() + '\n')
                    f2.write(str(gold.idx) + '\t' + gold.source.strip() + '\n')
                else:
                    f.write(pred_nl.strip() + '\n')
                    f1.write(gold.target.strip() + '\n')
                    f2.write(gold.source.strip() + '\n')

        (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, gold_fn)
        bleu = round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)

        result = {'em': np.mean(dev_accs) * 100, 'bleu': bleu}
        if args.task == 'concode':
            result['codebleu'] = codebleu * 100

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def main():
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logger.info(args)
    t0 = time.time()

    set_dist(args)
    set_seed(args)
    config, model, tokenizer = build_or_load_gen_model(args)
    model.to(args.device)
    if args.n_gpu > 1:
        # for DataParallel
        model = torch.nn.DataParallel(model)
    pool = multiprocessing.Pool(args.cpu_cont)
    args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        eval_examples, eval_data = load_and_cache_gen_data(args, args.test_filename, pool, tokenizer, 'test',
                                                           only_src=True, is_sample=False)
        torch.cuda.empty_cache()
        result = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'test', 'best-bleu')
        test_bleu, test_em = result['bleu'], result['em']
        test_codebleu = result['codebleu'] if 'codebleu' in result else 0
        result_str = "bleu-4: %.2f, em: %.4f, codebleu: %.4f\n" % (test_bleu, test_em, test_codebleu)
        logger.info(result_str)
        fa.write(result_str)
    logger.info(f"Finish and take {get_elapse_time(t0)}")
    fa.write(f"Finish and take {get_elapse_time(t0)}")
    fa.close()


if __name__ == "__main__":
    main()
