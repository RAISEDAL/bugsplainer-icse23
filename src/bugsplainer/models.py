import torch
import torch.nn as nn
import numpy as np
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)
import logging

logger = logging.getLogger(__name__)


def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))


def build_or_load_gen_model(args):
    config = T5Config.from_pretrained(args.config_name)
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, config=config)

    logger.info("Finish loading model [%s] from %s", get_model_size(model), args.model_name_or_path)

    # if args.load_model_path is not None:
    #     logger.info("Reload model from {}".format(args.load_model_path))
    #     model.load_state_dict(torch.load(args.load_model_path))

    return config, model, tokenizer
