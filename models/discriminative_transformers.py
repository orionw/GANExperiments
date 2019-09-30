""" Finetuning the library models for sequence classification on (Bert, XLM, XLNet, RoBERTa)."""
from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)

from pytorch_transformers import AdamW, WarmupLinearSchedule

from models.xlnet import XLNetForSequenceClassificationGivenEmbedding
from utils.helpers import set_seed

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, RobertaConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassificationGivenEmbedding, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}

class PretrainedDiscriminativeTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Load pretrained model and tokenizer
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        args.dis_model_type = args.dis_model_type.lower()
        num_labels = 2 # only generator and real samples
        finetuning_task = "cola" # classification
        config_class, self.model_class, self.tokenizer_class = MODEL_CLASSES[args.dis_model_type]
        config = config_class.from_pretrained(args.config_name if args.config_name else args.dis_model_name_or_path, num_labels=num_labels, finetuning_task=finetuning_task)
        self.tokenizer = self.tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.dis_model_name_or_path, do_lower_case=args.do_lower_case)
        self.model = self.model_class.from_pretrained(args.dis_model_name_or_path, from_tf=bool('.ckpt' in args.dis_model_name_or_path), config=config)

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    def forward(self, **kwargs):
        return self.model(**kwargs)



   