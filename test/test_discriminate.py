import unittest
import json
import os
import logging
import torch
import pandas as pd
import torch.nn as nn
from argparse import Namespace

from transformers import BertTokenizer, BertModel, BertForMaskedLM, XLMTokenizer, XLNetModel, XLNetTokenizer, XLNetLMHeadModel

from models.xlnet import XLNetForSequenceClassificationGivenEmbedding, XLNetEmbedder, XLNetModelWithoutEmbedding
from models.generative_transformers import PretrainedTransformerGenerator

logging.basicConfig(level=logging.INFO)

class TestEmbeddings(unittest.TestCase):
    def setUp(self):
        self.root_path = os.path.join("data", "emnlp_news", "train.csv")
        self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        self.input = torch.tensor(self.tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        self.embed_model = XLNetEmbedder.from_pretrained("xlnet-base-cased")
        self.transformer_input = {"input_ids": self.input}

    def test_embedder_discriminate_works(self):
        # create inital embedding
        embedding = self.embed_model(self.input)
        last_embedding = embedding[0]
        hidden = torch.mean(last_embedding, dim=1)  # get sentence embedding from mean of word embeddings
        hidden = hidden.unsqueeze(dim=0)

        # discriminate
        discriminator = XLNetForSequenceClassificationGivenEmbedding.from_pretrained("xlnet-base-cased")
        output = discriminator(hidden)
        assert output[0].shape == (1, 1), "did not disriminate one example: expected (1, 1) got {}".format(output[0].shape)

    def test_full_model_discriminate_works(self):
        # create inital embedding, one step
        args = create_args()
        full_model = PretrainedTransformerGenerator(args)
        embedding = full_model(**self.transformer_input)

        # discriminate
        discriminator = XLNetForSequenceClassificationGivenEmbedding.from_pretrained("xlnet-base-cased")
        output = discriminator(embedding)
        assert output[0].shape == (1, 1), "did not disriminate one example: expected (1, 1) got {}".format(output[0].shape)


def create_args():
    args = {
            "train_data_file": "./data/puns/val.csv",
            "data_dir": "data/emnlp_news",
            "eval_data_file": "./data/puns/val.csv",
            "gen_model_type": "xlnet",
            "gen_model_name_or_path": "xlnet-base-cased",
            "dis_model_type": "xlnet",
            "dis_model_name_or_path": "xlnet-base-cased",
            "do_train": True,
            "data_dir": "./data/puns",
            "max_seq_length": 16, 
            "per_gpu_train_batch_size": 16,
            "learning_rate": 5e-4,
            "num_train_epochs": 3.0,
            "do_lower_case": True,
            "overwrite_output_dir": True,
            "gradient_accumulation_steps": 12,
            "output_dir": "./gan_results",
            "autoencoder_epochs": 0,
            "no_cuda": True,
            "local_rank": -1,
            "config_name": None,
            "block_size": 16,
            "device": "cpu:0",
            "record_run": False,
            "tokenizer_name": None,
    }
    namespace_args = Namespace(**args)
    return namespace_args