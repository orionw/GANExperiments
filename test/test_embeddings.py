import unittest
import json
import os
import logging
import torch
import pandas as pd
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertForMaskedLM, XLMTokenizer, XLNetModel, XLNetTokenizer, XLNetLMHeadModel

from test.test_discriminate import create_args
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

    def test_embedding_matches_model(self):
        # try original model
        model = XLNetModel.from_pretrained('xlnet-base-cased')
        outputs = model(self.input)
        last_hidden_states = outputs[0]

        # try our version
        embed_outs = self.embed_model(self.input)
        last_embedding = embed_outs[0]
        assert torch.all(torch.eq(last_embedding, last_hidden_states)), "embeddings were not the same"
    
    def test_embedding_lm(self):
        # try original model
        lmmodel = XLNetLMHeadModel.from_pretrained('xlnet-base-cased')
        lm_outputs = lmmodel(self.input)
        last_hidden_states_lm = lm_outputs[0]  # The last hidden-state is the first element of the output tuple

        # try our version
        embed_outs_lm = self.embed_model.lm(self.input)
        last_embedding_lm = embed_outs_lm[0]
        assert torch.all(torch.eq(last_embedding_lm, last_hidden_states_lm)), "LM embeddings were not the same"

    def test_full_embedding(self):
        args = create_args()
        full_model = PretrainedTransformerGenerator(args, self.tokenizer)
        embedding = full_model(**self.transformer_input)

        # try original model
        model = XLNetModel.from_pretrained('xlnet-base-cased')
        outputs = model(self.input)
        last_hidden_states = outputs[0]
        hidden = torch.mean(last_hidden_states, dim=1)  # get sentence embedding from mean of word embeddings
        hidden_embedding = hidden.unsqueeze(dim=0)
        assert torch.all(torch.eq(embedding, hidden_embedding)), "full model embeddings were not the same"

    