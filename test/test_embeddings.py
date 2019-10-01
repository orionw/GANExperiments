import unittest
import json
import os
import logging
import torch
import pandas as pd
import torch.nn as nn
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM, XLMTokenizer, XLNetModel, XLNetTokenizer, XLNetLMHeadModel

from models.xlnet import XLNetForSequenceClassificationGivenEmbedding, XLNetEmbedder, XLNetModelWithoutEmbedding

logging.basicConfig(level=logging.INFO)

class TestEmbeddings(unittest.TestCase):
    def setUp(self):
        self.root_path = os.path.join("data", "emnlp_news", "train.csv")
        self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        self.input = torch.tensor(self.tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        self.embed_model = XLNetEmbedder.from_pretrained("xlnet-base-cased")


    def test_embedding_matches_model(self):
        model = XLNetModel.from_pretrained('xlnet-base-cased')
        outputs = model(self.input)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

        embed_outs = self.embed_model(self.input)
        last_embedding = embed_outs[0]
        assert torch.all(torch.eq(last_embedding, last_hidden_states)), "embeddings were not the same"
    
    def test_embedding_lm(self):
        lmmodel = XLNetLMHeadModel.from_pretrained('xlnet-base-cased')
        lm_outputs = lmmodel(self.input)
        last_hidden_states_lm = lm_outputs[0]  # The last hidden-state is the first element of the output tuple

        embed_outs_lm = self.embed_model.lm(self.input)
        last_embedding_lm = embed_outs_lm[0]
        assert torch.all(torch.eq(last_embedding_lm, last_hidden_states_lm)), "LM embeddings were not the same"
    