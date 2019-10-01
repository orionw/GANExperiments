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


    def test_embedder_discriminate_works(self):
        # create inital embedding
        embed_outs = self.embed_model(self.input)
        last_embedding = embed_outs[0]
        hidden = torch.mean(last_embedding, dim=1)  # get sentence embedding from mean of word embeddings
        hidden = hidden.unsqueeze(dim=0)

        # discriminate
        discriminator = XLNetForSequenceClassificationGivenEmbedding.from_pretrained("xlnet-base-cased")
        output = discriminator(hidden)
        assert output[0].shape == (1, 2), "did not disriminate one example: expected (1, 2) got {}".format(output[0].shape)
