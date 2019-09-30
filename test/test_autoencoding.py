import unittest
import json
import os
import logging
import torch
import pandas as pd
import torch.nn as nn
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM, XLMTokenizer, XLNetModel, XLNetTokenizer, XLNetLMHeadModel

from models.xlnet import XLNetEmbedder, XLNetModelWithoutEmbedding
from models.autoencoder import Autoencoder
from models.gru import GRUDecoder

logging.basicConfig(level=logging.INFO)

class TestAutoencoding(unittest.TestCase):
    def setUp(self):
        self.root_path = os.path.join("data", "emnlp_news", "train.csv")
        self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        self.source = "Hello, my dog is cute"
        self.input = torch.tensor(self.tokenizer.encode(self.source)).unsqueeze(0)  # Batch size 1

    def test_can_decode_basic(self):
        embed_model = XLNetEmbedder.from_pretrained("xlnet-base-cased")
        embed_outs = embed_model.encode(self.input)
        last_embedding = embed_outs[0]
    
        gru_decoder = GRUDecoder(768, self.tokenizer.vocab_size, 768, 4,.2)
        loss_df = pd.DataFrame(columns=['batch_num', 'loss'])

        autoencoder = Autoencoder(embed_model, gru_decoder, "cpu:0").to("cpu:0")
        output = autoencoder(self.input, self.input)

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        output = output.view(-1, output.shape[-1])
        trg = self.input.view(-1)
        loss = criterion(output, trg)
        assert type(loss.item()) == float, "could not get loss value: type {}".format(type(loss.item()))

    def test_decode_to_text(self):
        embed_model = XLNetEmbedder.from_pretrained("xlnet-base-cased")
        embed_outs = embed_model.encode(self.input)
        last_embedding = embed_outs
    
        gru_decoder = GRUDecoder(768, self.tokenizer.vocab_size, 768, 4,.2)
        loss_df = pd.DataFrame(columns=['batch_num', 'loss'])

        autoencoder = Autoencoder(embed_model, gru_decoder, "cpu:0").to("cpu:0")
        output = autoencoder(self.input, self.input)

        # get text output
        _, best_guess = torch.max(output, dim=2)
        predicted = self.tokenizer.convert_ids_to_tokens(best_guess.permute(1, 0).flatten().tolist())
        string_pred = " ".join(predicted)
        print('Predicted: ', string_pred)
        print('Actual: ', self.tokenizer.convert_ids_to_tokens(self.input.flatten().tolist()))
        assert type(string_pred) == str, "predicted value was not a string, was a {}".format(type(string_pred))


