import unittest
import json
import os
import logging
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import BertTokenizer, BertModel, BertForMaskedLM, XLMTokenizer, XLNetModel, XLNetTokenizer, XLNetLMHeadModel

from models.xlnet import XLNetEmbedder, XLNetModelWithoutEmbedding
from models.autoencoder import Autoencoder
from models.gru import GRUDecoder
from models.generative_transformers import PretrainedTransformerGenerator
from test.test_discriminate import create_args
from models.training_functions import train_autoencoder


logging.basicConfig(level=logging.INFO)

class TestAutoencoding(unittest.TestCase):
    def setUp(self):
        self.root_path = os.path.join("data", "emnlp_news", "train.csv")
        self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        self.source = "Hello, my dog is cute"
        self.input = torch.tensor(self.tokenizer.encode(self.source)).unsqueeze(0).unsqueeze(0)  # Batch size 1 for shape (1, 1, 7)
        self.args = create_args()
        self.target = self.input.squeeze(0)

    def test_can_decode_basic(self):
        embed_model = PretrainedTransformerGenerator(self.args, self.tokenizer)
        gru_decoder = GRUDecoder(embed_model.config.d_model, self.tokenizer.vocab_size, embed_model.config.d_model, n_layers=1, dropout=0)
        autoencoder = Autoencoder(embed_model, gru_decoder, "cpu:0").to("cpu:0")
        output = autoencoder(self.input, self.target)

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        output = output.permute((1, 2, 0)) # swap for loss
        loss = criterion(output, self.target)
        assert type(loss.item()) == float, "could not get loss value: type {}".format(type(loss.item()))

    def test_can_decode_basic_more_layers(self):
        embed_model = PretrainedTransformerGenerator(self.args, self.tokenizer)   
        gru_decoder = GRUDecoder(embed_model.config.d_model, self.tokenizer.vocab_size, embed_model.config.d_model, n_layers=4, dropout=.2)
        autoencoder = Autoencoder(embed_model, gru_decoder, "cpu:0").to("cpu:0")
        output = autoencoder(self.input, self.target)

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        output = output.permute((1, 2, 0)) # swap for loss
        loss = criterion(output, self.target)
        assert type(loss.item()) == float, "could not get loss value: type {}".format(type(loss.item()))

    def test_decode_to_text(self):
        embed_model = PretrainedTransformerGenerator(self.args, self.tokenizer) 
        gru_decoder = GRUDecoder(embed_model.config.d_model, self.tokenizer.vocab_size, embed_model.config.d_model, n_layers=4, dropout=.2)
        autoencoder = Autoencoder(embed_model, gru_decoder, "cpu:0").to("cpu:0")
        output = autoencoder(self.input, self.target)

        # get text output
        _, best_guess = torch.max(output, dim=2)
        predicted = self.tokenizer.convert_ids_to_tokens(best_guess.permute(1, 0).flatten().tolist())
        string_pred = " ".join(predicted)
        print('Predicted: ', string_pred)
        print('Actual: ', self.tokenizer.convert_ids_to_tokens(self.input.flatten().tolist()))
        assert type(string_pred) == str, "predicted value was not a string, was a {}".format(type(string_pred))

    def test_autoencoder_training(self):
        # create a one sample dataframe for the test
        train_ds = TensorDataset(self.input.squeeze(0), self.input.squeeze(0)) # need two dimensional (1, 7) shape for input
        train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)
        # create models
        embed_model = PretrainedTransformerGenerator(self.args, self.tokenizer) 
        decoder = GRUDecoder(embed_model.config.d_model, self.tokenizer.vocab_size, embed_model.config.d_model, n_layers=1, dropout=0)
        decoder = decoder.to(self.args.device) # warning: device is cpu for CI, slow
        autoencoder = Autoencoder(embed_model, decoder, self.args.device, tokenizer=self.tokenizer).to(self.args.device)
        # create needed params
        autoencoder_optimizer = optim.Adam(autoencoder.parameters(), lr=3e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        loss_df = pd.DataFrame(columns=['batch_num', 'loss'])
        # see if it works
        autoencoder = train_autoencoder(self.args, autoencoder, train_dl, train_dl, autoencoder_optimizer, criterion, 1, loss_df, num_epochs=2)

#    def test_autoencoder_loading(self):
#         # create a one sample dataframe for the test
#         train_ds = TensorDataset(self.input.squeeze(0), self.input.squeeze(0)) # need two dimensional (1, 7) shape for input
#         train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)
#         # create models
#         embed_model = PretrainedTransformerGenerator(self.args, self.tokenizer) 
#         decoder = GRUDecoder(embed_model.config.d_model, self.tokenizer.vocab_size, embed_model.config.d_model, n_layers=1, dropout=0)
#         decoder = decoder.to(self.args.device) # warning: device is cpu for CI, slow
#         autoencoder = Autoencoder(embed_model, decoder, self.args.device, tokenizer=self.tokenizer).to(self.args.device)
#         # create needed params
#         autoencoder_optimizer = optim.Adam(autoencoder.parameters(), lr=3e-4)
#         criterion = nn.CrossEntropyLoss(ignore_index=0)
#         loss_df = pd.DataFrame(columns=['batch_num', 'loss'])
#         # see if it works
#         autoencoder = train_autoencoder(self.args, autoencoder, train_dl, train_dl, autoencoder_optimizer, criterion, 1, loss_df, num_epochs=2)


