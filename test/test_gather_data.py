import unittest
import json
import os
import logging
import pandas as pd
# from load_data import get_dataloaders
import torch

logging.basicConfig(level=logging.INFO)

# class TestDataLoader(unittest.TestCase):
#     def setUp(self):
#         self.root_path = os.path.join("data", "emnlp_news", "train.csv")

#     def test_dataloader_creation(self):
#         batch_size = 64
#         train_dataloader = get_dataloaders(self.root_path, batch_size=batch_size)
#         assert type(train_dataloader) == torch.utils.data.dataloader.DataLoader
#         assert train_dataloader.batch_size == batch_size
