from torch.utils.data import Dataset, DataLoader
import typing
import os
import pandas as pd
import torch

def get_dataloaders(root_path: str, header: bool = None, batch_size: int = 48, device: str = "0",
                    shuffle=True, n_workers=1):
    """
    A function to return PyTorch dataloaders for a given csv file(s).
    :param root_path: the path where the dataset lives
    :param header: whether or not to return a header
    """
    dataset = GenerationDataset(root_path, header=header)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=shuffle, num_workers=n_workers)
    return dataloader


class GenerationDataset(Dataset):
    """
    A class to hold natural language text only, no label
    """
    def __init__(self, path: str, header: bool = None):
        self.data = pd.read_csv(path, header=header, sep="\n")
        assert self.data.shape[1] == 1, "has to many columns for natural language only dataset: {}".format(data.shape[1])
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        # grab the line needed
        return self.data.iloc[index, :][0]


class GenerationDatasetList(Dataset):
    """
    A class to hold natural language text only, no label
    """
    def __init__(self, text_list: list):
        self.data = text_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # grab the line needed
        return self.data[index]

class DiscriminatorDatasetFromList(Dataset):
    """
    A class to hold natural language text and labels
    """
    def __init__(self, positive: list, negative: list):
        pos = pd.DataFrame({"text": positive, "label": 1})
        neg = pd.DataFrame({"text": negative, "label": 0})
        self.data = pd.concat([pos, neg], axis=0)
        self.data.sample(frac=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # grab the line needed
        return self.data.iloc[index, :][0], self.data.iloc[index, :][1]

