from torch.utils.data import Dataset, DataLoader
import typing
import os
import pandas as pd
import torch
import logging
import pickle

from utils.utils_glue import (compute_metrics, convert_examples_to_features,
                        output_modes, processors)

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
        TensorDataset)

logger = logging.getLogger(__name__)


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


class DiscriminatorDatasetFromFile(Dataset):
    """
    A class to hold natural language text only, no label
    """
    def __init__(self, path: str, header: bool = None, label = "0"):
        self.data = pd.read_csv(path, header=header, sep="\n")
        assert self.data.shape[1] == 1, "has to many columns for natural language only dataset: {}".format(data.shape[1])
        self.label = label
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # grab the line needed
        return self.data.iloc[index, 0], self.label


class DiscriminatorDatasetFromList(Dataset):
    """
    A class to hold natural language text only, no label, made from a list object
    """
    def __init__(self, text_list: list, label = "0"):
        self.data = text_list
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # grab the line needed
        return self.data[index], self.label


class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path='train', block_size=512):
        assert os.path.isfile(file_path), "could not load the filepath: {}".format(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, f'cached_lm_{block_size}_{filename}')

        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s/%s", directory, file_path)

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()
            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            while len(tokenized_text) >= block_size:  # Truncate in block of block_size
                self.examples.append(tokenizer.add_special_tokens_single_sentence(tokenized_text[:block_size]))
                tokenized_text = tokenized_text[block_size:]
            # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should look for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


def load_and_cache_examples_generator(args, tokenizer, evaluate=False):
    dataset = TextDataset(tokenizer, file_path=args.eval_data_file if evaluate else args.train_data_file, block_size=32)
    return dataset

def load_and_cache_examples(args, task, tokenizer, given_data, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.dis_model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.dis_model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1] 
        examples = processor.get_dev_examples(given_data) if evaluate else processor.get_train_examples(given_data)
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=bool(args.dis_model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.dis_model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.dis_model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.dis_model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.dis_model_type in ['xlnet'] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    if output_mode != "gan":
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

    return dataset

