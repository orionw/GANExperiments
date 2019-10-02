import torch
import torch.nn as nn


class GeneratorBase(nn.Module):

    def __init__(self, args):
        super(GeneratorBase, self).__init__()
        self.max_seq_length = args.max_seq_length
        self.gpu = args.no_cuda is False
        self.model = None
        self.args = args

    def forward(self, **kwargs):
        """
        Embeds input and applies GRU one token at a time (seq_len = 1)
        """
        return self.model(**kwargs)

    def sample(self, num_samples):
        """
        Samples the network and returns num_samples samples of length args.max_seq_length.

        """
        return self.model.sample(num_samples)

    def sample_text(self, num_samples: int):
        """
        Samples the network and returns num_samples text samples of length args.max_seq_length.

        """
        return self.model.sample_text(num_samples)


class DiscriminatorBase(nn.Module):

    def __init__(self, args):
        super(DiscriminatorBase, self).__init__()
        self.max_seq_length = args.max_seq_length
        self.gpu = args.no_cuda is False
        self.model = None
        self.args = args

    def forward(self, **kwargs):
        """
        Decides whether the sample is real or fake, returns logits
        """
        return self.model(**kwargs)