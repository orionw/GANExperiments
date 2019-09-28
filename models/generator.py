import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
import torch.nn.init as init
from models.gru import GRUModelGenerator
from models.generative_transformers import PretrainedTransformerGenerator


class Generator(nn.Module):

    def __init__(self, args, tokenizer):
        super(Generator, self).__init__()
        # self.hidden_dim = hidden_dim
        # self.embedding_dim = embedding_dim
        self.max_seq_length = args.max_seq_length
        self.gpu = args.no_cuda is False

        # getting a model should also return a tokenizer for that model
        self.model = PretrainedTransformerGenerator(args, tokenizer)
        self.tokenizer = self.model.tokenizer

    def init_hidden(self, batch_size=1):
        h = autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))

        if self.gpu:
            return h.cuda()
        else:
            return h

    def forward(self, inputs, **kwargs):
        """
        Embeds input and applies GRU one token at a time (seq_len = 1)
        """
        return self.model.forward(inputs, **kwargs)

    def sample(self, num_samples):
        """
        Samples the network and returns num_samples samples of length args.max_seq_length.

        """
        return self.model.sample(num_samples)

    def sample_text(self, num_samples: int):
        return self.model.sample_text(num_samples)

    def batchNLLLoss(self, inp, target):
        """
        Returns the NLL Loss for predicting target sequence.

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len

            inp should be target with <s> (start letter) prepended
        """

        loss_fn = nn.NLLLoss()
        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)           # seq_len x batch_size
        target = target.permute(1, 0)     # seq_len x batch_size
        h = self.init_hidden(batch_size)

        loss = 0
        for i in range(seq_len):
            out, h = self.forward(inp[i], h)
            loss += loss_fn(out, target[i])

        return loss     # per batch

    def batchPGLoss(self, inp, target, reward):
        """
        Returns a pseudo-loss that gives corresponding policy gradients (on calling .backward()).
        Inspired by the example in http://karpathy.github.io/2016/05/31/rl/

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len
            - reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding
                      sentence)

            inp should be target with <s> (start letter) prepended
        """

        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)          # seq_len x batch_size
        target = target.permute(1, 0)    # seq_len x batch_size
        h = self.init_hidden(batch_size)

        loss = 0
        for i in range(seq_len):
            out, h = self.forward(inp[i], h)
            # TODO: should h be detached from graph (.detach())?
            for j in range(batch_size):
                loss += -out[j][target.data[i][j]]*reward[j]     # log(P(y_t|Y_1:Y_{t-1})) * Q

        return loss/batch_size

