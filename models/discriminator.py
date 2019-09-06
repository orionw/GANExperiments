import torch
import torch.autograd as autograd
import torch.nn as nn
import pdb
import pickle
from models.discriminative_transformers import PretrainedDiscriminativeTransformer

class Discriminator(nn.Module):

    def __init__(self, args):
        super(Discriminator, self).__init__()
        # self.hidden_dim = hidden_dim
        # self.embedding_dim = embedding_dim
        # self.max_seq_length = max_seq_length
        # self.gpu = gpu

        # self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=dropout)
        # self.gru2hidden = nn.Linear(2*2*hidden_dim, hidden_dim)
        # self.dropout_linear = nn.Dropout(p=dropout)
        # self.hidden2out = nn.Linear(hidden_dim, 1)

        self.max_seq_length = args.max_seq_length
        self.gpu = args.no_cuda is False

        # getting a model should also return a tokenizer for that model
        self.model = PretrainedDiscriminativeTransformer(args)
        self.tokenizer = self.model.tokenizer

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(2*2*1, batch_size, self.hidden_dim))

        if self.gpu:
            return h.cuda()
        else:
            return h

    def forward(self, **kwargs):
        return self.model(**kwargs)


    def batchBCELoss(self, inp, target):
        """
        Returns Binary Cross Entropy Loss for discriminator.

         Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size (binary 1/0)
        """

        loss_fn = nn.BCELoss()
        h = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h)
        return loss_fn(out, target)

