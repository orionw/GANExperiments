import torch
import torch.nn as nn
import random
from models.training_functions import create_transformer_mapping
from torch.autograd import Variable
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder, device, tokenizer=None, model_type="xlnet"):
        """
        Tokenizer is for the generator
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.number_of_batches_seen = 0
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.init_pad_tokens = None

    def forward(self, batch, trg, teacher_forcing_ratio=0.0):
        """
        The autoencoding function
        param: batch: a tensor containing (1, batch_size, word_len)
        param: trg: the target: identical to the batch
        param: teacher_forcing_ratio: a float containing the percentage of the time to use teacher forcing
        returns: a tensor containing the autoencoder value of shape (seq_len, batch_size, logits)
        """
        # prepare input
        batch = tuple(t.to(self.device) for t in batch)
        trg = trg.to(self.device)

        batch_size = trg.shape[0]
        if trg is None:
            max_len = 100
        else:
            max_len = trg.shape[1] # how many words to output

        # convert for transformer encoding
        if (type(batch) == list or type(batch) == tuple) and len(batch) == 4:
            inputs = create_transformer_mapping(batch, self.model_type)
        else:
            # only passed in the word tokens
            inputs =  {'input_ids': batch[0]}  # after -> (1, batch_size, word_length)

        hidden = self.encoder(**inputs) # size ()
        outputs = self.decoder(hidden, max_len, batch_size, trg, device=self.device, 
                               teacher_forcing_ratio=0.5 - min(0.5, self.number_of_batches_seen / 300))
    
        self.number_of_batches_seen += 1
        return outputs # (seq_len, batch_size, logits)

    def greedy_decode(self, model, memory, src_mask, batch_size, max_len=32, start_symbol=1):
        """
        To be used with a transformer decoder. NOT YET IMPLEMENTED
        """
        ys = torch.ones(1, 4).fill_(start_symbol).to(self.device)
        for i in range(max_len-1):
            out = model.decode(memory, src_mask, Variable(ys).to(self.device).long(), 
                                Variable(self.subsequent_mask(ys.size(1)).to(self.device).long()))
            prob = self.encoder.decode(out.squeeze(0))
            _, next_word = torch.max(prob[0], dim = 1)
            ys = torch.cat((ys, next_word.unsqueeze(0).float().to(self.device)), dim=1)
        return ys

    @staticmethod
    def subsequent_mask(self, size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0