import torch
import torch.nn as nn
import random
from models.training_functions import create_transformer_mapping
from torch.autograd import Variable
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder, device, tokenizer=None, model_type="xlnet"):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.number_of_batches_seen = 0
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.init_pad_tokens = None

    # only purpose is to train encoder and decoder; doesn't need one without a target
    def forward(self, batch, trg, teacher_forcing_ratio=0.5):
        """
        The autoencoding function
        param: batch: a tensor containing (1, batch_size, word_len)
        param: trg: the target: identical to the batch
        param: teacher_forcing_ratio: a float containing the percentage of the time to use teacher forcing
        returns: a tensor containing the autoencoder value of shape (seq_len, batch_size, logits)
        """
        batch = tuple(t.to(self.device) for t in batch)
        trg = trg.to(self.device)

        batch_size = trg.shape[0]
        self.init_pad_tokens = torch.zeros(trg[:, 1].shape).long().to(self.device) if self.init_pad_tokens is None else self.init_pad_tokens
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

        hidden = self.encoder(**inputs)

        # first input to the decoder is the <PAD> token
        outputs = torch.zeros(max_len, batch_size, self.decoder.vocab_size).to(self.device)
        # ys = torch.ones((1, batch_size, 768)).to(self.device)
        # self.greedy_decode(model=self.decoder, memory=hidden, src_mask=inputs["attention_mask"], batch_size=batch_size)
        # self.decoder(hidden, ys, src_mask=inputs["attention_mask"])
        # don't use the first token generated (<PAD>)
        curr_token = self.init_pad_tokens
        for t in range(0, max_len):
            curr_token = self.decoder.embed(curr_token)
            curr_token = curr_token.unsqueeze(dim=0)
            new_output, hidden = self.decoder(curr_token, hidden)
            top1 = new_output.max(1)[1]
            teacher_force = random.random() < teacher_forcing_ratio
            if t >= 0:
                outputs[t] = new_output
                curr_token = (trg[:, t] if teacher_force else top1)
            else:
                curr_token = top1

        self.number_of_batches_seen += 1
        return outputs # (seq_len, batch_size, logits)

    def convert_to_text(self, output_logits, given_ids=False):
        """
        param: output_logits: either a tensor of tokens or logits of shape (vocab, seq_len)
        """
        if self.tokenizer is None:
            raise Exception("Do not have a tokenizer to convert with")
        if not given_ids:
            _, tokens = torch.max(output_logits, dim=0) # is our best guess
        else:
            tokens = output_logits # not really logits, given id tokens
        predicted = self.tokenizer.convert_ids_to_tokens(tokens.tolist())
        return " ".join([item for item in predicted])

    def greedy_decode(self, model, memory, src_mask, batch_size, max_len=32, start_symbol=1):
        ys = torch.ones(1, 4).fill_(start_symbol).to(self.device)
        for i in range(max_len-1):
            out = model.decode(memory, src_mask, Variable(ys).to(self.device).long(), 
                                Variable(subsequent_mask(ys.size(1)).to(self.device).long()))
            prob = self.encoder.decode(out.squeeze(0))
            _, next_word = torch.max(prob[0], dim = 1)
            ys = torch.cat((ys, next_word.unsqueeze(0).float().to(self.device)), dim=1)
        return ys

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0