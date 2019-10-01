import torch
import torch.nn as nn
import random

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder, device, tokenizer=None):
        super().__init__()

        self.encoder = encoder
        # self.shrink = shrink_net
        self.decoder = decoder
        self.device = device
        self.number_of_batches_seen = 0
        # TODO: make this dynamic
        self.model_type = "xlnet"
        self.tokenizer = tokenizer

    # only purpose is to train encoder and decoder; doesn't need one without a target
    def forward(self, batch, trg, teacher_forcing_ratio=0.5):
        batch = tuple(t.to(self.device) for t in batch)
        trg = trg.to(self.device)

        batch_size = trg.shape[0]
        if trg is None:
            max_len = 100
        else:
            max_len = trg.shape[1] # how many words to output

        outputs = torch.zeros(max_len, batch_size, self.decoder.vocab_size).to(self.device)


        # convert for transformer encoding
        if (type(batch) == list or type(batch) == tuple) and len(batch) == 4:
            inputs =  { 'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2] if self.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                        'labels':         batch[3]}
        else:
            # only passed in the word tokens
            inputs =  { 'input_ids':      batch[0].unsqueeze(0)}

        hidden = self.encoder(**inputs)

        # TODO: move this into the encoding step
        #  https://github.com/huggingface/pytorch-pretrained-BERT#usage
        hidden = hidden[0]  # ignore pooled output
        hidden = torch.mean(hidden, dim=1)  # get sentence embedding from mean of word embeddings
        hidden = hidden.unsqueeze(dim=0) #
        # first input to the decoder is the <sos> tokens
        curr_token = trg[:, 0]

        for t in range(1, max_len):
            curr_token = self.decoder.embed(curr_token)
            curr_token = curr_token.unsqueeze(dim=0)
            new_output, hidden = self.decoder(curr_token, hidden)
            outputs[t] = new_output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = new_output.max(1)[1]
            curr_token = (trg[:, t] if teacher_force else top1)

        self.number_of_batches_seen += 1
        return outputs # (seq_len, batch_size, logits)

    def convert_to_text(self, output_logits, given_ids=False):
        if self.tokenizer is None:
            raise Exception("Do not have a tokenizer to convert with")
        if not given_ids:
            _, tokens = torch.max(output_logits, dim=0) # is our best guess
        else:
            tokens = output_logits # not really logits, given id tokens
        predicted = self.tokenizer.convert_ids_to_tokens(tokens.tolist())
        return " ".join([item for item in predicted])