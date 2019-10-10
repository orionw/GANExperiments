import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class GRUModelGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.gru2out = nn.Linear(hidden_dim, vocab_size)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

    def forward(self, inp, hidden):
        # input dim                                             # batch_size
        emb = self.embeddings(inp)                              # batch_size x embedding_dim
        emb = emb.view(1, -1, self.embedding_dim)               # 1 x batch_size x embedding_dim
        out, hidden = self.gru(emb, hidden)                     # 1 x batch_size x hidden_dim (out)
        out = self.gru2out(out.view(-1, self.hidden_dim))       # batch_size x vocab_size
        out = F.log_softmax(out, dim=1)
        return out, hidden

    def sample(num_samples, start_letter=0):
        """
         Outputs: samples, hidden
            - samples: num_samples x args.max_seq_lengthgth (a sampled sequence in each row)
        """

        samples = torch.zeros(num_samples, self.args.max_seq_length).type(torch.LongTensor)

        h = self.init_hidden(num_samples)
        inp = autograd.Variable(torch.LongTensor([start_letter]*num_samples))

        if self.gpu:
            samples = samples.cuda()
            inp = inp.cuda()

        for i in range(self.args.max_seq_length):
            out, h = self.forward(inp, h)               # out: num_samples x vocab_size
            out = torch.multinomial(torch.exp(out), 1)  # num_samples x 1 (sampling from each row)
            samples[:, i] = out.view(-1).data

            inp = out.view(-1)

        return samples


class GRUDecoder(nn.Module):
    def __init__(self, emb_dim, vocab_size, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim 
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(self.vocab_size, self.emb_dim)
        self.hid_dim = hid_dim
        self.n_layers = n_layers  # should be 1
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.out = nn.Linear(self.hid_dim, self.vocab_size)

    def step(self, last_word, last_hidden):
        """
        A step in the RNN: takes in the last hidden state and word and outputs the prediction and next hidden state
        :param last_word: the last word of the sequence
        :parma last_hidden: the last step's hidden state
        """
        # TODO: just repeating hidden state from encoder if extra layers?
        if last_hidden.shape[0] == 1 and self.n_layers != 1:
            last_hidden = last_hidden.repeat(self.n_layers, 1, 1)
        output, new_hidden = self.rnn(last_word.float(), last_hidden.float())
        prediction = self.out(output.squeeze(0))
        return prediction, new_hidden

    def forward(self, hidden, max_len, batch_size, target=None, device="cuda:0", teacher_forcing_ratio=0.5):
        """
        A function to decode the output from a encoder embedding. Recurrently decodes each token
        param: hidden: the output from the encoder network
        param: max_len: the max length of the sequence
        param: batch_size: the size of the batch being decoded
        param: target: the target sequence, if training
        param: device: which device to store the tensors on
        param: teaching_forcing_ration: the ratio at which to `teacher force` or supply the target value
        """
        if target is None:
            # can't teacher force during evaluation
            teacher_forcing_ratio = 0

        # first input to the decoder is the <PAD> token
        outputs = torch.zeros(max_len, batch_size, self.vocab_size).to(device)

        ### Stuff for Transformer that didn't work #####
        # ys = torch.ones((1, batch_size, 768)).to(self.device)
        # self.greedy_decode(model=self.decoder, memory=hidden, src_mask=inputs["attention_mask"], batch_size=batch_size)
        # self.decoder(hidden, ys, src_mask=inputs["attention_mask"])

        curr_token = torch.zeros((batch_size)).long().to(device)
        for t in range(0, max_len):
            # embed the token id to become a tensor
            curr_token = self.embed(curr_token).unsqueeze(dim=0)
            new_output, hidden = self.step(curr_token, hidden)
            top1 = new_output.max(1)[1]
            teacher_force = random.random() < teacher_forcing_ratio
            if t >= 0:
                outputs[t] = new_output
                curr_token = (target[:, t] if teacher_force else top1)
            else:
                curr_token = top1

        return outputs

    
