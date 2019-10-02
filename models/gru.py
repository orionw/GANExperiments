import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, last_word, last_hidden):
        # TODO: just repeating hidden state from encoder if extra layers?
        if last_hidden.shape[0] == 1 and self.n_layers != 1:
            last_hidden = last_hidden.repeat(self.n_layers, 1, 1)
        output, new_hidden = self.rnn(last_word.float(), last_hidden.float())
        prediction = self.out(output.squeeze(0))
        return prediction, new_hidden