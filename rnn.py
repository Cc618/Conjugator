import torch as T
import torch.nn.functional as F
from torch import nn, optim


def lstm_init_hidden(n_layer, n_hidden, n_batch, device):
        mk_hidden = lambda: T.zeros(
                [n_layer, n_batch, n_hidden],
                device=device)

        return mk_hidden(), mk_hidden()


class LstmEncoder(nn.Module):
    def __init__(self, n_embed, n_latent, n_hidden, n_layer):
        super().__init__()

        self.n_hidden = n_hidden
        self.n_embed = n_embed
        self.n_layer = n_layer

        self.rnn = nn.LSTM(n_embed, n_hidden, n_layer)
        self.fc = nn.Linear(n_hidden, n_latent)

    def init_hidden(self, n_batch, device):
        return lstm_init_hidden(self.n_layer, self.n_hidden, n_batch, device)

    def forward(self, x, h):
        '''
        Returns y, h (y of shape [batch, embed])
        '''
        y, h = self.rnn(x, h)
        y = self.fc(y)

        return y[-1], h


class LstmDecoder(nn.Module):
    def __init__(self, n_latent, n_token, n_hidden, n_layer):
        super().__init__()

        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_layer = n_layer

        self.rnn = nn.LSTM(n_latent, n_hidden, n_layer)
        self.fc = nn.Linear(n_hidden, n_token)

    def init_hidden(self, n_batch, device):
        return lstm_init_hidden(self.n_layer, self.n_hidden, n_batch, device)

    def forward(self, x, h):
        '''
        Returns y, h (y of shape [seq, batch, embed])
        '''
        y, h = self.rnn(x, h)
        y = self.fc(y)

        return y, h


class LstmNet(nn.Module):
    def __init__(self, n_token, n_embed, n_latent, n_hidden, n_layer):
        super().__init__()

        self.n_hidden = n_hidden
        self.n_embed = n_embed
        self.n_layer = n_layer

        self.embed = nn.Embedding(n_token, n_embed)
        self.encode = LstmEncoder(n_embed, n_latent, n_hidden, n_layer)
        self.decode = LstmDecoder(n_latent, n_token, n_hidden, n_layer)


def create_lstm(algo):
    return LstmNet(
            len(algo.conf.voc),
            32,
            512,
            256,
            2
        )
