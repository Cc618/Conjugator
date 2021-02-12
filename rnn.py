import torch as T
import torch.nn.functional as F
from torch import nn, optim


# LSTMs
class LstmBlock(nn.Module):
    def __init__(self, n_in, n_out, n_hidden, n_layer):
        '''
        - n_out : Can be None to remove the last fully connected layer
        '''
        super().__init__()

        self.rnn = nn.LSTM(n_in, n_hidden, n_layer)
        self.fc = None if n_out is None else nn.Linear(n_hidden, n_out)

    def forward(self, x, h):
        y, h = self.rnn(x, h)

        if self.fc is not None:
            y = self.fc(y)

        return y, h


class LstmNet(nn.Module):
    def __init__(self, n_token, n_embed, n_hidden, n_layer):
        super().__init__()

        self.n_hidden = n_hidden
        self.n_layer = n_layer

        self.embed = nn.Embedding(n_token, n_embed)
        self.encode = LstmBlock(n_embed, None, n_hidden, n_layer)
        self.decode = LstmBlock(n_embed, n_token, n_hidden, n_layer)

    def init_hidden(self, n_batch, device):
        mk_hidden = lambda: T.zeros(
                [self.n_layer, n_batch, self.n_hidden],
                device=device)

        return mk_hidden(), mk_hidden()


def create_lstm(algo):
    return LstmNet(
            len(algo.conf.voc),
            128,
            256,
            2
        )
