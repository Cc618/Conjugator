from algo import Config, Algo, RNN
from torch import nn, optim


class LstmNet(nn.Module):
    def __init__(self, n_tokens, n_embed, n_hidden, n_layers):
        super().__init__()

        self.n_hidden = n_hidden
        self.n_embed = n_embed
        self.n_layers = n_layers

        self.embed = nn.Embedding(n_tokens, n_embed)
        self.rnn = nn.LSTM(n_embed, n_hidden, n_layers)
        self.fc = nn.Linear(n_hidden, n_tokens)

    def init_hidden(self, batch, device):
        mk_hidden = lambda: T.zeros(
                [self.n_layers * self.n_hidden, batch, self.n_embed],
                device=device)

        return mk_hidden(), mk_hidden()

    def forward(self, x, h):
        y = self.embed(x)
        y, h = self.rnn(y, h)
        y = self.fc(y)

        return y, h


def create_lstm(algo):
    return LstmNet(
            len(algo.conf.voc),
            32,
            512,
            2
        )


def create_adam(net):
    return optim.Adam(net.parameters())


if __name__ == '__main__':
    net = RNN(create_lstm, create_adam)

    conf = Config()
    conf.kind = 'rnn'
    conf.epochs = 1
    algo = Algo(conf, net)

    algo.train()
