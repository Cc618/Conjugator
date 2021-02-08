from algo import Config, Algo, RNN
from torch import nn, optim


class LstmNet(nn.Module):
    def __init__(self, n_token, n_embed, n_hidden, n_layer):
        super().__init__()

        self.n_hidden = n_hidden
        self.n_embed = n_embed
        self.n_layer = n_layer

        self.embed = nn.Embedding(n_token, n_embed)
        self.rnn = nn.LSTM(n_embed, n_hidden, n_layer)
        self.fc = nn.Linear(n_hidden, n_token)

    def init_hidden(self, n_batch, device):
        mk_hidden = lambda: T.zeros(
                [self.n_layer * self.n_hidden, n_batch, self.n_embed],
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
    conf = Config()
    conf.kind = 'rnn'
    conf.epochs = 1

    net = RNN(create_lstm)
    trainer = Trainer(create_adam)

    algo = Algo(conf, net, trainer)

    algo.train()
