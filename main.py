from algo import Config, Algo
from torch import nn, optim


class Net(nn.Module):
    def __init__(self):
        super().__init__()



if __name__ == '__main__':
    net = Net()

    conf = Config()
    conf.kind = 'rnn'
    algo = Algo(conf, net)

    algo.train()
