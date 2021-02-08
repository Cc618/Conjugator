# Trains / evaluates model

import dataset
import torch as T
import torch.distributions as D
from tqdm import tqdm


class Config:
    def __init__(self):
        '''
        Algorithm configuration
        '''
        # IO
        self.dataset_path = 'dataset.json'
        self.dataset = None
        self.voc = None

        # Net
        self.kind = 'rnn'
        self.epochs = 2
        # TODO : CUDA
        # self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.device = T.device('cpu')


class Algo:
    def __init__(self, conf, net):
        self.conf = conf
        self.net = net.to(conf.device)

        # Load dataset
        if self.conf.dataset is None:
            self.conf.dataset = dataset.filter_data(
                    dataset.load(conf.dataset_path),
                    'Indicatif', 'Présent'
                )
            self.conf.voc = dataset.get_voc(self.conf.dataset)[0]

    def train(self):
        print('Training')
