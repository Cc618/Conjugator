# Trains / evaluates model

import random
import torch as T
import torch.distributions as D
from tqdm import tqdm
import dataset


class Config:
    def __init__(self):
        '''
        Algorithm configuration
        '''
        # IO
        self.dataset_path = 'dataset.json'
        self.dataset = None
        self.voc = None
        self.tok_start = '^'
        self.tok_end = '$'
        self.tok_pad = '?'
        self.tok_start_index = self.tok_end_index = self.tok_pad_index = None

        # Net
        self.kind = 'rnn'
        self.epochs = 2
        self.batch_size = 16
        # TODO : Useful ?
        self.encoder_seq_len = 32
        self.decoder_seq_len = 64
        # TODO : CUDA
        # self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.device = T.device('cpu')


# TODO : seed
class Algo:
    def __init__(self, conf, model):
        self.conf = conf
        self.model = model

        # Load dataset
        if self.conf.dataset is None:
            self.conf.dataset = dataset.filter_data(
                    dataset.load(conf.dataset_path),
                    'Indicatif', 'Présent'
                )
            self.conf.voc = dataset.get_voc(self.conf.dataset)[0]

            assert self.conf.tok_start not in self.conf.voc, (
                    f'Found start token within the vocabulary ({voc})'
                )
            assert self.conf.tok_end not in self.conf.voc, (
                    f'Found end token within the vocabulary ({voc})'
                )
            assert self.conf.tok_pad not in self.conf.voc, (
                    f'Found pad token within the vocabulary ({voc})'
                )

            self.conf.voc += self.conf.tok_start
            self.conf.voc += self.conf.tok_end
            self.conf.voc += self.conf.tok_pad

            self.conf.tok_start_index = self.conf.voc.index(
                    self.conf.tok_start
                )
            self.conf.tok_end_index = self.conf.voc.index(
                    self.conf.tok_end
                )
            self.conf.tok_pad_index = self.conf.voc.index(
                    self.conf.tok_pad
                )

        model.create(self)

    def iter_data(self):
        '''
        Iterates through the dataset (returns tensors)
        - Yields a batch of keys, values pair such that keys contains
            tense names and values one conjugation
        * Shapes are [seq_len, batch]
        * Start, end and pad tokens are added
        '''
        # Vocabulary for both keys and values is the same (this allows same
        # embedding)
        for keys, values in dataset.iter_data(
                self.conf.dataset,
                voc_keys=self.conf.voc,
                voc_values=self.conf.voc,
                pad_tok_keys=[self.conf.tok_pad_index],
                pad_tok_values=[self.conf.tok_pad_index],
                start_tok_keys=[self.conf.tok_start_index],
                start_tok_values=[self.conf.tok_start_index],
                end_tok_keys=[self.conf.tok_end_index],
                end_tok_values=[self.conf.tok_end_index],
                batch_size=self.conf.batch_size):
            keys = T.stack([T.LongTensor(k, device=self.conf.device)
                    for k in keys])

            # Choose a conjugation at random
            values = T.stack([T.LongTensor(v[random.randint(0, len(v) - 1)],
                    device=self.conf.device) for v in values])

            yield keys.transpose(0, 1), values.transpose(0, 1)

    def tensor2str(self, tensor):
        assert tensor.ndim == 1 and tensor.dtype == T.long, 'Invalid tensor'

        return ''.join(self.conf.voc[c] for c in tensor)

    def train(self):
        print('# Training')
        bar = tqdm(range(self.conf.epochs))
        for _ in bar:
            for b, (keys, values) in enumerate(self.iter_data()):
                print(keys.shape)
                print(values.shape)
                # print(keys)
                # print(values)
                print(self.tensor2str(values[:, 0]))

                loss = self.model.train(keys, values)

                bar.set_postfix({ 'batch': b, 'loss': f'{loss:.2f}' })


class Model:
    '''
    Abstract type describing a type of network (recurrent, transformer...)
    - net : Network containing all the weights we save.
        net.embed must embed a LongTensor into FloatTensor embeddings.
        net.forward must return logits (NOT probabilities).
    - create_net(algo) : Builds and returns the network (see net for details)
    - create_opti(net) : Builds and returns the optimizer
    '''
    def train(self, key, value):
        '''
        Trains the network, see Algo.iter_batch for details about key / value
        - Returns the loss (float)
        '''
        pass

    def create(self, algo):
        net = self.create_net(algo).to(algo.conf.device)


class RNN(Model):
    def __init__(self, create_net, create_opti):
        '''
        Recurrent network (LSTM / GRU)
        - net : Must contains a init_hidden(n_batch, device) function.
            net.forward(x, h) returns y, h_next
        '''
        super().__init__()

        self.create_net = create_net
        self.create_opti = create_opti

    def train(self, key, value):
        return 42
