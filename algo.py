# Trains / evaluates model

import random
import torch as T
import torch.nn.functional as F
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
        self.dataset_split = .1
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
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')


class Trainer:
    def __init__(self, create_opti, criterion=F.cross_entropy):
        '''
        Training only data
        - create_opti(algo, net) : Returns the optimizer
        '''
        self.create_opti = create_opti
        self.criterion = criterion


# TODO : seed
class Algo:
    def __init__(self, conf, model, trainer):
        self.conf = conf
        self.model = model
        self.trainer = trainer

        # Load dataset
        if self.conf.dataset is None:
            # Load dataset
            self.conf.dataset = dataset.filter_data(
                    dataset.load(conf.dataset_path),
                    'Indicatif', 'Présent'
                )
            self.conf.voc = dataset.get_voc(self.conf.dataset)[0]

            # Add special token
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

            # Split test data
            self.conf.dataset_test, self.conf.dataset = dataset.split(
                    self.conf.dataset, self.conf.dataset_split
                )

        self.model.create(self)
        self.trainer.opti = self.trainer.create_opti(self.model.net, self)

    def iter_data(self, training=True):
        '''
        Iterates through the dataset (returns tensors)
        - training : True if using training set
        - Yields a batch of keys, values pair such that keys contains
            tense names and values one conjugation
        * Shapes are [seq_len, batch]
        * Start, end and pad tokens are added
        '''
        # Vocabulary for both keys and values is the same (this allows same
        # embedding)
        for keys, values in dataset.iter_data(
                self.conf.dataset if training else self.conf.dataset_test,
                voc_keys=self.conf.voc,
                voc_values=self.conf.voc,
                pad_tok_keys=[self.conf.tok_pad_index],
                pad_tok_values=[self.conf.tok_pad_index],
                start_tok_keys=[self.conf.tok_start_index],
                start_tok_values=[self.conf.tok_start_index],
                end_tok_keys=[self.conf.tok_end_index],
                end_tok_values=[self.conf.tok_end_index],
                batch_size=self.conf.batch_size):
            keys = T.stack([T.LongTensor(k).to(self.conf.device)
                    for k in keys])

            # Choose a conjugation at random
            values = T.stack([T.LongTensor(v[random.randint(0, len(v) - 1)])
                .to(self.conf.device) for v in values])

            yield keys.transpose(0, 1), values.transpose(0, 1)

    def tensor2str(self, tensor):
        assert tensor.ndim == 1 and tensor.dtype == T.long, 'Invalid tensor'

        return ''.join(self.conf.voc[c] for c in tensor)

    def train(self):
        print('# Training')
        bar = tqdm(range(self.conf.epochs))
        for _ in bar:
            avg_loss = 0
            n_batch = 0
            for keys, values in self.iter_data():
                keys = keys.to(self.conf.device)
                values = values.to(self.conf.device)

                avg_loss += self.model.train(keys, values, self)
                n_batch += 1

            avg_loss /= n_batch
            tst_loss = self.eval()

            bar.set_postfix({
                    'loss': f'{avg_loss:.2f}',
                    'test_loss': f'{tst_loss:.2f}'
                })

    def eval(self):
        with T.no_grad():
            avg_loss = 0
            n_batch = 0
            for keys, values in self.iter_data():
                keys = keys.to(self.conf.device)
                values = values.to(self.conf.device)

                avg_loss += self.model.eval(keys, values, self)
                n_batch += 1

            return avg_loss / n_batch


class Model:
    '''
    Abstract type describing a type of network (recurrent, transformer...)
    - net : Network containing all the weights we save.
        net.embed must embed a LongTensor into FloatTensor embeddings.
        net.forward must return logits (NOT probabilities).
    - create_net(algo) : Builds and returns the network (see net for details)
    - create_opti(net) : Builds and returns the optimizer
    '''

    # TODO : Update to train
    def predict(self, key, value, algo):
        '''
        Trains the network, see Algo.iter_batch for details about key / value
        - Returns logits of the categorical distribution
        '''

    def create(self, algo):
        self.net = self.create_net(algo).to(algo.conf.device)


# TODO : WIP
class RNN(Model):
    def __init__(self, create_net):
        '''
        Recurrent network (LSTM / GRU)
        - net : Must contains a init_hidden(n_batch, device) function.
            net.encode(x, h) returns z (Encode keys), shape = [batch, latent]
            net.decode(z, h) returns y (Logits, decode latent from net.encode),
                shape = [seq, batch, embed]
        '''
        super().__init__()

        self.create_net = create_net

    def predict(self, key, value, algo):
        # Encode
        key = self.net.embed(key)
        z, _ = self.net.encode(key, self.net.encode.init_hidden(key.size(1),
                algo.conf.device))

        # Decode
        logits = T.empty([value.size(0), key.size(1), key.size(2)])
        decode_hidden = self.net.decode.init_hidden(key.size(1),
                algo.conf.device)

        for i in range(value_size):
            # TODO : Encode not used
            logits[i], _ = self.net.decode(values[i], decode_hidden)

        return logits


class Transformer(Model):
    def __init__(self, create_net):
        '''
        Transformer Encoder Decoder (T-ED) model
        '''
        super().__init__()

        self.create_net = create_net

    def train(self, key, value, algo):
        self.net.train()

        value_src = value[:-1]
        value_tgt = value[1:]

        logits = self.net(key, value_src)

        # TODO : Verify
        logits = logits.view(-1, logits.size(-1))
        value_tgt = value_tgt.reshape(-1)

        # Verify logits not probs
        loss = algo.trainer.criterion(logits, value_tgt)

        algo.trainer.opti.zero_grad()
        loss.backward()

        # TODO : Clip grad
        algo.trainer.opti.step()

        return loss.item()

    def eval(self, key, value, algo):
        self.net.eval()

        value_src = value[:-1]
        value_tgt = value[1:]

        logits = self.net(key, value_src)

        logits = logits.view(-1, logits.size(-1))
        value_tgt = value_tgt.reshape(-1)

        return algo.trainer.criterion(logits, value_tgt)
