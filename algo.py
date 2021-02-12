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
        self.epochs = 2
        self.batch_size = 16
        self.max_seq_len = 64
        self.temp = 1
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
        disp_loss = 0
        disp_tst = 0
        for _ in bar:
            avg_loss = 0
            n_batch = 0
            for keys, values in self.iter_data():
                keys = keys.to(self.conf.device)
                values = values.to(self.conf.device)

                avg_loss += self.model.train(keys, values, self)
                n_batch += keys.size(1)

                bar.set_postfix({
                        'loss': f'{disp_loss * 100:.2f} %',
                        'test_loss': f'{disp_tst * 100:.2f} %',
                        'batch': n_batch,
                    })

            avg_loss /= n_batch
            tst_loss = self.eval()

            disp_loss = avg_loss
            disp_tst = tst_loss

    def eval(self):
        with T.no_grad():
            avg_loss = 0
            n_batch = 0
            for keys, values in self.iter_data():
                keys = keys.to(self.conf.device)
                values = values.to(self.conf.device)

                avg_loss += self.model.eval(keys, values, self)
                n_batch += keys.size(1)

            return avg_loss / n_batch

    def pred(self, key, value, sample_char):
        '''
        Predicts what is after value given key
        - sample_char : Functor that takes logits, temp and algo to sample
            a token given the categorical probability distribution (its index)
        - Returns the output string
        '''
        out = ''
        hidden = self.model.pred_init(key, value, self)
        while value.size(0) < self.conf.max_seq_len:
            token = self.model.pred_next(hidden, value, self, sample_char)

            pred = self.conf.voc[token.item()]
            if pred == self.conf.tok_end:
                break

            value = T.cat([value, token.view(1, 1)])
            out += pred

        return out


class Model:
    '''
    Abstract type describing a type of network (recurrent, transformer...)
    - net : Network containing all the weights we save.
        net.embed must embed a LongTensor into FloatTensor embeddings.
        net.forward must return logits (NOT probabilities).
    - create_net(algo) : Builds and returns the network (see net for details)
    - create_opti(net) : Builds and returns the optimizer
    '''

    def train(self, key, value, algo):
        '''
        Trains the network, see Algo.iter_batch for details about key / value
        - Returns the loss value for this batch
        '''

    def eval(self, key, value, algo):
        '''
        Like train but doesn't update weights
        - Returns the loss value for this batch
        '''

    def create(self, algo):
        self.net = self.create_net(algo).to(algo.conf.device)

    def pred_init(self, key, value, algo):
        '''
        Inits a prediction
        - Returns None if no init required, custom data otherwise
        '''

    def pred_next(self, init, value, algo, sample_char):
        '''
        Next prediction
        - init : Data from pred_init
        - sample_char : Functor that takes logits, temp and algo to sample
            a token given the categorical probability distribution (its index)
        - Returns the next token index (shape of [1, 1])
        '''


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

    def train(self, key, value, algo):
        self.net.train()

        # Encode
        key = self.net.embed(key)
        _, z = self.net.encode(
                key,
                self.net.init_hidden(key.size(1), algo.conf.device)
            )

        # Decode
        value_src = self.net.embed(value[:-1])
        logits, _ = self.net.decode(value_src, z)

        logits = logits.view(-1, logits.size(-1))
        value_tgt = value[:-1].reshape(-1)

        loss = algo.trainer.criterion(logits, value_tgt)

        # Optimize
        algo.trainer.opti.zero_grad()
        loss.backward()

        # TODO : Clip grad
        algo.trainer.opti.step()

        return loss.item()

    def eval(self, key, value, algo):
        self.net.eval()

        # Encode
        key = self.net.embed(key)
        _, z = self.net.encode(
                key,
                self.net.init_hidden(key.size(1), algo.conf.device)
            )

        # Decode
        value_src = self.net.embed(value[:-1])
        logits, _ = self.net.decode(value_src, z)

        logits = logits.view(-1, logits.size(-1))
        value_tgt = value[:-1].reshape(-1)

        loss = algo.trainer.criterion(logits, value_tgt)

        return loss.item()

    def pred_init(self, key, value, algo):
        return self.net.encode(self.net.embed(key),
                self.net.init_hidden(key.size(1), algo.conf.device)
            )[1]

    def pred_next(self, hidden, value, algo, sample_char):
        logits, _ = self.net.decode(self.net.embed(value), hidden)
        logits = logits[-1].squeeze()

        token = sample_char(logits, algo.conf.temp, algo)

        return token.view(1, 1)


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

        logits = logits.view(-1, logits.size(-1))
        value_tgt = value_tgt.reshape(-1)

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

    def pred_init(self, key, value, algo):
        return key

    def pred_next(self, key, value, algo, sample_char):
        logits = self.net(key, value)[-1].squeeze()

        token = sample_char(logits, algo.conf.temp, algo)

        return token.view(1, 1)
