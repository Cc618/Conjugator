import torch as T
from torch import nn, optim
import torch.nn.functional as F
import torch.distributions as D
from algo import Config, Algo, RNN, Transformer, Trainer
from transformer import TransformerNet, create_transformer
from rnn import LstmNet, create_lstm


def sample_char(logits, temp, algo):
    '''
    Samples from the categorical distribution
    - logits : Logits of the probabilities (probs before softmax)
    - temp : Temperature, if 1, standard probs like softmax(logits)
    '''
    assert len(logits.shape) == 1, (
            'Logits must have a shape like [n_classes], ' +
            f'shape is {logits.shape}'
        )

    return D.Categorical(logits=logits / temp).sample()


def create_adam(net, algo):
    return optim.Adam(net.parameters())


if __name__ == '__main__':
    conf = Config()
    conf.epochs = 10

    # Use TransformerNet(create_transformer) for the transformer model
    net = RNN(create_lstm)
    trainer = Trainer(create_adam)

    algo = Algo(conf, net, trainer)

    algo.train()

    T.save(algo.model.net.state_dict(), 'model/last')

    keys = [
            '^jouer$', '^aller$', '^coder$', '^rougir$', '^vollir$', '^mager$',
            '^déglutir$', '^praxiter$', '^poulier$', '^patriarcher$',
            '^anticonstituer$'
        ]
    starts = [
            '^je ', '^il ', '^elles ', '^nous ', '^t', '^',
            '^ils ', '^vous ', '^je', '^tu',
            'j\''
        ]
    for key, start in zip(keys, starts):
        out_key = key
        out = start

        # Convert key / start
        key = T.LongTensor([conf.voc.index(c) for c in key]).unsqueeze(1) \
                .to(conf.device)
        start = T.LongTensor([conf.voc.index(c) for c in start]).unsqueeze(1) \
                .to(conf.device)

        # Predict
        out += algo.pred(key, start, sample_char)

        print(f'\nInput : "{out_key}"')
        print(f'Output : "{out}"')
