import math
import torch as T
from torch import nn, optim
import torch.nn.functional as F
import torch.distributions as D
from algo import Config, Algo, RNN, Transformer, Trainer


def sample_char(logits, temp=1):
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


class TransformerNet(nn.Module):
    def __init__(self, n_token, n_embed, n_hidden, n_layer, max_seq_len):
        super().__init__()

        n_head = 4

        self.embed_vec = nn.Embedding(n_token, n_embed)
        self.embed_pos = PosEmbedding(max_seq_len, n_embed)
        self.embed = lambda x: self.embed_pos(self.embed_vec(x))
        self.transformer = nn.Transformer(n_embed, n_head, n_layer, n_layer,
                n_hidden)
        self.fc = nn.Linear(n_embed, n_token)

    def forward(self, src, tgt):
        '''
        - src : Initial sentence
        - tgt : Beginning of the generated sentence
        - Returns logits
        '''
        # print(src.shape, tgt.shape)
        src = self.embed(src)
        tgt = self.embed(tgt)
        # print(src.shape, tgt.shape)

        # TODO : Bake mask
        mask = self.transformer.generate_square_subsequent_mask(tgt.size(0)) \
                .to(src.device)
        y = self.transformer(src, tgt, tgt_mask=mask)
        # print(y.shape)

        # Shape is now [seq, batch, embed]
        # We want [seq, batch, probs]
        y = self.fc(y)
        # print(y.shape)

        return y


class PosEmbedding(nn.Module):
    def __init__(self, max_len, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = T.zeros(max_len, d_model)
        position = T.arange(0, max_len, dtype=T.float).unsqueeze(1)
        div_term = T.exp(T.arange(0, d_model, 2).float() *
                (-math.log(10000.0) / d_model))
        pe[:, 0::2] = T.sin(position * div_term)
        pe[:, 1::2] = T.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # return self.dropout(self.pe[:x.size(0), :]).squeeze(0)
        # print(x.shape, self.pe.shape)
        # TODO : Works with batches ?
        x = x + self.pe[:x.size(0), :]

        return self.dropout(x)


def create_transformer(algo):
    return TransformerNet(
            len(algo.conf.voc),
            64,
            512,
            3,
            64
        )


def create_adam(net, algo):
    return optim.Adam(net.parameters())


# TODO : Train / test set
if __name__ == '__main__':
    conf = Config()
    conf.kind = 'transformer'
    conf.epochs = 300

    net = Transformer(create_transformer)
    trainer = Trainer(create_adam)

    algo = Algo(conf, net, trainer)

    algo.train()

    # TODO : mv in algo
    keys = '^jouer$', '^aller$', '^coder$', '^rougir$'
    starts = '^je ', '^il ', '^elles ', '^nous '
    for key, start in zip(keys, starts):
        max_seq_len = 64
        temp = 1

        out_key = key
        out = start
        key = T.LongTensor([conf.voc.index(c) for c in key]).unsqueeze(1) \
                .to(conf.device)
        start = T.LongTensor([conf.voc.index(c) for c in start]).unsqueeze(1) \
                .to(conf.device)

        for _ in range(max_seq_len):
            logits = algo.model.net(key, start)[-1].squeeze()

            token = sample_char(logits, temp=temp)
            pred = conf.voc[token.item()]
            if pred == conf.tok_end:
                break

            start = T.cat([start, token.view(1, 1)])
            out += pred

        print(f'\nInput : "{out_key}"')
        print(f'Output : "{out}"')
