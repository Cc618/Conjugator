import math
import torch as T
from torch import nn, optim
import torch.nn.functional as F


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
        src = self.embed(src)
        tgt = self.embed(tgt)

        # TODO : Bake mask
        mask = self.transformer.generate_square_subsequent_mask(tgt.size(0)) \
                .to(src.device)
        y = self.transformer(src, tgt, tgt_mask=mask)

        y = self.fc(y)

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
        return self.dropout(x + self.pe[:x.size(0), :])


def create_transformer(algo):
    return TransformerNet(
            len(algo.conf.voc),
            256,
            512,
            2,
            64
        )
