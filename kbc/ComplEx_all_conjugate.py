from kbc.KBCModel import KBCModel
from typing import Tuple
import torch
from torch import nn


'''
ComplEx model:
relation: a + bi,

ComplEx_all_conjugate model:
relation: split dimensions into two parts: a + bi, a - bi 
'''


class ComplEx_all_conjugate(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            # set a scale
            init_size: float = 1e-3
    ):
        super(ComplEx_all_conjugate, self).__init__()
        self.sizes = sizes
        self.rank = rank

        # assign two embedding parameters matrices form torch (at each batch)
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2*rank, sparse=True)
            for s in sizes[:2]
        ])
        # the 1st embedding matrix is for head/tail entities (multiply a small number to scale the value )
        self.embeddings[0].weight.data *= init_size
        # the 2nd embedding matrix is for relations (multiply a small number to scale the value )
        self.embeddings[1].weight.data *= init_size

    def score(self, x):
        return transformation(embeddings=self.embeddings, x=x, flag='score', rank=self.rank)

    def forward(self, x):
        return transformation(embeddings=self.embeddings, x=x, flag='forward', rank=self.rank)

    # get tails embeddings
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
            chunk_begin:chunk_begin+chunk_size
        ].transpose(0, 1)

    # get queries embeddings
    def get_queries(self, queries: torch.Tensor):
        return transformation(embeddings=self.embeddings, x=queries, flag="get_queries", rank=self.rank)


def transformation(embeddings, x, flag, rank):
    '''
    flag: ["score", "forward", "get_queries"]
    rank: dimensions number of embedding
    '''
    # x[:, 0]: head information
    # assign embedding parameters to head entity
    lhs = embeddings[0](x[:, 0])
    # x[:, 1]: relation information
    # assign embedding parameters to relation
    rel = embeddings[1](x[:, 1])
    # x[:. 2]: tail information
    # assign embedding parameters to tail entity
    rhs = embeddings[0](x[:, 2])

    rank_split = int(rank/2)
    # the real and imaginary part of head entity
    # lhs = lhs[:, :rank], lhs[:, rank:]
    lhs = lhs[:, :rank_split], lhs[:, rank_split:rank], lhs[:, rank:3*rank_split], lhs[:, 3*rank_split:]
    # the real and imaginary part of relation
    # rel = rel[:, :rank], rel[:, rank:]
    rel = rel[:, :rank_split], rel[:, rank_split:rank], rel[:, rank:3*rank_split], rel[:, 3*rank_split:]
    # the real and imaginary part of tail entity
    # rhs = rhs[:, :rank], rhs[:, rank:]
    rhs = rhs[:, :rank_split], rhs[:, rank_split:rank], rhs[:, rank:3*rank_split], rhs[:, 3*rank_split:]

    # head * transformation = tail
    # score function: scalar product <tail', tail>
    if flag == 'score':
        return torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True
        )
    elif flag == 'forward':
        # get the head/tail parameters/weight value
        to_score = embeddings[0].weight
        to_score = to_score[:, :rank], to_score[:, rank:]
        return (
            (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1)
        ), (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
            )
    elif flag == 'get_queries':
        return torch.cat([
            lhs[0] * rel[0] - lhs[1] * rel[1],
            lhs[0] * rel[1] + lhs[1] * rel[0]
        ], 1)
    else:
        raise ValueError('unsupported flag: {}'.format(flag))