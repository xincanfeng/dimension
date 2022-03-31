from kbc.KBCModel import KBCModel
from typing import Tuple
import torch
from torch import nn


'''
5*E model:
a, b, c, d in C
'''


class FiveStarE_gamma(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(FiveStarE_gamma, self).__init__()
        self.sizes = sizes
        self.rank = rank

        # assign two embedding parameters matrices from torch (at each batch)
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 8*rank, sparse=True)
            for s in sizes[:2]
        ])
        # the 1st embedding matrix is for head/tail entities
        # self.embeddings[0].weight.data *= init_size
        # the 2nd embedding matrix is for relations
        # self.embeddings[1].weight.data *= init_size

    def score(self, x):
        return transformation(embeddings=self.embeddings, x=x, flag="score", rank=self.rank)

    def forward(self, x):
        return transformation(embeddings=self.embeddings, x=x, flag="forward", rank=self.rank)

    # get tails embeddings
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
            chunk_begin:chunk_begin+chunk_size, :2*self.rank
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
    # x[:, 2]: tail information
    # assign embedding parameters to tail entity
    rhs = embeddings[0](x[:, 2])

    # the real and imaginary part of head
    re_head, im_head, u = lhs[:, :rank], lhs[:, rank:2*rank], lhs[:, 2*rank:3*rank]
    # the real and imaginary part of relation
    re_relation_a, im_relation_a, re_relation_b, im_relation_b, \
        re_relation_c, im_relation_c, re_relation_d, im_relation_d = \
        rel[:, :rank], rel[:, rank:2*rank], rel[:, 2*rank:3*rank], rel[:, 3*rank:4*rank], \
        rel[:, 4*rank:5*rank], rel[:, 5*rank:6*rank], rel[:, 6*rank:7*rank], rel[:, 7*rank:]
    # the real and imaginary part of tail
    re_tail, im_tail = rhs[:, :rank], rhs[:, rank:2*rank]

    # activation: the natural logarithm of the absolute value of the gamma function
    # re_head = torch.lgamma(re_head)
    # im_head = torch.lgamma(im_head)
    re_head = (re_head - re_head.min()) / (re_head.max() - re_head.min() + 0.000000001)
    im_head = (im_head - im_head.min()) / (im_head.max() - im_head.min() + 0.000000001)
    re_head = torch.pow(re_head, 1.5)
    im_head = torch.pow(im_head, 1.5)

    # start calculation
    # ah
    re_score_a = re_head * re_relation_a - im_head * im_relation_a
    im_score_a = re_head * im_relation_a + im_head * re_relation_a

    # ah + b
    re_score_top = re_score_a + re_relation_b
    im_score_top = im_score_a + im_relation_b

    # ch
    re_score_c = re_head * re_relation_c - im_head * im_relation_c
    im_score_c = re_head * im_relation_c + im_head * re_relation_c

    # ch + d
    re_score_dn = re_score_c + re_relation_d
    im_score_dn = im_score_c + im_relation_d

    # (ah + b)(ch + d)^-1
    dn_re = torch.sqrt(re_score_dn * re_score_dn + im_score_dn * im_score_dn)

    up_re = torch.div(re_score_top * re_score_dn + im_score_top * im_score_dn, dn_re)
    up_im = torch.div(re_score_top * im_score_dn - im_score_top * re_score_dn, dn_re)

    if flag == "score":
        return torch.sum(up_re * re_tail + up_im * im_tail, 1, keepdim=True)
    elif flag == "forward":
        to_score = embeddings[0].weight
        to_score = to_score[:, :rank], to_score[:, rank:2*rank]
        return (
                up_re @ to_score[0].transpose(0, 1) + up_im @ to_score[1].transpose(0, 1)
            ), (
                torch.sqrt(re_head ** 2 + im_head ** 2),
                torch.sqrt(re_relation_a ** 2 + im_relation_a ** 2 + re_relation_c ** 2 +
                           im_relation_c ** 2 + re_relation_b ** 2 + im_relation_b ** 2 +
                           re_relation_d ** 2 + im_relation_d ** 2),
                torch.sqrt(re_tail ** 2 + im_tail ** 2)
                )
    elif flag == "get_queries":
        return torch.cat([up_re, up_im], 1)
    else:
        raise ValueError('unsupported flag: {}'.format(flag))
