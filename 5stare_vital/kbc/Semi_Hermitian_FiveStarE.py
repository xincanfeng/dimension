from KBCModel import KBCModel
from typing import Tuple
import torch
from torch import nn


torch.manual_seed(1)


class Semi_Hermitian_FiveStarE(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(Semi_Hermitian_FiveStarE, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 6 * rank, sparse=True)
            for s in sizes[:2]
        ])

        for tmp in self.embeddings:
            nn.init.xavier_uniform_(tmp.weight, gain=1)

        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

    def score(self, x):
        return calculation(embeddings=self.embeddings, x=x, flag="scores", rank=self.rank)

    def forward(self, x):
        return calculation(embeddings=self.embeddings, x=x, flag="forward", rank=self.rank)

    def get_queries(self, queries: torch.Tensor):
        return calculation(embeddings=self.embeddings, x=queries, flag="queries", rank=self.rank)

    # get_entity embeddings
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
            chunk_begin:chunk_begin + chunk_size, :2 * self.rank
        ].transpose(0, 1)


def calculation(embeddings, x, flag, rank):
    '''
            semi-hermitian:
            c = Conjugate(b)
            thus,
            re_relation_c = re_relation_b
            im_relation_c = -im_relation_b
    '''
    # head
    lhs = embeddings[0](x[:, 0])
    # relation
    rel = embeddings[1](x[:, 1])

    # re_head: the real part of head
    # im_head: the imaginary part of head
    re_head, im_head = lhs[:, :rank], lhs[:, rank:2 * rank]

    # ditto, of relation
    # 0 - 1 re-a, 1 - 2 im-a
    re_relation_a, im_relation_a = rel[:, : rank], rel[:, rank:2 * rank]
    # 2 - 3 re-b 3-4 im-b
    # re_relation_c = re_relation_b
    # im_relation_c = -im_relation_b
    re_relation_b, im_relation_b = rel[:, 2 *
                                       rank:3 * rank], rel[:, 3 * rank:4 * rank]
    # 4 - 5 re-d, 5 - 6 im-d
    re_relation_d, im_relation_d = rel[:, 4 *
                                       rank:5 * rank], rel[:, 5 * rank: 6 * rank]

    # # ah
    re_score_a = re_head * re_relation_a - im_head * im_relation_a
    im_score_a = re_head * im_relation_a + im_head * re_relation_a

    # # ah + b
    re_score_top = re_score_a + re_relation_b
    im_score_top = im_score_a + im_relation_b

    # # ch
    re_score_c = re_head * re_relation_b - im_head * (-im_relation_b)
    im_score_c = re_head * (-im_relation_b) + im_head * re_relation_b

    # # ch + d
    re_score_dn = re_score_c + re_relation_d
    im_score_dn = im_score_c + im_relation_d

    # (ah + b)(ch+d)^-1
    dn_re = torch.sqrt(re_score_dn * re_score_dn + im_score_dn * im_score_dn)
    up_re = torch.div(re_score_top * re_score_dn +
                      im_score_top * im_score_dn, dn_re)
    up_im = torch.div(re_score_top * im_score_dn -
                      im_score_top * re_score_dn, dn_re)

    # calculation end

    if flag == "scores":
        rhs = embeddings[0](x[:, 2])
        # 0-1, 1-2
        re_tail, im_tail = rhs[:, :rank], rhs[:, rank:2 * rank]
        return torch.sum(up_re * re_tail + up_im * im_tail, 1, keepdim=True)
    elif flag == "forward":
        rhs = embeddings[0](x[:, 2])
        re_tail, im_tail = rhs[:, :rank], rhs[:, rank:2 * rank]
        to_score = embeddings[0].weight
        to_score = to_score[:, :rank], to_score[:, rank:2 * rank]
        return (up_re @ to_score[0].transpose(0, 1) + up_im @ to_score[1].transpose(0, 1)), \
               (torch.sqrt(re_head ** 2 + im_head ** 2),
                torch.sqrt(re_relation_a ** 2 + im_relation_a ** 2 + re_relation_b ** 2 + (-im_relation_b) ** 2
                           + re_relation_b ** 2 + im_relation_b ** 2 + re_relation_d ** 2 + im_relation_d ** 2),
                torch.sqrt(re_tail ** 2 + im_tail ** 2))
    elif flag == "queries":
        return torch.cat([up_re, up_im], 1)
    else:
        raise ValueError('unsupported flag: {}'.format(flag))
