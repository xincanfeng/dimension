# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# import sys
# sys.path.append("/home/cl/xincan-f/embedding/dimension/DimStarE")

import argparse
from typing import Dict

import torch
from torch import optim

from kbc.datasets import Dataset
from kbc.regularizers import F2, N3
from kbc.optimizers import KBCOptimizer

# If you created a new model, import it here!!
from kbc.FiveStarE import FiveStarE
from kbc.FiveStarE_semi_hermitian import FiveStarE_semi_hermitian
from kbc.FiveStarE_hermitian import FiveStarE_hermitian
from kbc.FiveStarE_all_conjugate import FiveStarE_all_conjugate
from kbc.CP import CP
from kbc.ComplEx import ComplEx


big_datasets = ['FB15K', 'WN', 'WN18RR', 'FB237', 'YAGO3-10']
datasets = big_datasets

parser = argparse.ArgumentParser(
    description="Relational learning contraption"
)

parser.add_argument(
    '--dataset', choices=datasets,
    help="Dataset in {}".format(datasets)
)

# set choices for running the project
# If you created a new model, add it here!! 
models = ['FiveStarE', 'CP', 'ComplEx',
          'FiveStarE_hermitian', 'FiveStarE_semi_hermitian', 'FiveStarE_all_conjugate']
parser.add_argument(
    '--model', choices=models,
    help="Model in {}".format(models)
)

regularizers = ['N3', 'F2']
parser.add_argument(
    '--regularizer', choices=regularizers, default='N3',
    help="Regularizer in {}".format(regularizers)
)

optimizers = ['Adagrad', 'Adam', 'SGD']
parser.add_argument(
    '--optimizer', choices=optimizers, default='Adagrad',
    help="Optimizer in {}".format(optimizers)
)

parser.add_argument(
    '--max_epochs', default=50, type=int,
    help="Number of epochs"
)

parser.add_argument(
    '--valid', default=3, type=float,
    help="Number of epochs before valid"
)

parser.add_argument(
    '--rank', default=1000, type=int,
    help="Factorization rank"
)

parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Number of training examples utilized in one iteration"
)

parser.add_argument(
    '--reg', default=0, type=float,
    help="Regularization weight"
)

parser.add_argument(
    '--init', default=1e-3, type=float,
    help="Initial scale"
)

parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)

parser.add_argument(
    '--decay1', default=0.9, type=float,
    help="decay rate for the first moment estimate in Adam"
)

parser.add_argument(
    '--decay2', default=0.999, type=float,
    help="decay rate for second moment estimate in Adam"
)

args = parser.parse_args()

dataset = Dataset(args.dataset)
examples = torch.from_numpy(dataset.get_train().astype('int64'))

# print config
print(args)

# print number of head entities, number of relations * 2, number of tail entities
print(dataset.get_shape())

# If you created a new model, add it here!! 
model = {
    'FiveStarE': lambda: FiveStarE(dataset.get_shape(), args.rank, args.init),
    'FiveStarE_semi_hermitian': lambda: FiveStarE_semi_hermitian(dataset.get_shape(), args.rank, args.init),
    'FiveStarE_hermitian': lambda: FiveStarE_hermitian(dataset.get_shape(), args.rank, args.init),
    'FiveStarE_all_conjugate': lambda: FiveStarE_all_conjugate(dataset.get_shape(), args.rank, args.init),
    'CP': lambda: CP(dataset.get_shape(), args.rank, args.init),
    'ComplEx': lambda: ComplEx(dataset.get_shape(), args.rank, args.init),
}[args.model]()

regularizer = {
    'F2': F2(args.reg),
    'N3': N3(args.reg),
}[args.regularizer]

device = 'cuda'
model.to(device)

optim_method = {
    'Adagrad': lambda: optim.Adagrad(model.parameters(), lr=args.learning_rate),
    'Adam': lambda: optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.decay1, args.decay2)),
    'SGD': lambda: optim.SGD(model.parameters(), lr=args.learning_rate)
}[args.optimizer]()

optimizer = KBCOptimizer(model, regularizer, optim_method, args.batch_size)


def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    """
    aggregate metrics for missing lhs and rhs
    :param mrrs: d
    :param hits:
    :return:
    """
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    return {'MRR': m, 'hits@[1,3,10]': h}


cur_loss = 0
curve = {'train': [], 'valid': [], 'test': []}
for e in range(args.max_epochs):
    cur_loss = optimizer.epoch(examples)

    if (e + 1) % args.valid == 0:
        valid, test, train = [
            avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
            for split in ['valid', 'test', 'train']
        ]

        curve['valid'].append(valid)
        curve['test'].append(test)
        curve['train'].append(train)

        # print epoch number
        print("\t epoch number: ", e + 1)
        # print train and valid results
        print("\t TRAIN: ", train)
        print("\t VALID: ", valid)

results = dataset.eval(model, 'test', -1)
print("\n\n TEST: ", results)
