# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# import sys
# sys.path.append("/home/cl/xincan-f/embedding/dimension")


import argparse
from typing import Dict

import torch
from torch import optim

from kbc.datasets import Dataset
from kbc.regularizers import F2, N3
from kbc.optimizers import KBCOptimizer

# if you created a new model, import it here!!
from kbc.FiveStarE import FiveStarE
from kbc.FiveStarE_semi_hermitian import FiveStarE_semi_hermitian
from kbc.FiveStarE_hermitian import FiveStarE_hermitian
from kbc.FiveStarE_all_conjugate import FiveStarE_all_conjugate
from kbc.FiveStarE_para_conjugate import FiveStarE_para_conjugate
from kbc.FiveStarE_down_conjugate import FiveStarE_down_conjugate
from kbc.FiveStarE_unitary import FiveStarE_unitary
from kbc.FiveStarE_unit_circle import FiveStarE_unit_circle
from kbc.FiveStarE_logistic import FiveStarE_logistic
from kbc.FiveStarE_gamma import FiveStarE_gamma
from kbc.FiveStarE_tradition import FiveStarE_tradition
from kbc.FiveStarE_diffeomorphism import FiveStarE_diffeomorphism
from kbc.FiveStarE_half_reg import FiveStarE_half_reg
from kbc.CP import CP
from kbc.ComplEx import ComplEx
from kbc.ComplEx_all_conjugate import ComplEx_all_conjugate
from kbc.ComplEx_pseudo_conjugate import ComplEx_pseudo_conjugate

# import subprocess as sp

# print GPU info in output file
# COMMAND = 'nvidia-smi -l 1 --query-gpu=memory.used --format=csv'
# sp.check_output(COMMAND.split())


big_datasets = ['FB15K', 'WN', 'WN18RR', 'FB237', 'YAGO3-10', 'UMLS']
datasets = big_datasets

parser = argparse.ArgumentParser(
    description="Relational learning contraption"
)

parser.add_argument(
    '--dataset', choices=datasets,
    help="Dataset in {}".format(datasets)
)

# set choices for running the project
# if you created a new model, add it here!! 
models = ['FiveStarE', 'CP', 'ComplEx',
          'FiveStarE_hermitian', 'FiveStarE_semi_hermitian', 'FiveStarE_all_conjugate', 
          'FiveStarE_para_conjugate', 'FiveStarE_down_conjugate', 'FiveStarE_unitary', 'FiveStarE_unit_circle',
          'FiveStarE_logistic', 'FiveStarE_gamma', 'FiveStarE_tradition', 'FiveStarE_diffeomorphism', 'FiveStarE_half_reg',
          'ComplEx_all_conjugate', 'ComplEx_pseudo_conjugate']
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

parser.add_argument(
    '--save_dir', default=None, type=str,
    help="directory to save the output"
)

args = parser.parse_args()

dataset = Dataset(args.dataset)
examples = torch.from_numpy(dataset.get_train().astype('int64'))

# print config
# print(args)

# print number of head entities, number of relations * 2, number of tail entities
print(dataset.get_shape())

# if you created a new model, add it here!! 
model = {
    'FiveStarE': lambda: FiveStarE(dataset.get_shape(), args.rank, args.init),
    'FiveStarE_semi_hermitian': lambda: FiveStarE_semi_hermitian(dataset.get_shape(), args.rank, args.init),
    'FiveStarE_hermitian': lambda: FiveStarE_hermitian(dataset.get_shape(), args.rank, args.init),
    'FiveStarE_all_conjugate': lambda: FiveStarE_all_conjugate(dataset.get_shape(), args.rank, args.init),
    'FiveStarE_para_conjugate': lambda: FiveStarE_para_conjugate(dataset.get_shape(), args.rank, args.init),
    'FiveStarE_down_conjugate': lambda: FiveStarE_down_conjugate(dataset.get_shape(), args.rank, args.init),
    'FiveStarE_unitary': lambda: FiveStarE_unitary(dataset.get_shape(), args.rank, args.init),
    'FiveStarE_unit_circle': lambda: FiveStarE_unit_circle(dataset.get_shape(), args.rank, args.init),
    'FiveStarE_logistic': lambda: FiveStarE_logistic(dataset.get_shape(), args.rank, args.init),
    'FiveStarE_gamma': lambda: FiveStarE_gamma(dataset.get_shape(), args.rank, args.init),
    'FiveStarE_tradition': lambda: FiveStarE_tradition(dataset.get_shape(), args.rank, args.init),
    'FiveStarE_diffeomorphism': lambda: FiveStarE_diffeomorphism(dataset.get_shape(), args.rank, args.init),
    'FiveStarE_half_reg': lambda: FiveStarE_half_reg(dataset.get_shape(), args.rank, args.init),
    'CP': lambda: CP(dataset.get_shape(), args.rank, args.init),
    'ComplEx': lambda: ComplEx(dataset.get_shape(), args.rank, args.init),
    'ComplEx_all_conjugate': lambda: ComplEx_all_conjugate(dataset.get_shape(), args.rank, args.init),
    'ComplEx_pseudo_conjugate': lambda: ComplEx_pseudo_conjugate(dataset.get_shape(), args.rank, args.init),
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
    'SGD': lambda: optim.SGD(model.parameters(), lr=args.learning_rate),
}[args.optimizer]()

optimizer = KBCOptimizer(model, regularizer, optim_method, args.batch_size)


def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    """
    aggregate metrics for missing lhs and rhs
    :param mrrs:
    :param hits:
    :return:
    """
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    # return {'MRR': m, 'hits@[1,3,10]': h}
    return {
        'MRR': m, 'hits@[1,3,10]': h,
        'mrrs_lhs': mrrs['lhs'], 'mrrs_rhs': mrrs['rhs'],
        'hits_lhs': hits['lhs'], 'hits_rhs': hits['rhs'],
        }


import sys, subprocess, pdb, codecs
from datetime import datetime


shell_cmd = ' '.join(sys.argv)
gpu_name = subprocess.check_output('nvidia-smi --query-gpu=gpu_name --format=csv', shell=True)
gpu_name = gpu_name.decode().split('\n')[1]
# gpu_name = gpu_name.replace('NVIDIA GeForce GTX', '')

print('\t Parameters: ', shell_cmd)
print('\t GPU: ', gpu_name)

cur_loss = 0
curve = {'train': [], 'valid': [], 'test': []}
valid_epoch = []

best_valid_epoch = 0
best_valid_mrr = 0

for e in range(args.max_epochs):
    if e + 1 == args.max_epochs:
        last_epoch_start_time = datetime.now()

    cur_loss = optimizer.epoch(examples)

    if (e + 1) % args.valid == 0:
        train, valid, test = [
            avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
            for split in ['train', 'valid', 'test']
        ]

        curve['train'].append(train)
        curve['valid'].append(valid)
        curve['test'].append(test)
        valid_epoch.append(e + 1)

        # print current epoch number
        print("\t current epoch: ", e + 1)
        # print train and valid results
        print("\t TRAIN: ", train)
        print("\t VALID: ", valid)

        if valid['MRR'] > best_valid_mrr:
            best_valid_mrr = valid['MRR']
            best_valid_epoch = e + 1

    if e + 1 == args.max_epochs:
        last_epoch_end_time = datetime.now()
    
results = dataset.eval(model, 'test', -1)
print("\n\n TEST: ", results)


import os

# output csv
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)
        
with codecs.open(f'{args.save_dir}/log.csv', 'w') as up:
    line = '\n\n\t\t\tParameters\t{0}\n'.format(shell_cmd)
    up.write(line)
    
    line = 'LastEpochStartTime\t{0}\n'.format(last_epoch_start_time.strftime('%Y-%m-%d %H:%M:%S'))
    up.write(line)
    
    line = 'LastEpochDurationTime\t{0}\n'.format((last_epoch_end_time - last_epoch_start_time).total_seconds())
    up.write(line)

    line = 'GPU\t{0}\n'.format(gpu_name)
    up.write(line)

    line = 'BestValidEpoch\t{0}\n'.format(best_valid_epoch)
    up.write(line)

    line = 'BestValidMRR\t{0}\n'.format(best_valid_mrr)
    up.write(line)

    line = '\tMRR\tH@1\tH@3\tH@10\n'
    up.write(line)
    
    line = 'rhs\t{0:4f}\t{1:4f}\t{2:4f}\t{3:4f}\n'.format(test['mrrs_rhs'], test['hits_rhs'][0].item(), test['hits_rhs'][1].item(), test['hits_rhs'][2].item())
    up.write(line)
    
    line = 'lhs\t{0:4f}\t{1:4f}\t{2:4f}\t{3:4f}\n'.format(test['mrrs_lhs'], test['hits_lhs'][0].item(), test['hits_lhs'][1].item(), test['hits_lhs'][2].item())
    up.write(line)
    
    line = '(rhs+lhs)/2\t{0:4f}\t{1:4f}\t{2:4f}\t{3:4f}\n'.format(test['MRR'], test['hits@[1,3,10]'][0].item(), test['hits@[1,3,10]'][1].item(), test['hits@[1,3,10]'][2].item())
    up.write(line)


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# plot
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(2, 2)

ax0 = fig.add_subplot(gs[0, :])
ax0.plot(valid_epoch, [y['MRR'] for y in curve['train']], color = 'Navy', linewidth = '2')
ax0.plot(valid_epoch, [y['MRR'] for y in curve['valid']], color = 'DarkTurquoise', linewidth = '2')
ax0.plot(valid_epoch, [y['MRR'] for y in curve['test']], color = 'Gold', linewidth = '2')
ax0.legend(['train', 'valid', 'test'])
ax0.set_ylabel('MRR', fontsize=12)
ax0.set_xlabel('epoch', fontsize=12)

ax1 = fig.add_subplot(gs[1, :1])
ax1.plot(valid_epoch, [y['mrrs_lhs'] for y in curve['train']], color = 'Navy', linewidth = '1.1')
ax1.plot(valid_epoch, [y['mrrs_lhs'] for y in curve['valid']], color = 'DarkTurquoise', linewidth = '1.1')
ax1.plot(valid_epoch, [y['mrrs_lhs'] for y in curve['test']], color = 'Gold', linewidth = '1.1')
ax1.legend(['train', 'valid', 'test'])
ax1.set_ylabel('mrr_lhs', fontsize=12)
ax1.set_xlabel('epoch', fontsize=12)

ax2 = fig.add_subplot(gs[1, 1:])
ax2.plot(valid_epoch, [y['mrrs_rhs'] for y in curve['train']], color = 'Navy', linewidth = '1.1')
ax2.plot(valid_epoch, [y['mrrs_rhs'] for y in curve['valid']], color = 'DarkTurquoise', linewidth = '1.1')
ax2.plot(valid_epoch, [y['mrrs_rhs'] for y in curve['test']], color = 'Gold', linewidth = '1.1')
ax2.legend(['train', 'valid', 'test'])
ax2.set_ylabel('mrr_rhs', fontsize=12)
ax2.set_xlabel('epoch', fontsize=12)

plt.savefig(args.save_dir + '/MRR.png')

# save checkpoint
# torch.save(model.state_dict(),f'{args.save_dir}/model_{e+1}.pt')