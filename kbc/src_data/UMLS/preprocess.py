import pandas as pd

train = pd.read_csv('train.txt', sep='\t', header=None)
test = pd.read_csv('test.txt', sep='\t', header=None)
valid = pd.read_csv('valid.txt', sep='\t', header=None)

ids = train[0].to_list() + train[2].to_list() + test[0].to_list() + test[2].to_list() + valid[0].to_list() + valid[2].to_list()
idy = train[1].to_list() + test[1].to_list() + valid[1].to_list()

ids = list(set(ids))
idy = list(set(idy))

ids_dict = {x:i for i,x in enumerate(ids)}
idy_dict = {x:i for i,x in enumerate(idy)}

train[0] = train[0].map(ids_dict)
train[1] = train[1].map(idy_dict)
train[2] = train[2].map(ids_dict)

test[0] = test[0].map(ids_dict)
test[1] = test[1].map(idy_dict)
test[2] = test[2].map(ids_dict)

valid[0] = valid[0].map(ids_dict)
valid[1] = valid[1].map(idy_dict)
valid[2] = valid[2].map(ids_dict)

train.to_csv('train', index=None, header=None, sep='\t')
test.to_csv('test', index=None, header=None, sep='\t')
valid.to_csv('valid', index=None, header=None, sep='\t')
