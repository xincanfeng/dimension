# $\mathrm{Compl\epsilon x}$, $5^{\bigstar}\mathrm{\epsilon}$

This is the code for implementation of "(Feng et al. 2022) Sharing Parameter by Conjugation for Knowledge Graph Embeddings in Complex Space"

## Our methods

Complex number employed in current Knowledge Graph Embedding (KGE) models enforces **multiplicative constraint** on representations; our method further adds **conjugate constraint**within the parameters. Note that we don't reduce the dimensions of the parameters, instead, we share the dimensions.

We economize **50\%** of the memory in relation embedding by sharing half of the parameters in the conjugate form. Our approach is at least comparable in accuracy to the baselines. In addition, our method reduces calculation in the regularization process, e.g., for the $5^{\bigstar}\mathrm{\epsilon}$ model, **31\%** of training time is saved on average for five benchmark datasets.

### Methods

• Constrain the relations with:

- [X] **conjugate**
- [X] **negative_conjugate**
- [X] **vertical_conjugate**
- [X] **horizontal_conjugate**

### Baseline models

- $\mathrm{ComplEx}$
- $5^{\bigstar}\mathrm{E}$

### Datasets

- [ ] UMLS
- [X] FB15K-237
- [X] WN18RR
- [X] YAGO3-10
- [X] FB15K
- [X] WN18

## Implementation of the project

### Prepare virtual environment

Setup virtual environment, and install required basic packages:

```
python -m venv .venv_kbc
source .venv_kbc/bin/activate
pip install -r requirements.txt
```

Install the kbc package into this environment. Please note that, you have to run this command to setup modified kbc package every time the model is modified:

```
python setup.py install
```

### Prepare datasets

Download datasets:

```
cd kbc/scripts
chmod +x download_data.sh
./download_data.sh
```

Once the datasets are downloaded, add them to the package data folder by running the command below. This will create the required files to compute the filtered metrics:

```
python kbc/process_datasets.py
```

### Running the code

```
python kbc/learn.py --dataset WN18RR --model FiveStarE_conjugate --regularizer N3 --optimizer Adagrad --max_epochs 100 --valid 50 --rank 500 --batch_size 100 --reg 1e-1 --init 1e-3 --learning_rate 1e-1 &> output/now.out &

```

## Hyper-parameters

Best hyper-parameter setting for $5^{\bigstar}\mathrm{E}$

| dataset  | model     | regularizer | optimizer | max_epochs | valid | rank | batch_size | reg     | init   | learning_rate |
| -------- | --------- | ----------- | --------- | ---------- | ----- | ---- | ---------- | ------- | ------ | ------------- |
| FB237    | FiveStarE | N3          | Adagrad   | 200        | 3     | 500  | 2000       | 1.E-01  | 1.E-03 | 1.E-02        |
| WN18RR   | FiveStarE | N3          | Adagrad   | 600        | 3     | 500  | 1000       | 5.E-01  | 1.E-03 | 1.E-01        |
| YAGO3-10 | FiveStarE | N3          | Adagrad   | 65         | 3     | 500  | 500        | 2.5E-03 | 1.E-03 | 1.E-01        |
| FB15K    | FiveStarE | N3          | Adagrad   | 25         | 1     | 500  | 1000       | 1.0E-03 | 1.E-03 | 5.E-02        |
| WN       | FiveStarE | N3          | Adagrad   | 400        | 3     | 500  | 500        | 5.E-02  | 1.E-03 | 1.E-01        |

Best hyper-parameter setting for $5^{\bigstar}\mathrm{\epsilon}$

| dataset  | model               | regularizer | optimizer | max_epochs | valid | rank | batch_size | reg     | init   | learning_rate |
| -------- | ------------------- | ----------- | --------- | ---------- | ----- | ---- | ---------- | ------- | ------ | ------------- |
| FB237    | FiveStarE_conjugate | N3          | Adagrad   | 640        | 3     | 500  | 1000       | 1.E-01  | 1.E-03 | 1.E-02        |
| WN18RR   | FiveStarE_conjugate | N3          | Adagrad   | 300        | 3     | 500  | 1000       | 5.E-01  | 1.E-03 | 1.E-01        |
| YAGO3-10 | FiveStarE_conjugate | N3          | Adagrad   | 60         | 3     | 500  | 1000       | 5.E-03  | 1.E-03 | 1.E-01        |
| FB15K    | FiveStarE_conjugate | N3          | Adagrad   | 20         | 1     | 500  | 1000       | 2.5E-03 | 1.E-03 | 1.E-01        |
| WN       | FiveStarE_conjugate | N3          | Adagrad   | 550        | 3     | 500  | 500        | 1.E-01  | 1.E-03 | 5.E-02        |

### License

kbc is CC-BY-NC licensed, as found in the LICENSE file
