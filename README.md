Updated 21-Feb, 2022.

# DimStarE
This is the code for implementation of "(Feng et al. 2022) Constrain on Relational Dimension for Knowledge Graph Embeddings"


## Our methods
We aim to modify several Knowledge Graph Embedding (KGE) baseline models using our proposed Constraint Methods on Dimensions.

### Methods
• Constrain the relations with:
- [x] **"semi_hermitian"**
- [x] **"hermitian"**
- [x] **"all_conjugate"**

• Constrain the heads with:
- [ ] **"lnx"**

### Baseline models
- [x] 5StarE

<p align="center">
<img src="https://github.com/mojtabanayyeri/5-StartE/blob/5-StarE/img/OtherTransfType.png" alt="Transformation of Exsiting KGE Models." width="500"/>
</p>

- [x] ComplEx

### Datasets
- [ ] UMLS
- [x] FB15K-237
- [x] WN18RR
- [x] YAGO3-10
- [x] FB15K
- [x] WN18

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
python kbc/learn.py --dataset datasetName --model FiveStarE_all_conjugate --rank dimension --optimizer
Adagrad --learning_rate lr --batch_size batchSize --regularizer N3 --reg regularizerValue
 --max_epochs EpochNumber --valid 50
```

### License
kbc is CC-BY-NC licensed, as found in the LICENSE file

