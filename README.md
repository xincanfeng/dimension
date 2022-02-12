# DimStarE
This is the code for implementation of "Constrain on Relational Dimension for Knowledge Graph Embeddings"


## Our methods
We aim to modify several Knowledge Graph Embedding (KGE) baseline models using our proposed Constraint Methods on Dimensions.

Baseline models:
- [x] 5StarE
- [ ] ComplEx

Methods:
- [x] constrain the parameters with **hermitian**, **semi-hermitian**.
- [ ] constrain the head with **lnx**


## Implementation of the project
### Code path
:blue_heart:Please help with ~dim5stare/~,  
:see_no_evil: Please ignore history/**, those are just history codes saved in case.

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

Once the datasets are downloaded, add them to the package data folder by running the command below. This will create the files required to compute the filtered metrics:
```
python kbc/process_datasets.py
```

### Running the code
```
python kbc/learn.py --dataset datasetName --model FiveStarE --rank dimension --optimizer
Adagrad --learning_rate lr --batch_size batchSize --regularizer N3 --reg regularizerValue
 --max_epochs EpochNumber --valid 50
```

### License
kbc is CC-BY-NC licensed, as found in the LICENSE file

