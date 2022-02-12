# DimStarE code implementation
This is the code for implementation of "Constrain on Relational Dimension for Knowledge Graph Embeddings"

## Code path
Please help with **dim5stare/**,  
Please ignore **history/**, those are just history codes saved in case.

## How to run the code
###### setup virtual environment:

python -m venv .venv_kbc
source .venv_kbc/bin/activate
pip install -r requirements.txt
run **python setup.py install** every time when model is modified.

###### download datasets:

cd kbc/scripts
chmod +x download_data.sh
./download_data.sh

###### Once the datasets are downloaded, add them to the package data folder by running:

python kbc/process_datasets.py

This will create the files required to compute the filtered metrics.

## Running the code
python kbc/learn.py --dataset datasetName --model FiveStarE --rank dimension --optimizer
Adagrad --learning_rate lr --batch_size batchSize --regularizer N3 --reg regularizerValue
 --max_epochs EpochNumber --valid 50


# My methods
I tried to modify several baseline models using my methods. Specifically, the methods are:
1. constrain the parameters with **hermitian**, **semi-hermitian**.


# License
kbc is CC-BY-NC licensed, as found in the LICENSE file
