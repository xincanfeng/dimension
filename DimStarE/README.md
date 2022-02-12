# 5*E
This is the code for implementation of [5* Knowledge Graph Embeddings with Projective Transformations](https://arxiv.org/abs/2006.04986) (AAAI 2021).

## Installation of baseline

The score functions of the following models are inlcuded in this code:

**Models**
- [x] CP
- [x] ComplEx
- [x] 5*E


The starting point is to install kbc framework. For which you need to create a conda environment with pytorch cython and scikit-learn :
```
conda create --name kbc_env python=3.7
source activate kbc_env
conda install --file requirements.txt -c pytorch
```

Then, in the next step, install the kbc package to this environment
```
python setup.py install
```

## Datasets

To download the datasets, go to the kbc/scripts folder and run:
```
chmod +x download_data.sh
./download_data.sh
```

Once the datasets are download, add them to the package data folder by running :
```
python kbc/process_datasets.py
```

This will create the files required to compute the filtered metrics.

## Running the code
Reproduce the results below with the following command :
```
python kbc/learn.py --dataset datasetName --model FiveStarE --rank dimension --optimizer
Adagrad --learning_rate lr --batch_size batchSize --regularizer N3 --reg regularizerValue
 --max_epochs EpochNumber --valid 50
```
## 5*E: Covering simultaneous transformations: translation, rotation, homothety, reflection and inversion.

Existing state of the art KGE models namely TransE, RotatE, ComplEx and QuatE cover only one or two transformation types among translation, rotation and homothety.
The following figure shows the transformations types followed by each of these models:

<p align="center">
<img src="https://github.com/mojtabanayyeri/5-StartE/blob/5-StarE/img/OtherTransfType.png" alt="Transformation of Exsiting KGE Models." width="500"/>
</p>


However, projective transformation includes simultaneous transformations covering: translation, rotation, homothety, reflection and inversion.
These transformation functions are used as the foundation of our approach because the landscape of transformation types then gets multiple forms namely: circular, elliptic, hyperbolic, loxodromic and parabolic. 

The following figures shows each of these trasnformation types (the visualiztion was done using [mobius transforms](https://github.com/timhutton/mobius-transforms)
). 


<p align="center">
<img src="https://github.com/mojtabanayyeri/5-StartE/blob/5-StarE/img/default.png" alt="Transformation of Exsiting KGE Models." width="500"/>
</p>

In our evaluations, we visualized embeddings for different relations usign the same dimension which is shown in the flloing figure:


<p align="center">
<img src="https://github.com/mojtabanayyeri/5-StartE/blob/5-StarE/img/flows.png" alt="Transformation of Exsiting KGE Models." width="500"/>
</p>

We captured the changes for inverse relations as you see. 
We fed a grid into the learned transformation by 5*E for partof and haspart relations as well as hypernym and hyponym relations which are in inverse relation. 
Lines mapped to circles and the transformation learned by each of pairs are inverse of each others. 


 
<p align="center">
<img src="https://github.com/mojtabanayyeri/5-StartE/blob/5-StarE/img/entities.png" alt="Transformation of Exsiting KGE Models." width="500"/>
</p>

We also investigated the types of transformations that 5*E learned on relation "hypernym" in WordNet.
Each pair of images visualises one dimension of the relation embedding. The top images show the embeddings of head entities of this relation in the KG at this dimension. Each entity embedding is visualised as a colored dot. The bottom images show the results of applying the relation specific transformation (the entity colors are preserved).

<p align="center">
<img src="https://github.com/mojtabanayyeri/5-StartE/blob/5-StarE/img/relation1.png" alt="Transformation of Exsiting KGE Models." width="500"/>
</p>



<p align="center">
<img src="https://github.com/mojtabanayyeri/5-StartE/blob/5-StarE/img/relation2.png" alt="Transformation of Exsiting KGE Models." width="500"/>
</p>



## License
kbc is CC-BY-NC licensed, as found in the LICENSE file
