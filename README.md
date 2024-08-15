# Rethinking Cancer Gene Identification through Graph Anomaly Analysis

The open-resourced code and data for AAAI 2025 Submission - HIPGNN.

HIPGNN is a HIerarchical-Perspective Graph Neural Network to identify cancer genes, 
- that not only determines spectral energy distribution variance on the spectral perspective, 
- but also perceives detailed protein interaction context on the spatial perspective.

## Environments

HIPGNN framework is implemented on Google Colab and major libraries include:

- [Pytorch == 2.3.0 + cu121](https://pytorch.org/)

- [Networkx](https://networkx.org/)

- [dgl](https://www.dgl.ai/)

- [numpy](https://github.com/numpy/numpy)


## Datasets

Protein interaction data are extracted from two publicly available protein databases: [STRINGdb](https://string-db.org/) and [CPDB](http://cpdb.molgen.mpg.de/)

Cancer gene-related attributes were extracted from the [TCGA](https://www.cancer.gov/ccg/research/genome-sequencing/tcga) database.

In addition to following [EMOGI](https://github.com/schulter/EMOGI)'s data preprocessing methodology, we reprocessed the two protein networks to extract protein interaction confidence and constructed a weighted graph.

We eventually obtained two weighted graph datasets [STRINGdb (data/graph_STRINGdb.bin)](data/graph_STRINGdb.bin) and [CPDB (data/graph_CPDB.bin)](data/graph_CPDB.bin)


## Method

Please see the example in HIPGNN.ipynb, it is run on Google Colab using the L4 GPU.

It mainly consists of two running steps.

- The first step is to preprocess the data:
```python
!python /content/drive/MyDrive/HIPGNN/graph_prep.py --dataset STRINGdb
```

- The second step is running the model:
```python
!python /content/drive/MyDrive/HIPGNN/main_node.py --dataset STRINGdb --loss 0.01 --n_train 0.8
```

## Baseline

- [EMOGI](https://github.com/schulter/EMOGI)[<sup>[1]</sup>](#refer-anchor-1)

- [MTGCN](https://github.com/weiba/MTGCN)[<sup>[2]</sup>](#refer-anchor-2)

- [SMG](https://github.com/C0nc/SMG)[<sup>[3]</sup>](#refer-anchor-3)






## Reference

<div id="refer-anchor-1"></div>

- [1] Schulte-Sasse R, Budach S, Hnisz D, et al. Integration of multiomics data with graph convolutional networks to identify new cancer genes and their associated molecular mechanisms[J]. Nature Machine Intelligence, 2021, 3(6): 513-526.

<div id="refer-anchor-2"></div>

- [2] Peng W, Tang Q, Dai W, et al. Improving cancer driver gene identification using multi-task learning on graph convolutional network[J]. Briefings in Bioinformatics, 2022, 23(1): bbab432.

<div id="refer-anchor-3"></div>

- [3] Cui Y, Wang Z, Wang X, et al. SMG: self-supervised masked graph learning for cancer gene identification[J]. Briefings in Bioinformatics, 2023, 24(6): bbad406.
