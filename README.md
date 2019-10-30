# Scalable Evaluation and Improvement of Document Set Expansion via Neural Positive-Unlabeled Learning
#### Authors:  Alon Jacovi, Gang Niu, Yoav Goldberg, Masashi Sugiyama
#### Paper: [https://arxiv.org/abs/1910.13339]

Abstract:
We consider the situation in which a user has collected a small set of 
documents on a cohesive topic, and they want to retrieve additional 
documents on this topic from a large collection. Information Retrieval 
(IR) solutions treat the document set as a query, and look for similar 
documents in the collection. We propose to extend the IR approach by 
treating the problem as an instance of positive-unlabeled (PU) 
learning---i.e., learning binary classifiers from only positive 
and unlabeled data, where the positive data corresponds to the query 
documents, and the unlabeled data is the results returned by the IR 
engine. Utilizing PU learning for text with big neural networks is a 
largely unexplored field. We discuss various challenges in applying PU 
learning to the setting, including an unknown class prior, extremely 
imbalanced data and large-scale accurate evaluation of models, and we 
propose solutions and empirically validate them. We demonstrate the 
effectiveness of the method using a series of experiments of retrieving 
PubMed abstracts adhering to fine-grained topics. We demonstrate 
improvements over the base IR solution and other baselines.

## Description

This repository has two separate functions:
1. Generating PubMed-DSE tasks
2. An AllenNLP extension package for training
PubMed-DSE models with Positive-Unlabeled Learning

The implementations are as described in the paper. 

## Usage

This repo contains a very small placeholder dataset. To use real data, either
generate your own PubMed tasks or download the ones we made (explained below).

To run a model, you can use any feature of your liking from AllenNLP
by appending this repository as an external package, and using an appropriate
`jsonnet` configuration:

```
allennlp train dse/experiments/nnpu.jsonnet \
-s <output_path> \
--include-package dse
```

Please note that to achieve similar performance to what is reported in the paper,
we recommend to fine-tune the `pu_gamma` hyper-parameter.

Please check AllenNLP for more details.

#### Config

The two config jsonnets in `experiments` contain example configurations
for running PU and PN experiments. Note that the dataset paths contain
additional parameters to the `DatasetReader` through HTTP protocol on the
dataset file path:

```
D000818.D001921.D051381/train.jsonl?label=label_L100
D000818.D001921.D051381/test.jsonl?label=label_true
```

The `label` argument allows to choose per-dataset which label to choose
from the dataset file. By default each dataset file contains two
labels: `label_L{x}` and `label_true`.

Under `label_L{x}`, where x is |LP| of the entire dataset, the labels will be
split between LP and U as in the PU setup.

Under `label_true`, the labels will be split between the true
P and N groups, as in the PN setting. This config uses the true
supervision for the "upper-bound" reference metric in the paper.



## PubMed-DSE
### Indexing and Querying PubMed (for task generation)

The scripts in `elasticsearch_dse` allow you to generate new PubMed-DSE
tasks.

Please check the relevant [README](https://github.com/sayaendo/document-set-expansion-pu/tree/master/dse/elasticsearch_dse).

### PubMed-DSE-15

In addition to creating your own PubMed-DSE tasks, you can download the 15
tasks that we made for our paper here:
* [Individual Files](http://nlp.biu.ac.il/~jacovia/pubmed-dse/)
* [Zip](http://nlp.biu.ac.il/~jacovia/pubmed-dse-15.zip) (535 MB)

The aforementioned [README](https://github.com/sayaendo/document-set-expansion-pu/tree/master/dse/elasticsearch_dse) contains more details.
