# PubMed Indexing and Querying

#### This folder allows you to generate new PubMed-DSE tasks.


## Requirements

#### Elasticsearch

`pip install elasticsearch`

An Elasticsearch installation: The scripts in this repository work with
`Elasticsearch v6.5.4`.
It seems that Elasticsearch v7+ have made changes to the API which
break my code. Please make sure you are running a version before v7 such as:
[https://www.elastic.co/downloads/past-releases/elasticsearch-6-5-4].
Or alternatively, upgrade the script to the v7 API (I'd appreciate it!).

#### PubMed data

Download from [here](https://www.nlm.nih.gov/databases/download/pubmed_medline.html).

## Usage

Before using either of the scripts, first boot up the Elasticsearch server
by running:
 
`elasticsearch-<version>/bin/elasticsearch`

in the background, assuming that you downloaded `elasticsearch-<version>`.

### Creating a new index

The `build_pubmed_es_index.py` script can create a new PubMed index.
Please use `--help` for more info.

Example command:

```
python dse/elasticsearch_dse/build_pubmed_es_index.py \
--pubmed-path <path-to/raw-pubmed-data> \
--index-name pubmed_index
```

Your PubMed data folder should contain the `.xml.gz` PubMed Files. Eg.:
```
path-to
    |- raw-pubmed-data
        |- pubmed19n0001.xml.gz
        |- pubmed19n0002.xml.gz
        |- pubmed19n0003.xml.gz
        |- pubmed19n0004.xml.gz
```


### Generating new tasks

The `query_pubmed_es_index.py` script will query an existing pubmed elasticsearch
index and generate new tasks from it based on a given topic
(= one or more MeSH terms). Please use `--help` for more info.

Example command:
```
python dse/elasticsearch_dse/query_pubmed_es_index.py \
--output-path dse/pubmed_dse \
--index-name pubmed_index \
--task-sizes-lp 50 \
--task-sizes-u 1000 \
--task-topics D000818.D001921.D051381
```

## PubMed-DSE-15

Instead of creating your own PubMed-DSE tasks, you can download the 15
tasks that we made for our paper here:
* [Individual Files](http://nlp.biu.ac.il/~jacovia/pubmed-dse/)
* [Zip](http://nlp.biu.ac.il/~jacovia/pubmed-dse-15.zip) (535 MB)

The 15 topics are:
* Animals + Brain + Rats. `D000818 D001921 D051381`
* Adult + Middle Aged + HIV Infections. `D000328 D008875 D015658`
* Lymphatic Metastasis + Middle Aged + Neoplasm Staging. `D008207 D008875 D009367`
* Base Sequence + Molecular Sequence Data + Promoter Regions, Genetic. `D001483 D008969 D011401`
* Renal Dialysis + Kidney Failure, Chronic + Middle Aged. `D006435 D007676 D008875`
* Aged + Middle Aged + Laparoscopy. `D000368 D008875 D010535`
* Apoptosis + Cell Line, Tumor + Cell Proliferation. `D017209 D045744 D049109`
* Disease Models, Animal + Rats, Sprague-Dawley + Rats. `D004195 D017207 D051381`
* Liver + Rats, Inbred Strains + Rats. `D008099 D011919 D051381`
* Dose-Response Relationship, Drug + Rats, Sprague-Dawley + Rats. `D004305 D017207 D051381`
* Female + Infant, Newborn + Pregnancy. `D005260 D007231 D011247`
* Molecular Sequence Data + Phylogeny + Sequence Alignment. `D008969 D010802 D016415`
* Cells, Cultured + Mice, Inbred C57BL + Mice. `D002478 D008810 D051379`
* Dose-Response Relationship, Drug + Rats, Sprague-Dawley + Rats. `D004305 D017207 D051381`
* Brain + Magnetic Resonance Imaging + Middle Aged. `D001921 D008279 D008875`

Each task contains a `metadata.json` file with metadata about the task.
[Eg.](http://nlp.biu.ac.il/~jacovia/pubmed-dse/L20/D000328.D008875.D015658/metadata.json):

| Field        | Value |
| ------------- |:-------------:|
| P_size | 36604 |
| LP_size |	40 |
| U_size |	40000|
| train_LP_size	| 20 |
| train_U_size	| 19999|
| valid_LP_size	|10|
| valid_U_size	|9999|
|test_LP_size	|10|
|test_U_size	|10002|
|train_size	|20019|
|test_size	|10012|
|valid_size	|10009|
|precision	|0.183325|
|recall	|0.20055245596761842|
|mesh_conjunction|"D000328", "D008875", "D015658"|
|mesh_conjunction_str|"Adult", "Middle Aged", "HIV Infections"|

Where `precision` and `recall` refer to the metrics of the Elasticsearch query result.
Meaning that given an 40 labeled-positive samples, 40,000 unlabeled samples
were retrieved by Elasticsearch (based on Okapi BM25), and 18.3325% of the
unlabeled samples were actually positive (= they contain the MeSH terms
 `"Adult", "Middle Aged", "HIV Infections"`) and they amount to 20.0552% of
 the total number of positive documents in PubMed.
