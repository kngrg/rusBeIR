## What is it?

rusBeIR is a [BeIR](https://github.com/beir-cellar/beir)-based Information Retrieval benchmark for Russian language.
It contains 10 datasets from different domains and more datasets will be added in future. Some of these datasets are parts of multilingual datasets, other are translated from the original ones or were originally in russian. 
## Baselines
Baselines could be found [here]([https://docs.google.com/document/d/1F1zHZm36eiK_uhiptbDWAOCyXInBaXQ1ZkpZMLx9eEc/edit?usp=sharing](https://docs.google.com/spreadsheets/d/19jUZigy-AolNOOhT0EzNggiRoEcvfqL7HRpq0bwHqXc/edit?usp=sharing)).
## Installation 
``` python
!git clone git@github.com:kngrg/rusBeIR.git
``` 

##  Available Datasets

 Dataset   | Website | rusBEIR-Name | Domain | Public? | Type | Splits | Queries  | Corpus | Download | 
| -------- | -----| ---------| ------- | --------- |----------- | ----------- | ----------- |----------- | ------------------ |
| rus-MMARCO <br> (Russian part of mmarco) | [Homepage of original](https://huggingface.co/datasets/unicamp-dl/mmarco)| ``rus-mmmarco`` | Information Retrieval |✅ | Part of multilingual |``dev``<br>``train``|  ``dev:`` 6,980 <br><br> ``train:`` 502,939   |  8.84M     | [rus-mmarco-google](https://huggingface.co/datasets/kngrg/rus-mmarco-google) <br> <br> [rus-mmarco-helsinki](https://huggingface.co/datasets/kngrg/rus-mmarco-helsinki) |
| SciFact| [Homepage of original](https://github.com/allenai/scifact) | ``rus-scifact``| Fact Checking| ✅ | Translated |``test``<br>``train``|  ``test:`` 300 <br> ``train:`` 800   |  5K    | [rus-scifact](https://huggingface.co/datasets/kngrg/rus-scifact)| 
| ArguAna    | [Homepage of original](http://argumentation.bplaced.net/arguana/data) | ``rus-arguana``| Argument Retrieval| ✅ | Translated |``test`` | 1,406     |  8.67K    |[rus-arguana](https://huggingface.co/datasets/kngrg/rus-arguana)|
| NFCorpus   | [Homepage of original](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/) | ``rus-nfcorpus`` | Bio-Medical IR | ✅ | Translated |``train``<br>``dev``<br>``test``| ``train:`` 2590 <br> ``dev:`` 324 <br> ``test:``  323     |  3.6K     | [rus-nfcorpus](https://huggingface.co/datasets/kngrg/rus-nfcorpus)|
| MIRACL   | [Homepage of original](https://github.com/project-miracl/miracl) | ``rus-miracl`` | Information Retrieval | ✅ | Part of multilingual |``train``<br>``dev``| ``train:`` 4683 <br>``dev:`` 1252    |   9.54M   | [rus-miracl](https://huggingface.co/datasets/kngrg/rus-miracl)|
| XQuAD   | [Homepage of original](https://github.com/google-deepmind/xquad) | ``rus-xquad`` | QA | ✅ | Part of multilingual |``dev`` | 1190    |  240    | [rus-xquad](https://huggingface.co/datasets/kngrg/rus-xquad)|
| XQuAD-sentences   | [Homepage of original](https://github.com/google-deepmind/xquad) | ``rus-xquad-sentences`` | QA | ✅ | Part of multilingual | ``dev`` | 1190   | 1.2K     | [rus-xquad-sentences](https://huggingface.co/datasets/kngrg/rus-xquad-sentences)|
| TyDi QA   | [Homepage of original](https://github.com/google-research-datasets/tydiqa) | ``rus-tydiqa`` | QA | ✅ | Part of multilingual |``dev``| 1162     |   89K   | [rus-tydiqa](https://huggingface.co/datasets/kngrg/rus-tydiqa)|
| RuBQ   | [Homepage of original]() | ``rubq`` | QA | ✅ | Russian originally  |``test``| 1692     |   57K   | [rubq](https://huggingface.co/datasets/kngrg/rubq)|
| Ria-News   | [Homepage of original]() | ``ria-news`` | QA | ✅ | Russian originally | ``test``|  10K    |   704K  | [ria-news](https://huggingface.co/datasets/kngrg/ria-news)|

All datasets are available at [HuggingFace](https://huggingface.co/collections/kngrg/rusbeir-66e28cb06e3e074be55ac0f3).


##  Quick Example

```python
"""
This example shows how to evaluate ElasticSearch-BM25 in rusBeIR.
We advise you to use docker for running ElasticSearch. 
To be able to run the code below you must have docker locally installed in your machine.
To install docker on your local machine, please refer here: https://docs.docker.com/get-docker/

After docker installation, please follow the steps below to get docker container up and running:

1. docker pull docker.elastic.co/elasticsearch/elasticsearch:7.5.2
2. docker run -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:7.5.2
""" 


from rusBeIR.beir.datasets.data_loader_hf import HFDataLoader
from rusBeIR.beir.retrieval.evaluation import EvaluateRetrieval
from rusBeIR.beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

#### Load dataset via HF 
corpus, queries, qrels = HFDataLoader(hf_repo="kngrg/rus-mmarco", hf_repo_qrels="kngrg/rus-mmarco", streaming=False,
                                       keep_in_memory=False).load(split='test') # select necessary split train/test/dev

#### Initialize BM25 model
from rusBeIR.beir.retrieval.search.lexical import BM25Search as BM25
from rusBeIR.beir.retrieval.evaluation import EvaluateRetrieval

#### Provide parameters for elastic-search
hostname = "localhost:9200"
index_name = "mmarco" 

#### Index dataset and retrieve documents
model = BM25(index_name=index_name, hostname=hostname, initialize=True)
retriever = EvaluateRetrieval(model)
results = retriever.retrieve(corpus, queries)

#### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

#### Evaluate your model with MRR@k where k = [1,3,5,10,100,1000]
mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, "mrr")
```
