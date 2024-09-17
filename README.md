rusBeIR is a [BeIR](https://github.com/beir-cellar/beir)-based Information Retrieval benchmark for Russian language.
It contains 10 datasets from different domains and more datasets will be added in future. Some of these datasets are parts of multilingual datasets, other are translated from the original ones or were originally in russian. 

| Dataset | BM25 | E5 | 
|----|----|----|
|[rus-mmarco-google](https://huggingface.co/datasets/kngrg/rus-mmarco-google)|  ✅ |  ✅  | 
|[rus-mmarco-helsinki](https://huggingface.co/datasets/kngrg/rus-mmarco-helsinki)|  ✅ |  ✅  | 
|[rus-scifact](https://huggingface.co/datasets/kngrg/rus-scifact)|  ✅ |  ✅  | 
|[rus-arguana](https://huggingface.co/datasets/kngrg/rus-arguana)|  ✅ |  ✅  | 
|[rus-nfcorpus](https://huggingface.co/datasets/kngrg/rus-nfcorpus)|  ✅ |  ✅  | 

|[rus-miracl](https://huggingface.co/datasets/kngrg/rus-miracl)|  ✅ |  ✅  | 
|[rus-xquad](https://huggingface.co/datasets/kngrg/rus-xquad)|  ✅ |  ✅  | 
|[rus-xquad-sentences](https://huggingface.co/datasets/kngrg/rus-xquad-sentences)|  ✅ |  ✅  | 
|[rus-tydiqa](https://huggingface.co/datasets/kngrg/rus-tydiqa)|  ✅ |  ✅  | 
|[rubq](https://huggingface.co/datasets/kngrg/rubq)|  ✅ |  ✅  | 
|[ria-news](https://huggingface.co/datasets/kngrg/ria-news)|  ✅ |  ✅  | 


 Dataset   | Website | rusBEIR-Name | Domain | Public? | Type | Splits | Queries  | Corpus | Download | 
| -------- | -----| ---------| ------- | --------- |----------- | ----------- | ----------- |----------- | ------------------ |
| rus-MMARCO <br> (Russian part of mmarco) | [Homepage of original](https://huggingface.co/datasets/unicamp-dl/mmarco)| ``rus-mmmarco`` | Information Retrieval |✅ | Part of multilingual |``train``<br>``dev``|  6,980   |  8.84M     | [rus-mmarco-google](https://huggingface.co/datasets/kngrg/rus-mmarco-google) <br> <br> [rus-mmarco-helsinki](https://huggingface.co/datasets/kngrg/rus-mmarco-helsinki) |
| SciFact| [Homepage of original](https://github.com/allenai/scifact) | ``rus-scifact``| Fact Checking| ✅ | Translated |``train``<br>``test``|  300     |  5K    | [rus-scifact](https://huggingface.co/datasets/kngrg/rus-scifact)| 
| ArguAna    | [Homepage of original](http://argumentation.bplaced.net/arguana/data) | ``rus-arguana``| Argument Retrieval| ✅ | Translated |``test`` | 1,406     |  8.67K    |[rus-arguana](https://huggingface.co/datasets/kngrg/rus-arguana)|
| NFCorpus   | [Homepage of original](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/) | ``rus-nfcorpus`` | Bio-Medical IR | ✅ | Translated |``train``<br>``dev``<br>``test``|  323     |  3.6K     | [rus-nfcorpus](https://huggingface.co/datasets/kngrg/rus-nfcorpus)|
| Miracl   | [Homepage of original]() | ``rus-miracl`` | Information Retrieval | ✅ | Part of multilingual |``train``<br>``dev``|     |      | [rus-miracl](https://huggingface.co/datasets/kngrg/rus-miracl)|

Baselines could be found [here](https://docs.google.com/document/d/1F1zHZm36eiK_uhiptbDWAOCyXInBaXQ1ZkpZMLx9eEc/edit?usp=sharing).

All datasets are available at [HuggingFace](https://huggingface.co/collections/kngrg/rusbeir-66e28cb06e3e074be55ac0f3).
