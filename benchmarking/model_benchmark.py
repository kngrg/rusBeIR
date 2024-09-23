from rusBeIR.beir.datasets.data_loader_hf import HFDataLoader
from rusBeIR.beir.retrieval.search.lexical import BM25Search as BM25
from rusBeIR.beir.retrieval.evaluation import EvaluateRetrieval
from rusBeIR.retrieval.models.e5 import E5Model
from tqdm import tqdm


# datasets is a list of datasets which are currently included in this benchmark
# elements are {'dataset_name' :(hf_corpus&queries_repo, hf_qrels_repo, split)}

class DatasetEvaluator:
    def __init__(self, k_values=[1, 3, 5, 10, 100, 1000], model='bm25'):

        metrics = ['NDCG', 'MAP', 'Recall', 'P', 'MRR']

        self.datasets = {'rus-mmarco': ('kngrg/rus-mmarco-google', 'kngrg/rus-mmarco-qrels', 'dev'),
                         'rus-scifact': ('kngrg/rus-scifact', 'kngrg/rus-scifact-qrels'),
                         'rus-arguana': ('kngrg/rus-arguana', 'kngrg/rus-arguana-qrels'),
                         'rus-nfcorpus': ('kngrg/rus-nfcorpus', 'kngrg/rus-nfcorpus-qrels'),
                         'rus-miracl': ('kngrg/rus-miracl', 'kngrg/rus-miracl-qrels'),
                         'rus-xquad': ('kngrg/rus-xquad', 'kngrg/rus-xquad-qrels'),
                         'rus-xquad-sentenes': ('kngrg/rus-xquad-sentences', 'kngrg/rus-xquad-sentences-qrels'),
                         'rus-tydiqa': ('kngrg/rus-tydiqa', 'kngrg/rus-tydiqa-qrels'),
                         'rubq': ('kngrg/rubq', 'kngrg/rubq-qrels'),
                         'ria-news': ('kngrg/ria-news', 'kngrg/ria-news-qrels')}

        self.metrics = metrics
        self.k_values = k_values
        self.model_type = model

        self.ndcg_sum = dict.fromkeys([f'NDCG@{k}' for k in k_values], 0)
        self.map_sum = dict.fromkeys([f'MAP@{k}' for k in k_values], 0)
        self.recall_sum = dict.fromkeys([f'Recall@{k}' for k in k_values], 0)
        self.precision_sum = dict.fromkeys([f'P@{k}' for k in k_values], 0)
        self.mrr_sum = dict.fromkeys([f'MRR@{k}' for k in k_values], 0)

    def evaluate(self):
        for dataset_name, args in tqdm(self.datasets.items(), desc="Processing datasets"):
            corpus, queries, qrels = HFDataLoader(hf_repo=args[0], hf_repo_qrels=args[1],
                streaming=False, keep_in_memory=False).load(split=args[2])

            retriever = EvaluateRetrieval()

            if self.model_type == 'bm25':
                hostname = "localhost:9200"
                index_name = dataset_name
                model = BM25(index_name=index_name, hostname=hostname, initialize=True)
                retriever = EvaluateRetrieval(model)
                results = retriever.retrieve(corpus, queries)

            elif self.model_type == 'e5':
                e5 = E5Model()
                corpus_emb = e5.encode_passages(corpus)
                results = e5.retrieve(queries, corpus_emb, corpus.keys())

            ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
            mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, "mrr")

            for k in self.k_values:
                self.ndcg_sum[f'NDCG@{k}'] += ndcg[f'NDCG@{k}']
                self.map_sum[f'MAP@{k}'] += _map[f'MAP@{k}']
                self.recall_sum[f'Recall@{k}'] += recall[f'Recall@{k}']
                self.precision_sum[f'P@{k}'] += precision[f'P@{k}']
                self.mrr_sum[f'MRR@{k}'] += mrr[f'MRR@{k}']

        ndcg_avg = {f'NDCG@{k}': self.ndcg_sum[f'NDCG@{k}'] / len(self.datasets) for k in self.k_values}
        map_avg = {f'MAP@{k}': self.map_sum[f'MAP@{k}'] / len(self.datasets) for k in self.k_values}
        recall_avg = {f'Recall@{k}': self.recall_sum[f'Recall@{k}'] / len(self.datasets) for k in self.k_values}
        precision_avg = {f'P@{k}': self.precision_sum[f'P@{k}'] / len(self.datasets) for k in self.k_values}
        mrr_avg = {f'MRR@{k}': self.mrr_sum[f'MRR@{k}'] / len(self.datasets) for k in self.k_values}

        self.metrics_results = {
            "ndcg": ndcg_avg,
            "map": map_avg,
            "recall": recall_avg,
            "precision": precision_avg,
            "mrr": mrr_avg
        }

    def print_results(self):
        for metric, results in self.metrics_results.items():
            print(f"Results for {metric}:")
            for k, val in results.items():
                print(f"{k}: {val}")
            print('\n')

evaluator = DatasetEvaluator(model='bm25')
evaluator.evaluate()
evaluator.print_results()
