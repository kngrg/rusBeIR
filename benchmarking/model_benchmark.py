from rusBeIR.beir.datasets.data_loader_hf import HFDataLoader
from rusBeIR.beir.retrieval.search.lexical import BM25Search as BM25
from rusBeIR.beir.retrieval.evaluation import EvaluateRetrieval
from rusBeIR.retrieval.models.E5Model import E5Model
from tqdm import tqdm
import json
from pathlib import Path
from datasets import load_dataset


# datasets is a list of datasets which are currently included in this benchmark
# elements are {'dataset_name' :(hf_corpus&queries_repo, hf_qrels_repo, split)}

retriever = EvaluateRetrieval()

class DatasetEvaluator:
    def __init__(self, model_name='bm25', k_values=[1, 3, 5, 10, 100]):

        metrics = ['NDCG', 'MAP', 'Recall', 'P', 'MRR']

        self.datasets = {'rus-mmarco': ('kngrg/rus-mmarco-google', 'kngrg/rus-mmarco-qrels', 'dev'),
                         'rus-scifact': ('kngrg/rus-scifact', 'kngrg/rus-scifact-qrels', 'test'),
                         'rus-arguana': ('kngrg/rus-arguana', 'kngrg/rus-arguana-qrels', 'test'),
                         'rus-nfcorpus': ('kngrg/rus-nfcorpus', 'kngrg/rus-nfcorpus-qrels', 'test'),
                         'rus-miracl': ('kngrg/rus-miracl', 'kngrg/rus-miracl-qrels', 'dev'),
                         'rus-xquad': ('kngrg/rus-xquad', 'kngrg/rus-xquad-qrels', 'dev'),
                         'rus-xquad-sentenes': ('kngrg/rus-xquad-sentences', 'kngrg/rus-xquad-sentences-qrels', 'dev'),
                         'rus-tydiqa': ('kngrg/rus-tydiqa', 'kngrg/rus-tydiqa-qrels', 'dev'),
                         'rubq': ('kngrg/rubq', 'kngrg/rubq-qrels', 'test'),
                         'ria-news': ('kngrg/ria-news', 'kngrg/ria-news-qrels', 'test')}

        self.metrics = metrics
        self.k_values = k_values
        self.model_type = model_name
        self.results_dir = Path('rusBeIR-results')
        self.model = None

        self.ndcg_sum = dict.fromkeys([f'NDCG@{k}' for k in k_values], 0)
        self.map_sum = dict.fromkeys([f'MAP@{k}' for k in k_values], 0)
        self.recall_sum = dict.fromkeys([f'Recall@{k}' for k in k_values], 0)
        self.precision_sum = dict.fromkeys([f'P@{k}' for k in k_values], 0)
        self.mrr_sum = dict.fromkeys([f'MRR@{k}' for k in k_values], 0)

    def retrieve(self, text_type: str = 'processed_text', results_path=Path('rusBeIR-results')):
        self.results_dir = Path(results_path)
        self.results_dir.mkdir(exist_ok=True)

        for dataset_name, args in tqdm(self.datasets.items(), desc="Processing datasets"):
            corpus, queries, _ = HFDataLoader(hf_repo=args[0], hf_repo_qrels=args[1],
                streaming=False, keep_in_memory=False, text_type=text_type).load(split=args[2])

            if self.model_type == 'bm25':
                hostname = "localhost:9200"
                index_name = dataset_name
                self.model = BM25(index_name=index_name, hostname=hostname, initialize=True)
                retriever = EvaluateRetrieval(retriever=self.model, k_values=self.k_values)
                results = retriever.retrieve(corpus, queries)

            elif self.model_type == 'e5':
                self.model = E5Model()
                corpus_emb = self.model.encode_passages(corpus)
                results = self.model.retrieve(queries, corpus_emb, list(corpus.keys()))

            out_file = self.results_dir / f"results_{dataset_name}_{args[2]}.json"
            with out_file.open('w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)


    def evaluate(self):
        evaluator = EvaluateRetrieval()
        processed_datasets = 0

        for dataset_name, args in tqdm(self.datasets.items(), desc="Evaluating results"):
            result_file = self.results_dir / f"results_{dataset_name}_{args[2]}.json"

            if not result_file.exists():
                print(f"File with results for {dataset_name} does not exist.")
                continue

            with result_file.open('r', encoding='utf-8') as f:
                results = json.load(f)

            qrels = load_dataset(args[1])[args[2]]
            qrels_dict = {str(qid): {str(cid): int(score)} for qid, cid, score in
                          zip(qrels['query-id'], qrels['corpus-id'], qrels['score'])}

            ndcg, _map, recall, precision = evaluator.evaluate(qrels=qrels_dict, results=results, k_values=self.k_values)
            mrr = retriever.evaluate_custom(qrels_dict, results, self.k_values, "mrr")

            for k in self.k_values:
                self.ndcg_sum[f'NDCG@{k}'] += ndcg[f'NDCG@{k}']
                self.map_sum[f'MAP@{k}'] += _map[f'MAP@{k}']
                self.recall_sum[f'Recall@{k}'] += recall[f'Recall@{k}']
                self.precision_sum[f'P@{k}'] += precision[f'P@{k}']
                self.mrr_sum[f'MRR@{k}'] += mrr[f'MRR@{k}']

            processed_datasets += 1

        if processed_datasets > 0:
            ndcg_avg = {f'NDCG@{k}': self.ndcg_sum[f'NDCG@{k}'] / processed_datasets for k in self.k_values}
            map_avg = {f'MAP@{k}': self.map_sum[f'MAP@{k}'] / processed_datasets for k in self.k_values}
            recall_avg = {f'Recall@{k}': self.recall_sum[f'Recall@{k}'] / processed_datasets for k in self.k_values}
            precision_avg = {f'P@{k}': self.precision_sum[f'P@{k}'] / processed_datasets for k in self.k_values}
            mrr_avg = {f'MRR@{k}': self.mrr_sum[f'MRR@{k}'] / processed_datasets for k in self.k_values}

            self.metrics_results = {
                "ndcg": ndcg_avg,
                "map": map_avg,
                "recall": recall_avg,
                "precision": precision_avg,
                "mrr": mrr_avg
            }
        else:
            print("None of results were retrieved.")
            self.metrics_results = {}

    def print_results(self):
        if not self.metrics_results:
            print("None of metrics were computed.")
            return

        for metric, results in self.metrics_results.items():
            print(f"Average {metric}:")
            for k, val in results.items():
                print(f"{k}: {val}")
            print('\n')
