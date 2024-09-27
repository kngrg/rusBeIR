import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
from typing import List, Dict
import os
from rusBeIR.retrieval.models.HFTransformers import HFTransformers

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class QueryDataset(Dataset):
    def __init__(self, queries, queries_ids):
        self.queries = queries
        self.query_ids = queries_ids

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx], self.query_ids[idx]


class E5Model(HFTransformers):
    def __init__(self, model_name: str = 'intfloat/multilingual-e5-large', device: str = 'cuda'):
        super().__init__(model_name, device)

    def encode_queries(self, queries: List[str]):
        queries = [f"query: {query}" for query in queries]
        return self._get_embeddings_full(queries)

    def encode_passages(self, corpus: Dict[str, Dict[str, str]], batch_size: int = 128):
        passages = [f"passage: {doc['text']}" for doc in corpus.values()]
        return super().encode_passages(passages, batch_size)

    def retrieve(self, queries: Dict[str, str], corpus_emb: np.ndarray, corpus_ids: List[str],
                 top_n: int = 100) -> Dict[str, Dict[str, float]]:

        batch_size = 16
        num_workers = 4
        top_n = top_n

        query_dataset = QueryDataset(list(queries.values()), list(queries.keys()))
        data_loader = DataLoader(query_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

        results = {}

        for batch_queries, batch_query_ids in tqdm(data_loader, desc="Processing Queries"):
            query_embs = self.encode_queries(batch_queries)
            query_embs = query_embs.cpu().numpy()
            similarities = cosine_similarity(query_embs, corpus_emb)

            for idx, query_id in enumerate(batch_query_ids):
                query_similarities = similarities[idx].flatten()
                top_n_indices = query_similarities.argsort()[-top_n:][::-1]
                top_n_results = {corpus_ids[j]: float(query_similarities[j]) * 100 for j in top_n_indices}
                results[query_id] = top_n_results
        return results

    def _get_embeddings_full(self, texts: List[str]):
        batch_dict = self.tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        with torch.no_grad():
            outputs = self.model(**batch_dict)

        embeddings = super()._average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings
