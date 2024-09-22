import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import logging
from typing import List, Dict, Union, Tuple


os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


class QueryDataset(Dataset):
    def __init__(self, queries, query_ids):
        self.queries = queries
        self.query_ids = query_ids

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx], self.query_ids[idx]


class E5Model:
    def __init__(self, model_name: str = 'intfloat/multilingual-e5-large', device: str = 'cuda:0'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)

    def encode_queries(self, queries: List[str]) -> torch.Tensor:
        queries = [f"query: {query.strip()}" for query in queries]
        return self._get_embeddings_full(queries)

    def encode_passages(self, passages: List[str], batch_size: int = 128) -> torch.Tensor:
        passages = [f"passage: {passage.strip()}" for passage in passages]
        return self._get_embeddings(passages, batch_size=batch_size)

    def _get_embeddings(self, texts: List[str], batch_size: int = 128) -> torch.Tensor:
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing Batches"):
            batch_texts = texts[i:i + batch_size]
            batch_dict = self.tokenizer(batch_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

            with torch.no_grad():
                outputs = self.model(**batch_dict)

            batch_embeddings = self._average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            embeddings.append(batch_embeddings.cpu())

        return torch.cat(embeddings, dim=0)

    def _average_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def retrieve(self, queries: List[str], corpus_emb: torch.Tensor, corpus_ids: List[str], batch_size: int = 128,
                 top_n: int = 1000) -> Dict[str, Dict[str, float]]:
        query_embs = self.encode_queries(queries)

        results = {}
        query_embs = query_embs.unsqueeze(1)
        corpus_emb = corpus_emb.unsqueeze(0)

        similarities = F.cosine_similarity(query_embs, corpus_emb, dim=2)  
        similarities = similarities.cpu().detach().numpy()

        for idx, query in enumerate(queries):
            query_similarities = similarities[idx].flatten()
            top_n_indices = query_similarities.argsort()[-top_n:][::-1]
            top_n_results = {corpus_ids[j]: float(query_similarities[j]) * 100 for j in top_n_indices}
            results[query] = top_n_results

        return results

    def _get_embeddings_full(self, texts: List[str]) -> torch.Tensor:
        batch_dict = self.tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        with torch.no_grad():
            outputs = self.model(**batch_dict)

        embeddings = self._average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings
