import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
from typing import List, Dict
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class QueryDataset(Dataset):
    def __init__(self, queries, queries_ids):
        self.queries = queries
        self.query_ids = queries_ids

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx], self.query_ids[idx]


class HFTransformers:
    def __init__(self, model_name: str, device: str = 'cuda'):
        """
        initialize model and tokenizer from hf-transformers
        :param model_name: hf-model repo
        :param device: where to run the model
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def encode_queries(self, queries: List[str], batch_size: int = 128):
        """
        Encodes queries
        :param queries: list of queries to encode
        :param batch_size: batch_size for encoding
        :return: queries embedding
        """
        return self._get_embeddings(queries, batch_size, max_len=512, pooling_method='average')

    def encode_passages(self, passages: List[str], batch_size: int = 128):
        """
        Encodes passages
        :param passages: list of passages to encode
        :param batch_size: batch_size for encoding
        :return:
        """
        return self._get_embeddings(passages, batch_size, max_len=512, pooling_method='average')

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
            similarities = cosine_similarity(query_embs, corpus_emb)

            for idx, query_id in enumerate(batch_query_ids):
                query_similarities = similarities[idx].flatten()
                top_n_indices = query_similarities.argsort()[-top_n:][::-1]
                top_n_results = {corpus_ids[j]: float(query_similarities[j]) * 100 for j in top_n_indices}
                results[query_id] = top_n_results
        return results

    def _get_embeddings(self, texts: List[str], batch_size: int = 128, max_len: int = 512, pooling_method: str = 'average'):
        """
        Get embeddings for given texts
        :param texts: list of texts to encode
        :param batch_size:
        :param max_len: max length for tokenizer
        :param pooling_method: 'average' or 'cls' are available by default
        :return: np.ndarray with embeddings
        """

        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing Batches"):
            batch_texts = texts[i:i + batch_size]
            batch_dict = self.tokenizer(batch_texts, max_length=max_len, padding=True, truncation=True,
                                        return_tensors='pt')
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

            with torch.no_grad():
                outputs = self.model(**batch_dict)
            if pooling_method == 'average':
                batch_embeddings = self._average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            elif pooling_method == 'cls':
                batch_embeddings = self._cls_pool(outputs.last_hidden_state)
            else:
                raise ValueError(f"Unknown pooling method: {pooling_method}")

            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)

    def _average_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def _cls_pool(self, model_output: torch.Tensor) -> torch.Tensor:
        return model_output[:, 0, :]