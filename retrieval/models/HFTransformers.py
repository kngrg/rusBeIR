import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
from typing import List, Dict


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

    def encode_queries(self, queries: List[str], batch_size: int = 128):
        """
        Encodes queries
        :param queries: list of queries to encode
        :param batch_size: batch_size for encoding
        :return: queries embedding
        """
        return self._get_embeddings(queries, batch_size)

    def encode_passages(self, passages: List[str], batch_size: int = 128):
        """
        Encodes passages
        :param passages: list of passages to encode
        :param batch_size: batch_size for encoding
        :return:
        """
        return self._get_embeddings(passages, batch_size)

    def _get_embeddings(self, texts: List[str], batch_size: int = 128):

        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing Batches"):
            batch_texts = texts[i:i + batch_size]
            batch_dict = self.tokenizer(batch_texts, max_length=512, padding=True, truncation=True,
                                        return_tensors='pt')
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

            with torch.no_grad():
                outputs = self.model(**batch_dict)

            batch_embeddings = self._average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)

    def _average_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]