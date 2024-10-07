from typing import List, Dict
from rusBeIR.retrieval.models.HFTransformers import HFTransformers
import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np


class rusSciTinyModel(HFTransformers):
    def __init__(self, model_name: str = 'mlsa-iai-msu-lab/sci-rus-tiny', maxlen: int = None, device: str = 'cuda'):
        """
        :param model_name: Name of the pre-trained BGE model from HF.
        :param device: Where to run the model ('cuda' or 'cpu').
        """
        super().__init__(model_name, maxlen=maxlen, device=device)

    def encode_queries(self, queries: List[str], batch_size: int = 128):
        """
        :param queries: List of query strings.
        :param batch_size: Batch size for encoding.
        :return: Query embeddings.
        """
        return self._get_embeddings(queries, batch_size)

    def encode_passages(self, passages: List[str], batch_size: int = 128):
        """
        :param passages: List of passage strings.
        :param batch_size: Batch size for encoding.
        :return: Passage embeddings.
        """
        return self._get_embeddings(passages, batch_size)

    def _get_embeddings(self, texts: List[str], batch_size: int = 128):
        """
        Get embeddings for given texts
        :param texts: list of texts to encode
        :param batch_size:
        :param pooling_method: 'average' or 'cls' are available by default
        :return: np.ndarray with embeddings
        """

        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing Batches"):
            batch_texts = texts[i:i + batch_size]
            batch_dict = self.tokenizer(batch_texts, max_length=self.max_len, padding=True, truncation=True,
                                        return_tensors='pt')
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

            with torch.no_grad():
                outputs = self.model(**batch_dict)

            batch_embeddings = self._average_pool(outputs, batch_dict['attention_mask'])
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            batch_embeddings = batch_embeddings.cpu().detach().numpy()[0]
            embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def _average_pool(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)