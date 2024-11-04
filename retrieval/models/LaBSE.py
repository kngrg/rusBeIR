from typing import List, Dict
from rusBeIR.retrieval.models.HFTransformers import HFTransformers
import torch


class LaBSEModel(HFTransformers):
    def __init__(self, model_name: str = 'cointegrated/LaBSE-en-ru', maxlen: int = 64, batch_size:int=128, device: str = 'cuda'):
        """
        :param model_name: Name of the pre-trained BGE model from HF.
        :param device: Where to run the model ('cuda' or 'cpu').
        """
        super().__init__(model_name, maxlen=maxlen, batch_size=batch_size, device=device)

    def encode_queries(self, queries: List[str]):
        """
        :param queries: List of query strings.
        :param batch_size: Batch size for encoding.
        :return: Query embeddings.
        """
        return self._get_embeddings(queries, pooling_method='cls')

    def encode_passages(self, passages: List[str]):
        """
        :param passages: List of passage strings.
        :param batch_size: Batch size for encoding.
        :return: Passage embeddings.
        """
        return self._get_embeddings(passages, pooling_method='cls')

    def _cls_pool(self, model_output: torch.Tensor) -> torch.Tensor:
        return model_output.pooler_output