from typing import List, Dict
from rusBeIR.retrieval.models.HFTransformers import HFTransformers


class BGETransformers(HFTransformers):
    def __init__(self, model_name: str='cointegrated/LaBSE-en-ru', device: str = 'cuda'):
        """
        :param model_name: Name of the pre-trained BGE model from HF.
        :param device: Where to run the model ('cuda' or 'cpu').
        """
        super().__init__(model_name, device)

    def encode_queries(self, queries: List[str], batch_size: int = 128):
        """
        :param queries: List of query strings.
        :param batch_size: Batch size for encoding.
        :return: Query embeddings.
        """
        return self._get_embeddings(queries, batch_size, max_len=512, pooling_method='cls')

    def encode_passages(self, passages: List[str], batch_size: int = 128):
        """
        :param passages: List of passage strings.
        :param batch_size: Batch size for encoding.
        :return: Passage embeddings.
        """
        return self._get_embeddings(passages, batch_size, max_len=512, pooling_method='cls')
