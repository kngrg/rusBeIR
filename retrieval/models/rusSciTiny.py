from typing import List, Dict
from rusBeIR.retrieval.models.HFTransformers import HFTransformers
import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np


class rusSciTinyModel(HFTransformers):
    def __init__(self, model_name: str = 'mlsa-iai-msu-lab/sci-rus-tiny', maxlen: int = None, batch_size: int = 128,
                 device: str = 'cuda'):
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
        return self.get_sentence_embedding(queries)

    def encode_passages(self, passages: List[str]):
        """
        :param passages: List of passage strings.
        :param batch_size: Batch size for encoding.
        :return: Passage embeddings.
        """
        return self.get_sentence_embedding(passages)

    def get_sentence_embedding(self, texts):
        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Processing Batches"):
            batch_texts = texts[i:i + self.batch_size]
            encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt',
                                           max_length=self.max_len).to(self.model.device)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            sentence_embeddings = self._average_pool(model_output, encoded_input['attention_mask'])
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            embeddings.append(sentence_embeddings.cpu().detach().numpy())
        return np.vstack(embeddings)

    def _average_pool(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)