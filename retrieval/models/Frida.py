from typing import List
from transformers import T5EncoderModel, AutoTokenizer
from rusBeIR.retrieval.models.HFTransformers import HFTransformers

class FridaTransformers(HFTransformers):
    def __init__(self, 
                 model_name: str = "ai-forever/FRIDA", 
                 maxlen: int = 512, 
                 batch_size: int = 128, 
                 device: str = 'cuda'):
        super().__init__(model_name, maxlen=maxlen, batch_size=batch_size, device=device)

    def load_model(self, model_name: str, device: str = 'cuda'):
        model = T5EncoderModel.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.eval()
        return model, tokenizer

    def encode_queries(self, queries: List[str]):
        """
        :param queries: List of query strings.
        :param batch_size: Batch size for encoding.
        :return: Query embeddings.
        """
        queries = [f"search_query: {query}" for query in queries]  
        return self._get_embeddings(queries, pooling_method='cls')

    def encode_passages(self, corpus: List[str]):
        """
        :param passages: List of passage strings.
        :param batch_size: Batch size for encoding.
        :return: Passage embeddings.
        """
        passages = [f"search_document: {doc}" for doc in corpus]
        return self._get_embeddings(passages,pooling_method='cls')