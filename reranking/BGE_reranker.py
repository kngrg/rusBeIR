import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict


class ReRanker:
    def __init__(self, model_name: str, max_length: int = 8192):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to("cuda:0")
        self.model.eval()
        self.rerank_results = {}
        self.max_length = max_length

    def rerank(self,
               corpus: Dict[str, Dict[str, str]],
               queries: Dict[str, str],
               results: Dict[str, Dict[str, float]],
               top_k: int) -> Dict[str, Dict[str, float]]:

        self.rerank_results = {}

        for query_id in results:
            sentence_pairs, corpus_ids = [], []
            if len(results[query_id]) > top_k:
                for (doc_id, _) in sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)[:top_k]:
                    corpus_ids.append(doc_id)
                    corpus_text = (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).strip()
                    sentence_pairs.append([queries[query_id][:self.max_length], corpus_text[:self.max_length]])
            else:
                for doc_id in results[query_id]:
                    corpus_ids.append(doc_id)
                    corpus_text = (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).strip()
                    sentence_pairs.append([queries[query_id][:self.max_length], corpus_text[:self.max_length]])

            with torch.no_grad():
                inputs = self.tokenizer(sentence_pairs, padding=True, truncation=True, return_tensors='pt',
                                        max_length=512).to('cuda:0')
                scores = self.model(**inputs, return_dict=True).logits.view(-1, ).cpu().numpy()

            top_n_results = {corpus_ids[j]: float(scores[j]) for j in range(0, len(corpus_ids))}
            self.rerank_results[query_id] = top_n_results

        return self.rerank_results
