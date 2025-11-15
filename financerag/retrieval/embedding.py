from typing import Dict, List, Literal, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from financerag.common import Encoder


def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


# Adopted by https://github.com/beir-cellar/beir/blob/main/beir/retrieval/models/sentence_bert.py
class TransformersEncoder(Encoder):

    def __init__(
            self,
            model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
            query_prompt: Optional[
                str] = "Given a financial question, retrieve user replies that best answer the question",
            doc_prompt: Optional[str] = None,
            **kwargs
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            **kwargs
        ).to(torch.device("cuda"))

        self.query_prompt = query_prompt
        self.doc_prompt = doc_prompt

    def encode_queries(
            self, queries: List[str], batch_size: int = 16, **kwargs
    ) -> Union[np.ndarray, Tensor]:
        if self.query_prompt is not None:
            queries = [get_detailed_instruct(self.query_prompt, query) for query in queries]

        return self.encode(queries, batch_size=batch_size, **kwargs)

    def encode_corpus(
            self,
            corpus: Union[
                List[Dict[Literal["title", "text"], str]],
                Dict[Literal["title", "text"], List],
            ],
            batch_size: int = 8,
            **kwargs
    ) -> Union[np.ndarray, Tensor]:
        if isinstance(corpus, dict):
            sentences = [
                (
                    (corpus["title"][i] + " " + corpus["text"][i]).strip()
                    if "title" in corpus
                    else corpus["text"][i].strip()
                )
                for i in range(len(corpus["text"]))
            ]
        else:
            sentences = [
                (
                    (doc["title"] + " " + doc["text"]).strip()
                    if "title" in doc
                    else doc["text"].strip()
                )
                for doc in corpus
            ]
        if self.doc_prompt is not None:
            sentences = [self.doc_prompt + s for s in sentences]
        return self.encode(sentences, batch_size=batch_size, **kwargs)

    def encode(
            self,
            texts: List[str],
            batch_size: int,
            max_length: int = 512,
            padding: bool = True,
            truncation: bool = True,
            return_tensors: str = "pt"
            ):
        embeddings = []
        with torch.no_grad():
            for start_idx in range(0, len(texts), batch_size):
                batch_dict = self.tokenizer(
                    texts[start_idx: start_idx + batch_size],
                    max_length=max_length, 
                    padding=padding,
                    truncation=truncation,
                    return_tensors=return_tensors
                    )
                ctx_input = ctx_input.to(self.model.device)
                ctx_output = self.model(**ctx_input)
                embeddings.append(last_token_pool(ctx_output.last_hidden_state, batch_dict['attention_mask']))

        embeddings = torch.cat(embeddings, dim=0)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
