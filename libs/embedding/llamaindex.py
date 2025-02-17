#!/usr/bin/env python

from llama_index.embeddings.langchain import LangchainEmbedding
from typing import Callable


def LlamaIndexEmbedding(langchain_embedding_adapter: Callable) -> Callable:
    return LangchainEmbedding(langchain_embedding_adapter)
