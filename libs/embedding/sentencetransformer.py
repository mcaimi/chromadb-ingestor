#!/usr/bin/env python

from chromadb.utils import embedding_functions as ef


def s_transformer(model: str = "all-MiniLM-L6-v2"):
    return ef.SentenceTransformerEmbeddingFunction(model_name=model)
