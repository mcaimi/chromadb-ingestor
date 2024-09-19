#!/usr/bin/env python


def s_transformer(model: str = "all-MiniLM-L6-v2", hf=True):
    if (hf is True):
        from langchain_huggingface import HuggingFaceEmbeddings as hfe
        return hfe(model_name=model)
    else:
        from chromadb.utils import embedding_functions as ef
        return ef.SentenceTransformerEmbeddingFunction(model_name=model)
