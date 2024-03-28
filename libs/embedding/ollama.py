#!/usr/bin/env/python

from typing import Callable
from langchain_community.embeddings import OllamaEmbeddings
from chromadb import Documents, EmbeddingFunction, Embeddings


def ollama_instance(base_url="http://localhost:11434", model="llama2:7b") -> Callable:
    return OllamaEmbeddings(model=model)


class ChromaOllamaEmbedder7b(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        return ollama_instance(model="llama2:7b").embed_documents(input)


class ChromaOllamaEmbedder13b(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        return ollama_instance(model="llama2:13b").embed_documents(input)


class ChromaOllamaEmbedder(EmbeddingFunction):
    def __init__(self, model="llama2:latest"):
        super().__init__()
        self.model = model

    def __call__(self, input: Documents) -> Embeddings:
        return ollama_instance(model=self.model).embed_documents(input)
