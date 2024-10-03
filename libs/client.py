#!/usr/bin/env python

import uuid
from tqdm import tqdm
from typing import Callable
from langchain_chroma import Chroma
from chromadb import PersistentClient, Collection
from .loaders.textdata import (load_text_documents, prepare_corpus)
from .splitters.splitters import split_text_documents_nltk as splitter


class ChromaClient(object):
    def __init__(self, persistence_directory: str = ".",
                 collection: str = "default",
                 collection_similarity: str = "l2",
                 embedding_function: Callable = None):
        self.persistence_dir: str = persistence_directory
        self.collection_name = collection
        self.collection_similarity = collection_similarity
        if embedding_function is None:
            raise Exception("ChromaClient: embedding_function cannot be None, you must specify and embedding function")
        else:
            self.embedding_function: Callable = embedding_function

        # instantiate client and adapter
        self.chroma_client: PersistentClient = PersistentClient(path=self.persistence_dir)
        self.collection = self.chroma_client.get_or_create_collection(self.collection_name,
                                                                      metadata={"hnsw:space": self.collection_similarity})

        self.chroma_adapter = Chroma(client=self.chroma_client, collection_name=self.collection_name, embedding_function=self.embedding_function)

    def Collection(self) -> Collection:
        return self.collection

    def GenerateEmbeddings(self,
                           training_data_path: str = ".",
                           pattern: str = "**/*.txt",
                           separator: str = '\n\n',
                           language: str = 'english',
                           multithread: bool = False) -> None:
        self.documents: list = load_text_documents(path=training_data_path,
                                                   pattern=pattern, multithread=multithread)
        print(f"Loaded {len(self.documents)} Documents...")
        knowledge_body = prepare_corpus(self.documents)

        # tokenize
        self.tokenized_documents: list = splitter(documents=knowledge_body,
                                                  separator=separator,
                                                  language=language)
        print(f"Tokenized documents number: {len(self.tokenized_documents)}.")
        del (knowledge_body)

        # ingest documents
        if len(self.tokenized_documents) > 0:
            for doc in tqdm(self.tokenized_documents, ascii=True, desc="Ingesting..."):
                self.chroma_adapter.add_documents(ids=[str(uuid.uuid1())], documents=[doc])

#    TO BE REMOVED
#    def GenerateEmbeddings(self,
#                           training_data_path: str = ".",
#                           pattern: str = "**/*.txt",
#                           chunk_size: int = 1000,
#                           chunk_overlap: int = 0,
#                           multithread: bool = False) -> None:
#        self.documents: list = load_text_documents(path=training_data_path,
#                                                   pattern=pattern, multithread=multithread)
#        print(f"Loaded {len(self.documents)} Documents...")
#        knowledge_body = prepare_corpus(self.documents)
#
#        # tokenize
#        self.tokenized_documents: list = splitter(documents=knowledge_body,
#                                                  chunk_size=chunk_size,
#                                                  chunk_overlap=chunk_overlap)
#        print(f"Tokenized documents number: {len(self.tokenized_documents)}.")
#        del (knowledge_body)
#
#        # ingest documents
#        if len(self.tokenized_documents) > 0:
#            for doc in tqdm(self.tokenized_documents, ascii=True, desc="Ingesting..."):
#                self.chroma_adapter.add_documents(ids=[str(uuid.uuid1())], documents=[doc])
