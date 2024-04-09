#!/usr/bin/env python

import uuid
from tqdm import tqdm
from typing import Callable
from chromadb import PersistentClient, Collection
from .loaders.textdata import load_text_documents, split_text_documents


class ChromaClient(object):
    def __init__(self, persistence_directory: str = ".",
                 embedding_function: Callable = None):
        self.persistence_dir: str = persistence_directory
        if embedding_function is None:
            raise Exception("ChromaClient: embedding_function cannot be None, you must specify and embedding function")
        else:
            self.embedding_function: Callable = embedding_function

    def Collection(self) -> Collection:
        return self.collection

    def TokenizeDocs(self,
                     training_data_path: str = ".",
                     pattern: str = "**/*.txt",
                     chunk_size: int = 1000,
                     chunk_overlap: int = 0) -> None:
        self.documents: list = load_text_documents(path=training_data_path,
                                                   pattern=pattern, multithread=True)
        print(f"Loaded {len(self.documents)} Documents...")
        self.tokenized_documents: list = split_text_documents(documents=self.documents,
                                                              chunk_size=chunk_size,
                                                              chunk_overlap=chunk_overlap)
        print(f"Tokenized documents number: {len(self.tokenized_documents)}.")

    def GenerateEmbeddings(self, collection_name: str = "default") -> None:
        self.chroma_client: PersistentClient = PersistentClient(path=self.persistence_dir)
        self.collection = self.chroma_client.get_or_create_collection(collection_name,
                                                                      embedding_function=self.embedding_function)
        if len(self.tokenized_documents) > 0:
            for doc in tqdm(self.tokenized_documents, ascii=True, desc="Ingesting..."):
                self.collection.add(ids=[str(uuid.uuid1())],
                                      documents=doc.page_content,
                                      metadatas=doc.metadata)
