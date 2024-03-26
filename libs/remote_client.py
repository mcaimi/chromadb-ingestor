#!/usr/bin/env python

import uuid
from tqdm import tqdm
from chromadb import HttpClient, Collection
from .loaders.textdata import (load_text_documents, split_text_documents)
from .vectorstore.remote import (chroma_client, s_transformer)


class RemoteChromaClient(object):
    def __init__(self, host: str = "localhost",
                 port: int = 8080,
                 collection: str = "default",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        self._client: HttpClient = chroma_client(host=host, port=port)
        self._collection: Collection = self._client.get_or_create_collection(collection, embedding_function=s_transformer(model=embedding_model))

    def Client(self) -> HttpClient:
        return self._client

    def Collection(self) -> Collection:
        return self._collection

    def Heartbeat(self) -> int:
        return self._client.heartbeat()

    def GenerateEmbeddings(self, training_data_path: str = "."):
        # load custom knowledge data and tokenize it
        knowledge_body = load_text_documents(path=training_data_path)
        print(f"Loaded {len(knowledge_body)} Documents...")
        tokenized_docs = split_text_documents(documents=knowledge_body)
        print(f"Tokenized documents number: {len(tokenized_docs)}.")

        if len(tokenized_docs) > 0:
            for doc in tqdm(tokenized_docs, ascii=True, desc="Ingesting..."):
                self.Collection().add(ids=[str(uuid.uuid1())],
                                      documents=doc.page_content,
                                      metadatas=doc.metadata)

    def __str__(self) -> str:
        return f"ChromaDB Client: {self._client.database} - Collection: {self._collection}"
