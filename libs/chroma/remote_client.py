#!/usr/bin/env python

import uuid
from typing import Callable
from tqdm import tqdm
from langchain_chroma import Chroma
from chromadb import HttpClient, Collection
from libs.loaders.dataloader import prepare_corpus
from libs.loaders.formats import loaders
from libs.splitters.splitters import split_text_documents_nltk as splitter
from libs.vectorstore.remote import chroma_client
from libs.loaders.dirloader import dirLoader, loadDocuments
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext


# plain langchain connector for a remote chromadb instance
class RemoteChromaClient(object):
    def __init__(self, host: str = "localhost",
                 port: int = 8080,
                 collection: str = "default",
                 collection_similarity: str = "l2",
                 embedding_function: Callable = None):
        self._client: HttpClient = chroma_client(host=host, port=port)
        if embedding_function is None:
            raise Exception("RemoteChromaClient: embedding_function cannot be None: you must specify an embedding function")
        else:
            self._collection: Collection = self._client.get_or_create_collection(collection, metadata={"hnsw:space": collection_similarity})
            self._chroma_adapter: Chroma = Chroma(client=self._client, collection_name=collection, embedding_function=embedding_function)

    def Client(self) -> HttpClient:
        return self._client

    def Adapter(self) -> Chroma:
        return self._chroma_adapter

    def Collection(self) -> Collection:
        return self._collection

    def Heartbeat(self) -> int:
        return self._client.heartbeat()

    def GenerateEmbeddings(self, training_data_path: str = ".",
                           data_type: str = "text",
                           pattern: str = "**/*.txt",
                           separator: str = "\n\n",
                           language: str = "english",
                           multithread: bool = False):
        # load custom knowledge data and tokenize it
        knowledge_body = loaders[data_type](path=training_data_path,
                                            pattern=pattern,
                                            multithread=multithread)
        print(f"Loaded {len(knowledge_body)} Documents...")

        # prepare knowledge corpus
        corpus_data = prepare_corpus(knowledge_body)
        print(f"Prepared {len(corpus_data)} data sources...")

        tokenized_docs = splitter(documents=corpus_data,
                                  separator=separator,
                                  language=language)
        print(f"Tokenized documents number: {len(tokenized_docs)}.")

        if len(tokenized_docs) > 0:
            for doc in tqdm(tokenized_docs, ascii=True, desc="Ingesting..."):
                self.Adapter().add_documents(ids=[str(uuid.uuid1())], documents=[doc])

    def __str__(self) -> str:
        return f"ChromaDB Client: {self._client.database} - Collection: {self._collection}"


# llamaindex wrapper for a remote chromadb instance
class LlamaIndexChromaRemote(object):
    def __init__(self, host: str = "localhost",
                 port: int = 8080,
                 collection: str = "default",
                 collection_similarity: str = "l2",
                 embedding_function: Callable = None):
        self._client: HttpClient = chroma_client(host=host, port=port)
        if embedding_function is None:
            raise Exception("RemoteChromaClient: embedding_function cannot be None: you must specify an embedding function")
        else:
            self._embed_function = embedding_function
            self._collection: Collection = self._client.get_or_create_collection(collection, metadata={"hnsw:space": collection_similarity})
            self._vector_store: ChromaVectorStore = ChromaVectorStore(chroma_collection=self._collection)
            self._storage_context: StorageContext = StorageContext.from_defaults(vector_store=self._vector_store)

    def Client(self) -> HttpClient:
        return self._client

    def Adapter(self) -> ChromaVectorStore:
        return self._vector_store

    def Collection(self) -> Collection:
        return self._collection

    def Heartbeat(self) -> int:
        return self._client.heartbeat()

    def GenerateEmbeddings(self, training_data_path: str = ".",
                           pattern: dict = [".txt"],
                           show_progress: bool = True):
        # load custom knowledge data and tokenize it
        data_loader = dirLoader(training_data_path, extensions=pattern)
        knowledge_body = loadDocuments(data_loader)
        print(f"Loaded {len(knowledge_body)} Documents...")

        if len(knowledge_body) > 0:
            self._vector_index = VectorStoreIndex.from_documents(knowledge_body,
                                                                 storage_context=self._storage_context,
                                                                 embed_model=self._embed_function,
                                                                 show_progress=show_progress)

    def __str__(self) -> str:
        return f"ChromaDB Client: {self._client.database} - Collection: {self._collection}"
