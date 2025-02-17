#!/usr/bin/env python

import uuid
from tqdm import tqdm
from typing import Callable
from langchain_chroma import Chroma
from chromadb import PersistentClient, Collection
from libs.loaders.dataloader import prepare_corpus
from libs.loaders.formats import loaders
from libs.splitters.splitters import split_text_documents_nltk as splitter
from libs.loaders.dirloader import dirLoader, loadDocuments
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext


# ChromaDB client for a local session, langchain version
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
                           data_type: str = "text",
                           pattern: str = "**/*.txt",
                           separator: str = '\n\n',
                           language: str = 'english',
                           multithread: bool = False) -> None:
        self.documents: list = loaders[data_type](path=training_data_path,
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

        # cleanup
        del self.documents


# llamaindex wrapper for a local chromadb instance
class LlamaIndexChroma(object):
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
            self._embed_function: Callable = embedding_function

        # instantiate client and adapter
        self.chroma_client: PersistentClient = PersistentClient(path=self.persistence_dir)
        self._collection = self.chroma_client.get_or_create_collection(self.collection_name,
                                                                       metadata={"hnsw:space": self.collection_similarity})
        self._vector_store: ChromaVectorStore = ChromaVectorStore(chroma_collection=self._collection)
        self._storage_context: StorageContext = StorageContext.from_defaults(vector_store=self._vector_store)

    def Collection(self) -> Collection:
        return self._collection

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
