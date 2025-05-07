#!/usr/bin/env python

from typing import Callable
from chromadb import HttpClient, Collection
from libs.vectorstore.remote import chroma_client
from libs.loaders.dirloader import dirLoader, loadDocuments
from libs.splitters.semantic_splitter import semanticSplitterPipeline
from libs.utils.tools import splitList
from libs.loaders.dataloader import prepare_corpus
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext


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
                           show_progress: bool = True, batches: int = 1):
        # load custom knowledge data and tokenize it
        data_loader = dirLoader(training_data_path, extensions=pattern)
        knowledge_body = prepare_corpus(loadDocuments(data_loader))
        print(f"Loaded {len(knowledge_body)} Documents...")
        splitterPipeline = semanticSplitterPipeline(documents=knowledge_body,
                                                    embedder=self._embed_function)
        # run node splitter
        print(f"Splitting documents in semantically similar chunks...")
        nodes_list = splitterPipeline.run(documents=knowledge_body)
        print(f"Produced {len(nodes_list)} semantically split nodes")

        print(f"Preparing {batches} node batches...")
        nodes: list = splitList(nodes_list, batches)

        if len(nodes) > 0:
            for batch_of_nodes in nodes:
                self._vector_index = VectorStoreIndex(batch_of_nodes,
                                                      storage_context=self._storage_context,
                                                      embed_model=self._embed_function,
                                                      show_progress=show_progress)

    def __str__(self) -> str:
        return f"ChromaDB Client: {self._client.database} - Collection: {self._collection}"
