#!/usr/bin/env python

# ChromaDB Ingestor
# RAG example project
#
# v0.1 - Very Basic initial implementation - mcaimi@redhat.com

import argparse
from sys import exit
from yaml import safe_load, YAMLError
from libs.utils.console_utils import ANSIColors
from libs.utils.parameters import Parameters

if __name__ == "__main__":
    ttyWriter = ANSIColors()
    parser = argparse.ArgumentParser(prog="ChromaDB Ingestor",
                                     description="Document ingestor for ChromaDB")

    parser.add_argument("-c", "--config_file", action="store", required=True)
    arguments = parser.parse_args()

    ttyWriter.print_success(text=f"Loading Configuration File {arguments.config_file}...")
    try:
        with open(arguments.config_file, "r") as f:
            config_parms = safe_load(f)

        parms = Parameters(config_parms)
    except YAMLError as e:
        ttyWriter.print_error(text=e)
        exit(1)
    except Exception as e:
        ttyWriter.print_error(text=e)
        exit(1)

    ttyWriter.print_success(text=f"Running ingestor in remote={parms.chromadb.remote} mode...")

    if parms.embeddings.local:
        ttyWriter.print_warning(f"Running Sentence Transformer embedding with model {parms.embeddings.sentence_transformer.model}")
        from libs.embedding.sentencetransformer import s_transformer
        embed_func = s_transformer(model=parms.embeddings.sentence_transformer.model)
    else:
        ttyWriter.print_warning(f"Running Ollama embedding with model {parms.embeddings.ollama.model}")
        ttyWriter.print_warning(f"Ollama API URL: {parms.embeddings.ollama.baseurl}")
        from libs.embedding.ollama import ChromaOllamaEmbedder
        embed_func = ChromaOllamaEmbedder(model=parms.embeddings.ollama.model)

    if parms.chromadb.remote:
        ttyWriter.print_success("Chroma Ingestor: Initializing Remote Client")
        ttyWriter.print_warning(f"Chroma Host: {parms.chromadb.host} - Chroma Port: {parms.chromadb.port}")

        from libs.remote_client import RemoteChromaClient
        try:
            cc = RemoteChromaClient(host=parms.chromadb.host,
                                    port=int(parms.chromadb.port),
                                    collection=parms.chromadb.collection,
                                    embedding_function=embed_func)
            ttyWriter.print_warning(f"Objects in collection: {cc.Collection().count()}")
            cc.GenerateEmbeddings(training_data_path=parms.training_data.path, pattern=parms.training_data.pattern)
            ttyWriter.print_warning(f"Objects in collection after ingestion: {cc.Collection().count()}")
        except Exception as e:
            ttyWriter.print_error(f"{e}")

    else:
        ttyWriter.print_success("Chroma Ingestor: Initializing Local Client")
        ttyWriter.print_warning(f"Chroma persistence dir: {parms.chromadb.persist_dir}")

        from libs.client import ChromaClient
        try:
            cc = ChromaClient(persistence_directory=parms.chromadb.persist_dir,
                              embedding_function=embed_func)
            ttyWriter.print_warning(f"Objects in collection: {cc.Collection().count()}")
            cc.TokenizeDocs(training_data_path=parms.training_data.path, pattern=parms.training_data.pattern)
            cc.GenerateEmbeddings(collection_name=parms.chromadb.collection)
            ttyWriter.print_warning(f"Objects in collection after ingestion: {cc.Collection().count()}")
        except Exception as e:
            ttyWriter.print_error(f"{e}")
