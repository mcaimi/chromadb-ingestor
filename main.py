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
        if parms.embeddings.remote_service == "ollama":
            ttyWriter.print_warning(f"Running Ollama embedding with model {parms.embeddings.ollama.model}")
            ttyWriter.print_warning(f"Ollama API URL: {parms.embeddings.ollama.baseurl}")
            from libs.embedding.ollama import ollama_instance
            embed_func = ollama_instance(base_url=parms.embeddings.ollama.baseurl, model=parms.embeddings.ollama.model)
        elif parms.embeddings.remote_service == "openai":
            ttyWriter.print_warning(f"Running OpenAI-Compatible embedding with model {parms.embeddings.openai.model}")
            ttyWriter.print_warning(f"OpenAI API URL: {parms.embeddings.openai.baseurl} - APIKEY: {parms.embeddings.openai.apikey}")
            from libs.embedding.openai import openai_instance
            embed_func = openai_instance(base_url=parms.embeddings.openai.baseurl, model=parms.embeddings.openai.model, api_key=parms.embeddings.openai.apikey)
        else:
            ttyWriter.print_error(f"Unsupported Remote Service Type: {parms.embeddings.remote_service}. Aborting.")
            exit(-1)

    if parms.chromadb.remote:
        ttyWriter.print_success("Chroma Ingestor: Initializing Remote Client")
        ttyWriter.print_warning(f"Chroma Host: {parms.chromadb.host} - Chroma Port: {parms.chromadb.port}")

        if parms.backend == "langchain":
            ttyWriter.print_warning(f"Backend Selected: {parms.backend}")
            from libs.chroma.remote_client import RemoteChromaClient
            try:
                cc = RemoteChromaClient(host=parms.chromadb.host,
                                        port=int(parms.chromadb.port),
                                        collection=parms.chromadb.collection,
                                        collection_similarity=parms.chromadb.collection_similarity,
                                        embedding_function=embed_func)
                ttyWriter.print_warning(f"Objects in collection: {cc.Collection().count()}")
                for src in parms.training_data.sources:
                    cc.GenerateEmbeddings(training_data_path=src.get("path"),
                                          data_type=src.get("data_type"),
                                          pattern=src.get("pattern"),
                                          separator=parms.training_data.separator,
                                          language=parms.training_data.language)
                ttyWriter.print_warning(f"Objects in collection after ingestion: {cc.Collection().count()}")
            except Exception as e:
                ttyWriter.print_error(f"{e}")
        elif parms.backend == "llamaindex":
            ttyWriter.print_warning(f"Backend Selected: {parms.backend}")
            from libs.embedding.llamaindex import LlamaIndexEmbedding
            from libs.chroma.remote_client import LlamaIndexChromaRemote
            try:
                llama_embed_model = LlamaIndexEmbedding(embed_func)
                cc = LlamaIndexChromaRemote(host=parms.chromadb.host,
                                            port=int(parms.chromadb.port),
                                            collection=parms.chromadb.collection,
                                            collection_similarity=parms.chromadb.collection_similarity,
                                            embedding_function=llama_embed_model)
                ttyWriter.print_warning(f"Objects in collection: {cc.Collection().count()}")
                cc.GenerateEmbeddings(training_data_path=parms.llamaindex.data_path,
                                      pattern=parms.llamaindex.extensions,
                                      show_progress=True)
            except Exception as e:
                ttyWriter.print_error(f"{e}")
        else:
            ttyWriter.print_error(f"Backend {parms.backend} is not supported.")

    else:
        if parms.backend == "langchain":
            ttyWriter.print_success("Chroma Ingestor: Initializing Local Client")
            ttyWriter.print_warning(f"[{parms.backend}] Chroma persistence dir: {parms.chromadb.persist_dir}")

            from libs.chroma.client import ChromaClient
            try:
                cc = ChromaClient(persistence_directory=parms.chromadb.persist_dir,
                                  embedding_function=embed_func)
                ttyWriter.print_warning(f"Objects in collection: {cc.Collection().count()}")
                for src in parms.training_data.sources:
                    print(f"Ingesting data type {src.get('data_type')}")
                    cc.GenerateEmbeddings(training_data_path=src.get("path"),
                                          data_type=src.get("data_type"),
                                          pattern=src.get("pattern"),
                                          separator=parms.training_data.separator,
                                          language=parms.training_data.language)
                ttyWriter.print_warning(f"Objects in collection after ingestion: {cc.Collection().count()}")
            except Exception as e:
                ttyWriter.print_error(f"{e}")
        elif parms.backend == "llamaindex":
            ttyWriter.print_success("Chroma Ingestor: Initializing Local Client")
            ttyWriter.print_warning(f"[{parms.backend}] Chroma persistence dir: {parms.chromadb.persist_dir}")

            from libs.chroma.client import LlamaIndexChroma
            from libs.embedding.llamaindex import LlamaIndexEmbedding
            try:
                llama_embed_model = LlamaIndexEmbedding(embed_func)
                cc = LlamaIndexChroma(persistence_directory=parms.chromadb.persist_dir,
                                      collection=parms.chromadb.collection,
                                      collection_similarity=parms.chromadb.collection_similarity,
                                      embedding_function=llama_embed_model)
                ttyWriter.print_warning(f"Objects in collection: {cc.Collection().count()}")
                cc.GenerateEmbeddings(training_data_path=parms.llamaindex.data_path,
                                      pattern=parms.llamaindex.extensions,
                                      show_progress=True)
                ttyWriter.print_warning(f"Objects in collection after ingestion: {cc.Collection().count()}")
            except Exception as e:
                ttyWriter.print_error(f"{e}")
        else:
            ttyWriter.print_error(f"Backend {parms.backend} is not supported.")
