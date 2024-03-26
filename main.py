#!/usr/bin/env python

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="ChromaDB Ingestor",
                                     description="Document ingestor for ChromaDB")

    parser.add_argument("-r", "--remote", action="store_true")
    parser.add_argument("-i", "--input_dir", action="store", required=True)
    parser.add_argument("-d", "--persist_dir", action="store")
    parser.add_argument("-H", "--host", action="store")
    parser.add_argument("-p", "--port", action="store")
    parser.add_argument("-c", "--collection", action="store", required=True)
    arguments = parser.parse_args()

    if arguments.remote:
        print("Chroma Ingestor: Initializing Remote Client")
        from libs.remote_client import RemoteChromaClient
        try:
            cc = RemoteChromaClient(host=arguments.host, port=int(arguments.port), collection=arguments.collection)
            cc.GenerateEmbeddings(training_data_path=arguments.input_dir)
        except Exception as e:
            print(f"{e}")

    else:
        print("Chroma Ingestor: Initializing Local Client")
        from libs.client import ChromaClient
        try:
            cc = ChromaClient(persistence_directory=arguments.persist_dir)
            cc.TokenizeDocs(training_data_path=arguments.input_dir)
            cc.GenerateEmbeddings(collection_name=arguments.collection)
        except Exception as e:
            print(f"{e}")
