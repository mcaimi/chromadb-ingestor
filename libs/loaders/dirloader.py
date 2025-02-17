#!/usr/bin/env python

from llama_index.core import SimpleDirectoryReader


def dirLoader(data_path: str = None, extensions: str = [".txt"]) -> SimpleDirectoryReader:
    if data_path is not None:
        return SimpleDirectoryReader(input_dir=data_path,
                                     required_exts=extensions,
                                     recursive=True)
    else:
        return None


def loadDocuments(loader: SimpleDirectoryReader = None):
    if loader is not None:
        return loader.load_data()
    else:
        return []


def iterate(loader: SimpleDirectoryReader = None):
    if loader is not None:
        for document in loader.iter_data():
            yield document
    else:
        return None
