#!/usr/bin/env python

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter as rts


def load_text_documents(path: str = ".", pattern: str = "**/*.txt",
                        multithread: bool = False) -> list:
    loader = DirectoryLoader(path,
                             glob=pattern,
                             loader_cls=TextLoader,
                             loader_kwargs={'autodetect_encoding': True},
                             use_multithreading=multithread,
                             silent_errors=True,
                             show_progress=True)
    return loader.load()


def split_text_documents(documents: list = None,
                         chunk_size: int = 1000,
                         chunk_overlap: int = 0) -> list:
    if documents is None:
        return None

    splitter = rts(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)
