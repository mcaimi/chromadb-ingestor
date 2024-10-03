#!/usr/bin/env python
try:
    from langchain_text_splitters import NLTKTextSplitter as nts
    from langchain_text_splitters import RecursiveCharacterTextSplitter as rts
except Exception as e:
    print(f"Caught Exception {e}")


def split_text_documents_recursive(documents: list = None,
                                   chunk_size: int = 1000,
                                   chunk_overlap: int = 0) -> list:
    if documents is None:
        return None

    splitter = rts(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def split_text_documents_nltk(documents: list = None,
                              separator: str = '\n\n',
                              language: str = 'english') -> list:

    if documents is None:
        return None

    splitter = nts(separator=separator, language=language)
    return splitter.split_documents(documents)
