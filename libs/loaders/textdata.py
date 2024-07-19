#!/usr/bin/env python

try:
    from langchain_core.documents import Document
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter as rts
    from nltk import word_tokenize, sent_tokenize, download
    from prettytable import PrettyTable
    from tqdm import tqdm
    import os
    download("punkt", download_dir=os.environ.get("NLTK_DATA", "/".join((os.environ.get("HOME"), "nltk_data"))))
except Exception as e:
    print(f"Caught Exception {e}")


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


def prepare_corpus(raw_loader: list) -> list:
    # collect statistics in relation to the data corpus
    corpus = []
    for doc in raw_loader:
        document_tokens = word_tokenize(doc.page_content)
        document_sentences = sent_tokenize(doc.page_content)
        corpus.append({"metadata": doc.metadata,
                       "raw_sentences": document_sentences,
                       "sentence_count": len(document_sentences),
                       "wordcount": len(document_tokens),
                       "vocabulary": set(document_tokens),
                       "lexical_richness": len(set(document_tokens)) / len(document_tokens)})

    # display corpus
    data_table = PrettyTable()
    data_table.field_names = ["Dataset", "Word Count", "Sentence Count", "Vocabulary", "Lexical Richness"]
    for dataset in corpus:
        data_table.add_row([dataset.get("metadata").get("source"), dataset.get("wordcount"), dataset.get("sentence_count"), len(dataset.get("vocabulary")), dataset.get("lexical_richness")])

    # display dataset statistics
    print(data_table)

    # return processed data
    tokenized_data = []
    for doc in tqdm(corpus, ascii=True, desc="Tokenizing..."):
        metadata = doc.get("metadata")
        for data in doc.get("raw_sentences"):
            tokenized_data.append(Document(metadata=metadata, page_content=data))
    return tokenized_data


def split_text_documents(documents: list = None,
                         chunk_size: int = 1000,
                         chunk_overlap: int = 0) -> list:
    if documents is None:
        return None

    splitter = rts(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)
