#!/usr/bin/env python

try:
    from langchain_core.documents import Document
    from nltk import word_tokenize, sent_tokenize, download
    from prettytable import PrettyTable
    from tqdm import tqdm
    import os
    download("punkt_tab", download_dir=os.environ.get("NLTK_DATA", "/".join((os.environ.get("HOME"), "nltk_data"))))
except Exception as e:
    print(f"Caught Exception {e}")


# Prepare data corpus, load from training data set
def prepare_corpus(raw_loader: list) -> list:
    # collect statistics in relation to the data corpus
    corpus = []
    for doc in raw_loader:
        document_tokens = word_tokenize(doc.page_content)
        document_sentences = sent_tokenize(doc.page_content)
        if (len(document_tokens) != 0) and (len(document_sentences) != 0):
            corpus.append({"metadata": doc.metadata,
                           "raw_sentences": document_sentences,
                           "page_content": doc.page_content,
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
    prepared_data = []
    for doc in tqdm(corpus, ascii=True, desc="Preparing..."):
        doc_metadata = doc.get("metadata")
        prepared_data.append(Document(metadata=doc_metadata, page_content=doc.get("page_content")))
    return prepared_data
