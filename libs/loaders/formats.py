#!/usr/bin/env python

try:
    from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
except Exception as e:
    print(f"Caught exception {e}")


# create a new loader
def new_loader(path: str = ".",
               glob_pattern: str = None,
               loader_class = None,
               l_kwargs: dict = None,
               multithread: bool = False) -> DirectoryLoader:

    if loader_class is None:
        loader_class = TextLoader

    return DirectoryLoader(path=path,
                           glob=glob_pattern,
                           loader_cls=loader_class,
                           use_multithreading=multithread,
                           loader_kwargs=l_kwargs,
                           silent_errors=True,
                           show_progress=True)


# Directory Loader -> TXT support
def load_text_documents(path: str = ".", pattern: str = "**/*.txt",
                        multithread: bool = False) -> list:
    loader = new_loader(path,
                        glob_pattern=pattern,
                        loader_class=TextLoader,
                        l_kwargs={'autodetect_encoding': True},
                        multithread=True)

    return loader.load()


# Directory Loader -> PDF Support
def load_pdf_documents(path: str = ".", pattern: str = "**/*.pdf",
                       multithread: bool = False) -> list:

    loader = new_loader(path,
                        glob_pattern=pattern,
                        loader_class=PyPDFLoader,
                        multithread=True)

    return loader.load()


# callback table
loaders = { "text": load_text_documents,
            "pdf": load_pdf_documents }
