#!/usr/bin/env python

from langchain_huggingface import HuggingFaceEmbeddings as hfe

def s_transformer(model: str = "all-MiniLM-L6-v2"):
    return hfe(model_name=model)
