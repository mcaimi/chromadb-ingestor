#!/bin/bash

# prepare virtual environment
python3 -m venv $HOME/.virtualenv/chromadb
source $HOME/.virtualenv/chromadb/bin/activate
pip install --upgrade pip

# clone repository
cd $HOME && git clone https://github.com/mcaimi/chromadb-ingestor.git
pip install -r chromadb-ingestor/requirements.txt
