#!/bin/bash

# Ingestor startup script
source $HOME/.virtualenv/chromadb/bin/activate
pip list
$HOME/chromadb-ingestor/main.py $@
