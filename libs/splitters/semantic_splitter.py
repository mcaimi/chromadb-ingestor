#!/usr/bin/env python
try:
    from typing import Callable
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.node_parser import SemanticSplitterNodeParser
except Exception as e:
    print(f"Caught Exception {e}")


def semanticSplitterPipeline(documents: list,
                             embedder: Callable) -> IngestionPipeline:
    ip: IngestionPipeline = IngestionPipeline(
            transformations=[
                    SemanticSplitterNodeParser(embed_model=embedder),
                ],
            )

    return ip
