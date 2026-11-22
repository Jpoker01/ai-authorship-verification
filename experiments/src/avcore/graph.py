"""Integrated syntactic graph (ISG) vectors of text pairs, built once with text2graphapi and saved.

Every text becomes one graph of its lemmas, part-of-speech tags and dependency relations.
The vector of a text counts the lemmas, the tags and the relations of its graph.
"""

import hashlib
import json
import logging
import multiprocessing
import sys
from array import array
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from functools import lru_cache
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

from avcore.features import (
    PAIRS_COLUMNS,
    SPLITS,
    PairFeatures,
    PairMatrices,
    SplitMatrices,
    _check_splits,
    iter_pair_texts,
    read_pair_labels,
    split_sides,
)
from avcore.tracking import RunContext, _sha256

if TYPE_CHECKING:
    from text2graphapi.src.IntegratedSyntacticGraph import ISG

SPACY_MODEL = "en_core_web_sm"
# text2graphapi keeps only ASCII letters, removes its stop words and lowercases the text
APPLY_PREP = True
# ensures reproducibility as text2graph picks synonyms for words based on this value
HASH_SEED = "0"
SETTINGS = {
    "spacy_model": SPACY_MODEL,
    "apply_prep": APPLY_PREP,
    "hash_seed": HASH_SEED,
}
# texts handed to a worker process at once
CHUNK_SIZE = 16
# texts sent to the workers per round, so the texts of a split are never all in memory
BATCH_SIZE = 4_096
TEXT2GRAPHAPI_LOGGERS = ["IntegratedSyntacticGraph", "Preprocessing", "Utils"]
COLUMNS_FILE = "columns.json"
MANIFEST_FILE = "manifest.json"
# NLTK data outside uv.lock decides tokens and synonyms, so the manifest records its hash
NLTK_RESOURCES = ["tokenizers/punkt_tab.zip", "corpora/wordnet.zip"]


def parse_graph_node(graph_node: object) -> tuple[str, str]:
    """Split the node name "lemma_TAG" into the lowercased lemma and the part-of-speech tag.

    Args:
        graph_node: Node of a text2graphapi graph.

    Returns:
        The lemma in lowercase and the tag.
    """
    lemma, tag = str(graph_node).rsplit("_", 1)
    return lemma.lower(), tag


def parse_graph_dependency(edge_data: Mapping[str, str]) -> str:
    """Return the dependency relation of an edge without the count text2graphapi appends.

    Args:
        edge_data: Attributes of an edge of a text2graphapi graph.

    Returns:
        The relation, e.g. "nsubj" for "nsubj_12".
    """
    # text2graphapi adds a count to the relation name: "nsubj" is stored as "nsubj_12"
    relation_with_count = edge_data["gramm_relation"]
    relation = relation_with_count.split("_")[0]
    return relation


def extract_features_from_isg(graph: nx.DiGraph) -> Counter[str]:
    """Count the lemmas, part-of-speech tags and dependency relations of a graph.

    Args:
        graph: Graph of one text built by text2graphapi.

    Returns:
        "LEX::<lemma>" and "POS::<tag>" with the number of nodes,
        "DEP::<relation>" with the number of edges.
    """
    features: Counter[str] = Counter()
    for node in graph.nodes:
        lemma, tag = parse_graph_node(node)
        features[f"LEX::{lemma}"] += 1
        if tag:
            features[f"POS::{tag}"] += 1
    for _, _, edge_data in graph.edges(data=True):
        dependency = parse_graph_dependency(edge_data)
        if dependency:
            features[f"DEP::{dependency}"] += 1
    return features


@lru_cache(maxsize=1)
def _load_isg() -> "ISG":
    """Create the graph builder of text2graphapi with a spaCy pipeline that tags and parses.

    The builder loads spaCy twice, so every process builds it once and reuses it for every text.
    """
    # imported here, not at the top: importing text2graphapi sets up logging and checks NLTK data
    import spacy
    from text2graphapi.src.IntegratedSyntacticGraph import ISG

    for logger_name in TEXT2GRAPHAPI_LOGGERS:
        logging.getLogger(f"text2graphapi.src.{logger_name}").setLevel(logging.WARNING)
    isg = ISG(
        graph_type="DiGraph",
        language="en",
        apply_prep=APPLY_PREP,
        output_format="networkx",
    )

    # text2graphapi loads spaCy without tok2vec, the layer its tagger and parser read,
    # so its pipeline is replaced by the full one (ner is left out, the graph does not use it)
    text2graphapi_pipeline = isg.prep.nlp
    working_pipeline = spacy.load(SPACY_MODEL, exclude=["ner"])
    # text2graphapi raised the text length limit of spaCy (1 000 000 characters by default)
    working_pipeline.max_length = text2graphapi_pipeline.max_length
    isg.prep.nlp = working_pipeline
    return isg
