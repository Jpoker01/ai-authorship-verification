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


def _check_hash_seed() -> None:
    """Raise RuntimeError unless Python was started with PYTHONHASHSEED=0."""
    # Python sets this flag at start: 0 for PYTHONHASHSEED=0, 1 for any other seed or none
    started_with_hash_seed_0 = sys.flags.hash_randomization == 0
    if not started_with_hash_seed_0:
        raise RuntimeError(
            f"Start Python with PYTHONHASHSEED={HASH_SEED} - the synonyms depend on the hash seed.",
        )


def extract_isg_features(text: str) -> Counter[str]:
    """Build the graph of one text and count its lemmas, tags and dependency relations.

    Args:
        text: One cleaned text.

    Returns:
        The output of extract_features_from_isg for the graph of the text.

    Raises:
        RuntimeError: If Python was not started with PYTHONHASHSEED=0.
    """
    _check_hash_seed()
    isg = _load_isg()
    # one text per call - the texts of one call become one graph
    graph = isg.transform([{"id": 0, "doc": text}])[0]["graph"]
    return extract_features_from_isg(graph)


def _iter_text_batches(texts: Iterable[str]) -> Iterator[list[str]]:
    """Yield the texts in their order, in lists of at most BATCH_SIZE texts."""
    # an iterator, so every round takes the next texts and not the first ones again
    remaining_texts = iter(texts)
    while True:
        batch = list(islice(remaining_texts, BATCH_SIZE))
        if not batch:
            break
        yield batch


def _iter_isg_features(
    executor: Executor,
    texts: Iterable[str],
    on_batch: Callable[[int], None] | None = None,
) -> Iterator[Counter[str]]:
    """Yield the features of every text in the order of the texts.

    The texts go to the workers one batch at a time, and the next batch is read only after
    the caller has taken the features of every text of the current one.

    Args:
        executor: Pool of workers that build the graphs.
        texts: Texts of one split.
        on_batch: Called after every batch with the number of texts done so far.
    """
    texts_done = 0
    for batch in _iter_text_batches(texts):
        # map returns the features in the order of the batch, whichever worker finishes first
        batch_features = executor.map(extract_isg_features, batch, chunksize=CHUNK_SIZE)
        # hands the caller one text at a time and pauses until it asks for the next one
        yield from batch_features
        texts_done += len(batch)
        if on_batch is not None:
            on_batch(texts_done)


def _add_new_columns(features: Mapping[str, int], columns: dict[str, int]) -> None:
    """Give every feature that has no column yet the next free column.

    Args:
        features: Features of one text, e.g. {"LEX::walk": 1, "POS::VERB": 2}.
        columns: Column number of every known feature; the new features are added to it.
    """
    for name in features:
        if name not in columns:
            # the columns are numbered 0, 1, 2, ..., so the next free one is their count
            columns[name] = len(columns)


def _feature_row(
    features: Mapping[str, int],
    columns: Mapping[str, int],
) -> dict[int, int]:
    """Return the matrix row of one text as {column: count}.

    Args:
        features: Features of one text, e.g. {"LEX::walk": 1, "POS::VERB": 2}.
        columns: Column number of every known feature; a feature without a column is dropped.
    """
    row = {}
    for name, count in features.items():
        if name in columns:
            column = columns[name]
            row[column] = count
    return row


def _build_sparse_matrix(
    cell_counts: array,
    cell_columns: array,
    cells_per_row: Sequence[int],
    *,
    n_columns: int,
) -> sp.csr_matrix:
    """Build a sparse matrix from its non-zero cells, listed row after row.

    Args:
        cell_counts: Count of every non-zero cell, as float32.
        cell_columns: Column of every non-zero cell, as int32, at the position of its count.
        cells_per_row: Number of non-zero cells of every row.
        n_columns: Number of columns of the matrix.

    Returns:
        The matrix in the csr format of scipy, the cells of every row in the order of the columns.
    """
    # scipy takes the position where every row starts in the list of cells and where the
    # last row ends: for rows of 2, 2 and 1 cells that is 0, 2, 4, 5
    row_boundaries = np.concatenate(([0], np.cumsum(cells_per_row, dtype=np.int64)))
    matrix = sp.csr_matrix(
        (
            # np.frombuffer reads the typed arrays where they are, without copying them
            np.frombuffer(cell_counts, dtype=np.float32),
            np.frombuffer(cell_columns, dtype=np.int32),
            row_boundaries,
        ),
        shape=(len(cells_per_row), n_columns),
    )
    # the cells of a row are in the order its features were seen, not in the order of the columns
    matrix.sort_indices()
    return matrix


def _to_matrix(
    all_features: Iterable[Mapping[str, int]],
    columns: dict[str, int],
    *,
    add_columns: bool,
) -> sp.csr_matrix:
    """Return one row per text with the counts of its features.

    Args:
        all_features: Features of every text.
        columns: Column number of every known feature; new features are added to it if add_columns.
        add_columns: Whether a feature without a column gets the next free column or is dropped.

    Returns:
        The counts as float32, one column per feature in columns.
    """
    cell_counts = array("f")
    cell_columns = array("i")
    cells_per_row = []
    for features in all_features:
        if add_columns:
            _add_new_columns(features, columns)
        row = _feature_row(features, columns)
        cell_counts.extend(row.values())
        cell_columns.extend(row.keys())
        cells_per_row.append(len(row))
    return _build_sparse_matrix(
        cell_counts,
        cell_columns,
        cells_per_row,
        n_columns=len(columns),
    )


def _text_file_sha256(path: Path) -> str:
    """Return the sha256 of a text file, the same for a Windows (CRLF) and a Linux checkout."""
    text_bytes = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(text_bytes).hexdigest()


def _graph_module_sha256() -> str:
    """Return the sha256 of this file."""
    return _text_file_sha256(Path(__file__))


def _nltk_data_sha256() -> dict[str, str]:
    """Return the sha256 of every file of NLTK_RESOURCES found in the NLTK data folders.

    Raises:
        LookupError: If a resource is not installed.
    """
    import nltk

    nltk_data_sha256 = {}
    for resource in NLTK_RESOURCES:
        resource_path = Path(str(nltk.data.find(resource)))
        nltk_data_sha256[resource] = _sha256(resource_path)
    return nltk_data_sha256


def _sources(context: RunContext, data_names: Sequence[str]) -> dict[str, str]:
    """Return what the vectors depend on - this file, uv.lock and the data files.

    Raises:
        ValueError: If the context has no hash of one of the data files.
    """
    sources = {
        "graph_py_sha256": _graph_module_sha256(),
        "uv_lock_sha256": context.environment["uv_lock_sha256"],
    }
    for name in data_names:
        key = f"data_sha256.{name}"
        if key not in context.environment:
            raise ValueError(f"The run context has no {key}.")
        sources[key] = context.environment[key]
    return sources


def _check_data_files(
    cleaned_paths: Mapping[str, Path],
    sources: Mapping[str, str],
) -> None:
    """Raise ValueError unless every cleaned file is the file the run context hashed."""
    for path in cleaned_paths.values():
        if _sha256(path) != sources[f"data_sha256.{path.name}"]:
            raise ValueError(f"{path} is not the file the run context hashed.")


def _check_new_folder(vectors_dir: Path) -> Path:
    """Return the .partial folder next to vectors_dir after checking that neither exists.

    Raises:
        ValueError: If vectors_dir is relative.
        FileExistsError: If vectors_dir or vectors_dir.partial already exists.
    """
    if not vectors_dir.is_absolute():
        raise ValueError(f"Folder {vectors_dir} has to be absolute.")
    # vectors are first written to .partial folder
    partial_dir = vectors_dir.with_name(f"{vectors_dir.name}.partial")
    for folder in (vectors_dir, partial_dir):
        if folder.exists():
            raise FileExistsError(folder)
    return partial_dir
