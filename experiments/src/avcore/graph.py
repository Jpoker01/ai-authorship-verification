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
# a worker is replaced after this many chunks, so the memory of its caches stays bounded
CHUNKS_PER_WORKER = 250
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


def _build_split(
    executor: Executor,
    name: str,
    cleaned_path: Path,
    columns: dict[str, int],
    on_progress: Callable[..., None] | None,
) -> SplitMatrices:
    """Build the vectors of both texts of every pair of one split."""
    pairs_df = read_pair_labels(cleaned_path)
    texts = iter_pair_texts(cleaned_path)
    texts_total = 2 * len(pairs_df)

    def report(texts_done: int) -> None:
        if on_progress is not None:
            on_progress(split=name, texts_done=texts_done, texts_total=texts_total)

    all_features = tqdm(
        _iter_isg_features(executor, texts, on_batch=report),
        total=texts_total,
        desc=f"ISG {name}",
    )
    # only the training texts add new columns
    vectors = _to_matrix(all_features, columns, add_columns=name == "train")
    return split_sides(pairs_df, vectors)


def _write_isg_vectors(
    vectors_dir: Path,
    isg_vectors: PairMatrices,
    sources: Mapping[str, str],
    built_with: Mapping[str, object],
) -> None:
    """Write the vectors into vectors_dir.partial with a manifest, then rename it to vectors_dir.

    The manifest holds the sources load_isg_vectors compares and built_with, the build run
    and the NLTK data, which are only recorded.
    """
    partial_dir = _check_new_folder(vectors_dir)
    partial_dir.mkdir(parents=True)
    (partial_dir / COLUMNS_FILE).write_text(
        json.dumps(isg_vectors.columns),
        encoding="utf-8",
    )
    for name, split in isg_vectors.splits.items():
        pairs = {}
        for column in PAIRS_COLUMNS:
            pairs[column] = split.pairs_df[column].tolist()
        (partial_dir / f"{name}_pairs.json").write_text(
            json.dumps(pairs),
            encoding="utf-8",
        )
        sp.save_npz(partial_dir / f"{name}_1.npz", split.first)
        sp.save_npz(partial_dir / f"{name}_2.npz", split.second)

    files = {}
    for path in sorted(partial_dir.iterdir()):
        files[path.name] = _sha256(path)
    manifest = {
        "settings": isg_vectors.settings,
        "sources": dict(sources),
        "built_with": dict(built_with),
        "files": files,
    }
    (partial_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    partial_dir.rename(vectors_dir)


def build_isg_vectors(
    cleaned_paths: Mapping[str, Path],
    vectors_dir: Path,
    context: RunContext,
    *,
    workers: int,
    on_progress: Callable[..., None] | None = None,
) -> None:
    """Build the ISG vector of every text once and save the vectors in vectors_dir.

    The columns are the features of the training texts, in the order they were first seen;
    a feature that only validation or test texts have is dropped. The workers are new
    processes, so a script has to call this under if __name__ == "__main__".

    Args:
        cleaned_paths: Cleaned jsonl file of the train, validation and test split.
        vectors_dir: Absolute folder that does not exist yet.
        context: Output of collect_run_context, with the hashes of the cleaned files.
        workers: Number of worker processes that build the graphs.
        on_progress: Called after every batch with split, texts_done and texts_total,
            e.g. ProgressFile.update; it must not touch the vectors.

    Raises:
        RuntimeError: If Python was not started with PYTHONHASHSEED=0.
        LookupError: If the NLTK data of NLTK_RESOURCES is not installed.
        ValueError: If vectors_dir is relative, a split is missing or unknown, or the
            context has no hash of a cleaned file or the hash of another file.
        FileExistsError: If vectors_dir or vectors_dir.partial already exists.
    """
    # all checks first - the graphs take hours on the full data
    _check_hash_seed()
    _check_new_folder(vectors_dir)
    if set(cleaned_paths) != set(SPLITS):
        raise ValueError(f"The splits have to be {list(SPLITS)}.")
    data_names = []
    for name in SPLITS:
        data_names.append(cleaned_paths[name].name)
    sources = _sources(context, data_names)
    _check_data_files(cleaned_paths, sources)
    built_with = {"context": context.environment, "nltk_data": _nltk_data_sha256()}

    columns: dict[str, int] = {}
    splits = {}
    # spawn: every worker is a new Python process that inherits PYTHONHASHSEED from this one
    mp_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp_context,
        max_tasks_per_child=CHUNKS_PER_WORKER,
    ) as executor:
        # SPLITS starts with train, so the training texts decide the columns
        for name in SPLITS:
            splits[name] = _build_split(
                executor,
                name,
                cleaned_paths[name],
                columns,
                on_progress,
            )

    isg_vectors = PairMatrices(
        columns=list(columns),
        settings=dict(SETTINGS),
        splits=splits,
    )
    _write_isg_vectors(vectors_dir, isg_vectors, sources, built_with)


def load_isg_vectors(vectors_dir: Path, context: RunContext) -> PairMatrices:
    """Load the vectors of build_isg_vectors, only if they were built by the same code on the same data.

    Args:
        vectors_dir: Folder written by build_isg_vectors.
        context: Output of collect_run_context, with the hashes of the cleaned files.

    Returns:
        The vectors of both texts of every pair of the three splits.

    Raises:
        ValueError: If the settings, graph.py, uv.lock, a cleaned file or a saved file
            differ from the ones the vectors were built with, or a side does not have
            one row per pair and one column per feature.
    """
    manifest = json.loads((vectors_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    if manifest["settings"] != SETTINGS:
        raise ValueError(
            f"The vectors were built with {manifest['settings']}, not {SETTINGS}.",
        )
    saved_sources = manifest["sources"]
    data_names = []
    for key in saved_sources:
        if key.startswith("data_sha256."):
            data_names.append(key.removeprefix("data_sha256."))
    sources = _sources(context, data_names)
    for key, value in saved_sources.items():
        if sources[key] != value:
            raise ValueError(f"{key} differs from the one the vectors were built with.")
    for file_name, value in manifest["files"].items():
        if _sha256(vectors_dir / file_name) != value:
            raise ValueError(f"{file_name} was changed after it was written.")

    columns = json.loads((vectors_dir / COLUMNS_FILE).read_text(encoding="utf-8"))
    splits = {}
    for name in SPLITS:
        pairs = json.loads(
            (vectors_dir / f"{name}_pairs.json").read_text(encoding="utf-8"),
        )
        split = SplitMatrices(
            pairs_df=pd.DataFrame(pairs, columns=PAIRS_COLUMNS),
            first=sp.load_npz(vectors_dir / f"{name}_1.npz").tocsr(),
            second=sp.load_npz(vectors_dir / f"{name}_2.npz").tocsr(),
        )
        expected_shape = (len(split.pairs_df), len(columns))
        for side in (split.first, split.second):
            if side.shape != expected_shape:
                raise ValueError(
                    f"{name} has a side of shape {side.shape}, not {expected_shape}.",
                )
        splits[name] = split
    return PairMatrices(columns=columns, settings=manifest["settings"], splits=splits)


def build_isg_features(
    isg_vectors: PairMatrices,
    fit_splits: Sequence[str],
    target_split: str,
) -> PairFeatures:
    """Return the ISG vectors of the fit pairs and of the target pairs.

    The columns stay those of the training texts for every combination of splits.

    Args:
        isg_vectors: Output of load_isg_vectors.
        fit_splits: ["train"] for tuning and the held-out check, ["train", "validation"]
            for the final model.
        target_split: "validation" for tuning, "test" otherwise.

    Returns:
        The vectors of the fit and of the target pairs; vectorizer is None. The target
        vectors are the loaded matrices themselves, so they must not be changed in place.

    Raises:
        ValueError: If the splits are not the tuning, the final or the held-out combination.
    """
    _check_splits(fit_splits, target_split)
    fit = [isg_vectors.splits[name] for name in fit_splits]
    target = isg_vectors.splits[target_split]
    fit_sides = (
        sp.vstack([split.first for split in fit], format="csr"),
        sp.vstack([split.second for split in fit], format="csr"),
    )
    # copies of their own, so a column a caller adds never reaches the loaded pairs
    fit_pairs_df = pd.concat(
        [split.pairs_df for split in fit],
        ignore_index=True,
    ).copy()
    return PairFeatures(
        fit_sides=fit_sides,
        target_sides=(target.first, target.second),
        fit_pairs_df=fit_pairs_df,
        target_pairs_df=target.pairs_df.copy(),
        vectorizer=None,
    )
