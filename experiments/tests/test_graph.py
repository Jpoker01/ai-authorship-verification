import json
import sys
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import Executor, ThreadPoolExecutor
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from avcore import graph
from avcore.data import write_pairs_jsonl
from avcore.features import SPLITS, PairMatrices, SplitMatrices
from avcore.graph import (
    _iter_isg_features,
    _load_isg,
    _text_file_sha256,
    _to_matrix,
    _write_isg_vectors,
    build_isg_features,
    build_isg_vectors,
    extract_features_from_isg,
    extract_isg_features,
    load_isg_vectors,
    parse_graph_dependency,
    parse_graph_node,
)
from avcore.tracking import RunContext, _sha256

TUNING = (["train"], "validation")
FINAL = (["train", "validation"], "test")
HELD_OUT = (["train"], "test")
# Python sets this flag at start: 0 for PYTHONHASHSEED=0, 1 for any other seed or none
STARTED_WITH_HASH_SEED_0 = sys.flags.hash_randomization == 0
needs_hash_seed_0 = pytest.mark.skipif(
    not STARTED_WITH_HASH_SEED_0,
    reason="needs Python started with PYTHONHASHSEED=0",
)
needs_random_hash_seed = pytest.mark.skipif(
    STARTED_WITH_HASH_SEED_0,
    reason="needs Python started without PYTHONHASHSEED=0",
)


def _context(data_hashes: dict[str, str]) -> RunContext:
    """A run context with a commit, a uv.lock hash and the given data hashes."""
    environment = {"commit": "a" * 40, "uv_lock_sha256": "b" * 64}
    for name, value in data_hashes.items():
        environment[f"data_sha256.{name}"] = value
    return RunContext(environment=environment, packages={})


def _context_of(cleaned_paths: dict[str, Path]) -> RunContext:
    """A run context that hashed the given cleaned files, as collect_run_context does."""
    data_hashes = {}
    for path in cleaned_paths.values():
        data_hashes[path.name] = _sha256(path)
    return _context(data_hashes)


CONTEXT = _context(
    {
        "train.jsonl": "0" * 64,
        "validation.jsonl": "1" * 64,
        "test.jsonl": "2" * 64,
    },
)
BUILT_WITH = {"context": CONTEXT.environment, "nltk_data": {}}


def _cleaned_paths(folder: Path) -> dict[str, Path]:
    """The path of the cleaned file of every split in folder, written or not."""
    paths = {}
    for name in SPLITS:
        paths[name] = folder / f"{name}.jsonl"
    return paths


def _pairs(texts: list[tuple[str, str]], prefix: str) -> pd.DataFrame:
    """Pairs with the ids f"{prefix}{i}"; pair i is a same-author pair when i is even."""
    records = []
    for index, (first_text, second_text) in enumerate(texts):
        records.append(
            {
                "id": f"{prefix}{index}",
                "pair": [first_text, second_text],
                "same": index % 2 == 0,
            },
        )
    return pd.DataFrame(records)


def _write_splits(
    pairs_by_split: dict[str, pd.DataFrame],
    folder: Path,
) -> dict[str, Path]:
    """Write every split as a cleaned jsonl file and return the paths by split."""
    paths = _cleaned_paths(folder)
    for name, pairs_df in pairs_by_split.items():
        write_pairs_jsonl(pairs_df, paths[name])
    return paths


def _isg_vectors() -> PairMatrices:
    """Hand-made vectors of 2 train, 1 validation and 1 test pair over 3 columns.

    Every value tells where it comes from - split number + 1 for a first text,
    split number + 10 for a second text.
    """
    splits = {}
    rows_by_split = {"train": 2, "validation": 1, "test": 1}
    for index, (name, n_pairs) in enumerate(rows_by_split.items()):
        ids = []
        labels = []
        for row in range(n_pairs):
            ids.append(f"00{index}{row}")
            labels.append(row % 2 == 0)
        pairs_df = pd.DataFrame({"id": ids, "same": labels})
        first = sp.csr_matrix(np.full((n_pairs, 3), index + 1, dtype=np.float32))
        second = sp.csr_matrix(np.full((n_pairs, 3), index + 10, dtype=np.float32))
        splits[name] = SplitMatrices(pairs_df=pairs_df, first=first, second=second)
    return PairMatrices(
        columns=["LEX::root", "POS::0", "LEX::walk"],
        settings=dict(graph.SETTINGS),
        splits=splits,
    )


def _sources(context: RunContext) -> dict[str, str]:
    """What the hand-made vectors depend on, read from the context."""
    data_names = []
    for name in SPLITS:
        data_names.append(f"{name}.jsonl")
    return graph._sources(context, data_names)


def _row_features(columns: list[str], row: sp.csr_matrix) -> dict[str, float]:
    """The feature names and values of one matrix row."""
    features = {}
    for column, value in zip(row.indices, row.data, strict=True):
        features[columns[column]] = value
    return features


def _word_counts(text: str) -> Counter[str]:
    """Features of a text without spaCy - every word counted as a lemma."""
    features: Counter[str] = Counter()
    for word in text.split():
        features[f"LEX::{word}"] += 1
    return features


@pytest.fixture
def fake_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """build_isg_vectors without spaCy: threads instead of processes, a word count as features."""

    def thread_pool(max_workers: int, mp_context: object) -> Executor:
        return ThreadPoolExecutor(max_workers)

    monkeypatch.setattr(graph, "ProcessPoolExecutor", thread_pool)
    monkeypatch.setattr(graph, "extract_isg_features", _word_counts)
    monkeypatch.setattr(graph, "_check_hash_seed", lambda: None)
    monkeypatch.setattr(
        graph,
        "_nltk_data_sha256",
        lambda: {"corpora/wordnet.zip": "d" * 64},
    )


@pytest.fixture
def small_paths(tmp_path: Path) -> dict[str, Path]:
    """Cleaned files: 2 train, 1 validation and 1 test pair; "seen" is a train word only."""
    pairs_by_split = {
        "train": _pairs([("a b", "b c"), ("c seen", "a")], "train"),
        "validation": _pairs([("a unseen", "b")], "validation"),
        "test": _pairs([("c", "unseen b")], "test"),
    }
    return _write_splits(pairs_by_split, tmp_path)


@pytest.fixture
def real_text_paths(tmp_path: Path) -> dict[str, Path]:
    """Cleaned files: 2 train, 1 validation and 1 test pair of short English texts."""
    pairs_by_split = {
        "train": _pairs(
            [
                ("The old man walked slowly to the market.", "She bought fresh bread."),
                ("A small dog barked at the stranger.", "They walked home together."),
            ],
            "train",
        ),
        "validation": _pairs(
            [("The pilot landed the zeppelin.", "Rain fell.")],
            "validation",
        ),
        "test": _pairs([("The dog walked.", "Bread was fresh.")], "test"),
    }
    return _write_splits(pairs_by_split, tmp_path)


@pytest.mark.parametrize(
    ("graph_node", "expected"),
    [
        ("walk_VERB", ("walk", "VERB")),
        ("New_York_PROPN", ("new_york", "PROPN")),
        ("ROOT_0", ("root", "0")),
        ("dog_", ("dog", "")),
    ],
)
def test_parse_graph_node(graph_node: str, expected: tuple[str, str]) -> None:
    """The text after the last underscore is the tag, the lemma is lowercased."""
    assert parse_graph_node(graph_node) == expected


def test_parse_graph_dependency_cuts_off_the_count() -> None:
    """The count text2graphapi appends to a relation is dropped."""
    edge_data = {"gramm_relation": "nsubj_12"}

    assert parse_graph_dependency(edge_data) == "nsubj"


def test_extract_features_from_isg_counts_lemmas_tags_and_relations() -> None:
    """A node counts once for its lemma and once for its tag, an edge once for its relation."""
    graph_of_text = nx.DiGraph()
    graph_of_text.add_edge("ROOT_0", "walk_VERB", gramm_relation="ROOT_1")
    graph_of_text.add_edge("walk_VERB", "dog_NOUN", gramm_relation="nsubj_3")
    graph_of_text.add_edge("walk_VERB", "Dog_", gramm_relation="")
    expected = Counter(
        {
            "LEX::root": 1,
            "LEX::walk": 1,
            "LEX::dog": 2,
            "POS::0": 1,
            "POS::VERB": 1,
            "POS::NOUN": 1,
            "DEP::ROOT": 1,
            "DEP::nsubj": 1,
        },
    )

    assert extract_features_from_isg(graph_of_text) == expected
