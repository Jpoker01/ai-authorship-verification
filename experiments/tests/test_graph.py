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


@pytest.mark.cerit
def test_load_isg_keeps_tok2vec_and_the_text_length_limit_of_text2graphapi() -> None:
    """The replaced pipeline has tok2vec and accepts texts over spaCy's default 1 000 000 characters."""
    spacy_default_limit = 1_000_000

    isg = _load_isg()

    assert "tok2vec" in isg.prep.nlp.pipe_names
    assert isg.prep.nlp.max_length > spacy_default_limit


@needs_random_hash_seed
def test_extract_isg_features_refuses_a_random_hash_seed() -> None:
    """Without PYTHONHASHSEED=0 the synonyms would change from run to run."""
    text = "The dog walked."

    with pytest.raises(RuntimeError, match="PYTHONHASHSEED=0"):
        extract_isg_features(text)


def test_iter_isg_features_keeps_the_order_of_the_texts_across_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Texts sent in several rounds come back in their own order, each text once."""
    texts = []
    for index in range(10):
        texts.append(f"text{index}")
    monkeypatch.setattr(graph, "BATCH_SIZE", 3)
    monkeypatch.setattr(graph, "CHUNK_SIZE", 2)
    monkeypatch.setattr(graph, "extract_isg_features", lambda text: Counter([text]))

    with ThreadPoolExecutor(max_workers=2) as executor:
        all_features = list(_iter_isg_features(executor, texts))

    assert all_features == [Counter([text]) for text in texts]


def test_iter_isg_features_reads_only_one_batch_ahead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first result comes after BATCH_SIZE texts are read, not after all of them."""
    texts_read = []

    def read_texts() -> Iterator[str]:
        for index in range(10):
            texts_read.append(index)
            yield f"text{index}"

    monkeypatch.setattr(graph, "BATCH_SIZE", 3)
    monkeypatch.setattr(graph, "extract_isg_features", lambda text: Counter([text]))

    with ThreadPoolExecutor(max_workers=2) as executor:
        all_features = _iter_isg_features(executor, read_texts())
        next(all_features)

    assert len(texts_read) == 3


def test_to_matrix_numbers_the_columns_as_the_features_first_appear() -> None:
    """A new feature gets the next free column, a known one keeps its column."""
    all_features = iter([{"b": 2, "a": 1}, {"c": 3, "a": 4}])
    columns: dict[str, int] = {}
    expected = np.array([[2, 1, 0], [0, 4, 3]], dtype=np.float32)

    matrix = _to_matrix(all_features, columns, add_columns=True)

    assert columns == {"b": 0, "a": 1, "c": 2}
    assert matrix.dtype == np.float32
    assert matrix.has_sorted_indices
    assert np.array_equal(matrix.toarray(), expected)


def test_to_matrix_drops_features_without_a_column() -> None:
    """A feature that is not a column yet is dropped and adds no column."""
    all_features = iter([{"a": 1, "new": 5}])
    columns = {"a": 0}
    expected = np.array([[1]], dtype=np.float32)

    matrix = _to_matrix(all_features, columns, add_columns=False)

    assert columns == {"a": 0}
    assert np.array_equal(matrix.toarray(), expected)


def test_text_file_sha256_ignores_line_endings_but_not_edits(tmp_path: Path) -> None:
    """A Windows (CRLF) checkout hashes like a Linux one; a changed line does not."""
    linux_file = tmp_path / "linux.py"
    windows_file = tmp_path / "windows.py"
    edited_file = tmp_path / "edited.py"
    linux_file.write_bytes(b"x = 1\ny = 2\n")
    windows_file.write_bytes(b"x = 1\r\ny = 2\r\n")
    edited_file.write_bytes(b"x = 1\ny = 3\n")

    assert _text_file_sha256(windows_file) == _text_file_sha256(linux_file)
    assert _text_file_sha256(edited_file) != _text_file_sha256(linux_file)


def test_manifest_names_every_saved_file(tmp_path: Path) -> None:
    """Every file of the folder but the manifest has its hash in the manifest."""
    vectors_dir = tmp_path / "isg"
    isg_vectors = _isg_vectors()
    sources = _sources(CONTEXT)

    _write_isg_vectors(vectors_dir, isg_vectors, sources, BUILT_WITH)
    manifest_text = (vectors_dir / graph.MANIFEST_FILE).read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)

    saved_files = {path.name for path in vectors_dir.iterdir()}
    saved_files.discard(graph.MANIFEST_FILE)
    assert set(manifest["files"]) == saved_files


@needs_random_hash_seed
def test_build_isg_vectors_refuses_a_random_hash_seed(tmp_path: Path) -> None:
    """The hash seed is checked before any text is read."""
    cleaned_paths = _cleaned_paths(tmp_path)
    vectors_dir = tmp_path / "isg"
    workers = 1

    with pytest.raises(RuntimeError, match="PYTHONHASHSEED=0"):
        build_isg_vectors(cleaned_paths, vectors_dir, CONTEXT, workers=workers)


@pytest.mark.parametrize(
    ("case", "error", "message"),
    [
        ("relative folder", ValueError, "absolute"),
        ("folder exists", FileExistsError, "isg"),
        ("partial folder exists", FileExistsError, "isg.partial"),
        ("split missing", ValueError, "splits have to be"),
        ("no data hash", ValueError, "has no data_sha256"),
    ],
)
def test_build_isg_vectors_checks_everything_before_building(
    case: str,
    error: type[Exception],
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each wrong input stops the run before a text is read (the files do not even exist)."""
    monkeypatch.setattr(graph, "_check_hash_seed", lambda: None)
    cleaned_paths = _cleaned_paths(tmp_path)
    vectors_dir = tmp_path / "isg"
    context = CONTEXT
    workers = 1
    if case == "relative folder":
        vectors_dir = Path("isg")
    elif case == "folder exists":
        vectors_dir.mkdir()
    elif case == "partial folder exists":
        (tmp_path / "isg.partial").mkdir()
    elif case == "split missing":
        del cleaned_paths["test"]
    elif case == "no data hash":
        context = _context({"train.jsonl": "0" * 64})

    with pytest.raises(error, match=message):
        build_isg_vectors(cleaned_paths, vectors_dir, context, workers=workers)


def test_build_isg_vectors_refuses_a_file_the_context_did_not_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cleaned file with the right name but other content stops the run before any graph."""
    monkeypatch.setattr(graph, "_check_hash_seed", lambda: None)
    pairs_by_split = {}
    for name in SPLITS:
        pairs_by_split[name] = _pairs([("A text.", "B text.")], name)
    cleaned_paths = _write_splits(pairs_by_split, tmp_path)
    context = _context_of(cleaned_paths)
    cleaned_paths["test"].unlink()
    other_test_pairs = _pairs([("Another text.", "B text.")], "test")
    write_pairs_jsonl(other_test_pairs, cleaned_paths["test"])
    vectors_dir = tmp_path / "isg"
    workers = 1

    with pytest.raises(ValueError, match="is not the file the run context hashed"):
        build_isg_vectors(cleaned_paths, vectors_dir, context, workers=workers)
    assert not vectors_dir.exists()


@pytest.mark.usefixtures("fake_build")
def test_build_isg_vectors_numbers_the_columns_from_the_training_texts(
    small_paths: dict[str, Path],
    tmp_path: Path,
) -> None:
    """Columns follow the first appearance in the training texts; other splits use them."""
    vectors_dir = tmp_path / "isg"
    context = _context_of(small_paths)
    workers = 2
    expected_columns = ["LEX::a", "LEX::b", "LEX::c", "LEX::seen"]
    expected_test_first = np.array([[0, 0, 1, 0]], dtype=np.float32)
    expected_test_second = np.array([[0, 1, 0, 0]], dtype=np.float32)

    build_isg_vectors(small_paths, vectors_dir, context, workers=workers)
    vectors = load_isg_vectors(vectors_dir, context)

    assert vectors.columns == expected_columns
    test = vectors.splits["test"]
    assert test.pairs_df["id"].tolist() == ["test0"]
    assert np.array_equal(test.first.toarray(), expected_test_first)
    assert np.array_equal(test.second.toarray(), expected_test_second)


@pytest.mark.usefixtures("fake_build")
def test_build_isg_vectors_reports_progress_after_every_batch(
    small_paths: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """on_progress gets the split, the texts done and the texts of the split after every batch."""
    vectors_dir = tmp_path / "isg"
    context = _context_of(small_paths)
    workers = 1
    calls = []
    monkeypatch.setattr(graph, "BATCH_SIZE", 3)
    expected_calls = [
        {"split": "train", "texts_done": 3, "texts_total": 4},
        {"split": "train", "texts_done": 4, "texts_total": 4},
        {"split": "validation", "texts_done": 2, "texts_total": 2},
        {"split": "test", "texts_done": 2, "texts_total": 2},
    ]

    def record_progress(**fields: object) -> None:
        calls.append(fields)

    build_isg_vectors(
        small_paths,
        vectors_dir,
        context,
        workers=workers,
        on_progress=record_progress,
    )

    assert calls == expected_calls


@pytest.mark.usefixtures("fake_build")
def test_build_isg_vectors_records_the_build_run_and_the_nltk_data(
    small_paths: dict[str, Path],
    tmp_path: Path,
) -> None:
    """The manifest keeps the context of the build and the hashes of the NLTK data."""
    vectors_dir = tmp_path / "isg"
    context = _context_of(small_paths)
    workers = 1
    expected_built_with = {
        "context": context.environment,
        "nltk_data": {"corpora/wordnet.zip": "d" * 64},
    }

    build_isg_vectors(small_paths, vectors_dir, context, workers=workers)
    manifest_text = (vectors_dir / graph.MANIFEST_FILE).read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)

    assert manifest["built_with"] == expected_built_with


@pytest.mark.cerit
@needs_hash_seed_0
def test_build_isg_vectors_tags_and_parses_the_words(
    real_text_paths: dict[str, Path],
    tmp_path: Path,
) -> None:
    """The tagger, the parser and the lemmatizer work: "walked" is the verb "walk" with a subject."""
    vectors_dir = tmp_path / "isg"
    context = _context_of(real_text_paths)
    workers = 1

    build_isg_vectors(real_text_paths, vectors_dir, context, workers=workers)
    vectors = load_isg_vectors(vectors_dir, context)

    columns = set(vectors.columns)
    assert {"LEX::walk", "POS::VERB", "POS::ADJ", "DEP::nsubj"} <= columns
    assert "LEX::walked" not in columns
    assert "LEX::the" not in columns


@pytest.mark.cerit
@needs_hash_seed_0
def test_build_isg_vectors_keeps_the_features_of_the_training_texts(
    real_text_paths: dict[str, Path],
    tmp_path: Path,
) -> None:
    """Every split uses the training columns; row i holds pair i of its split."""
    vectors_dir = tmp_path / "isg"
    context = _context_of(real_text_paths)
    workers = 1

    build_isg_vectors(real_text_paths, vectors_dir, context, workers=workers)
    vectors = load_isg_vectors(vectors_dir, context)

    assert "LEX::zeppelin" not in vectors.columns
    walk = vectors.columns.index("LEX::walk")
    nsubj = vectors.columns.index("DEP::nsubj")
    train = vectors.splits["train"]
    test = vectors.splits["test"]
    assert train.pairs_df["id"].tolist() == ["train0", "train1"]
    assert train.first[0, walk] == 1
    assert train.second[1, walk] == 1
    assert train.second[0, walk] == 0
    assert test.pairs_df["id"].tolist() == ["test0"]
    assert test.first[0, walk] == 1
    assert test.first[0, nsubj] > 0
    assert test.second[0, walk] == 0
    for name in SPLITS:
        split = vectors.splits[name]
        expected_shape = (len(split.pairs_df), len(vectors.columns))
        assert split.first.shape == expected_shape
        assert split.second.shape == expected_shape
    assert vectors.settings == graph.SETTINGS


@pytest.mark.cerit
@needs_hash_seed_0
def test_build_isg_vectors_gives_the_same_vectors_in_every_run(
    real_text_paths: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two workers with one text per chunk give what one worker with large chunks gives."""
    first_dir = tmp_path / "isg_1"
    second_dir = tmp_path / "isg_2"
    context = _context_of(real_text_paths)
    two_workers = 2
    one_worker = 1

    monkeypatch.setattr(graph, "CHUNK_SIZE", 1)
    monkeypatch.setattr(graph, "BATCH_SIZE", 3)
    build_isg_vectors(real_text_paths, first_dir, context, workers=two_workers)
    monkeypatch.undo()
    build_isg_vectors(real_text_paths, second_dir, context, workers=one_worker)
    first_run = load_isg_vectors(first_dir, context)
    second_run = load_isg_vectors(second_dir, context)

    assert first_run.columns == second_run.columns
    for name in SPLITS:
        first_split = first_run.splits[name]
        second_split = second_run.splits[name]
        assert (first_split.first != second_split.first).nnz == 0
        assert (first_split.second != second_split.second).nnz == 0


@pytest.mark.cerit
@needs_hash_seed_0
def test_build_isg_vectors_keeps_only_the_root_of_a_text_without_words(
    real_text_paths: dict[str, Path],
    tmp_path: Path,
) -> None:
    """An empty text and a text of stop words and digits only get the root node and nothing else."""
    empty_text = ""
    stop_words_only = "The and of 123 it is was"
    train_pairs = _pairs([(empty_text, stop_words_only)], "train")
    real_text_paths["train"].unlink()
    write_pairs_jsonl(train_pairs, real_text_paths["train"])
    vectors_dir = tmp_path / "isg"
    context = _context_of(real_text_paths)
    workers = 1
    expected = {"LEX::root": 1.0, "POS::0": 1.0}

    build_isg_vectors(real_text_paths, vectors_dir, context, workers=workers)
    vectors = load_isg_vectors(vectors_dir, context)
    train = vectors.splits["train"]

    assert _row_features(vectors.columns, train.first[0]) == expected
    assert _row_features(vectors.columns, train.second[0]) == expected


def test_load_isg_vectors_returns_the_written_vectors(tmp_path: Path) -> None:
    """Columns, settings, ids (as strings), labels and both sides of every split come back."""
    isg_vectors = _isg_vectors()
    vectors_dir = tmp_path / "isg"
    sources = _sources(CONTEXT)

    _write_isg_vectors(vectors_dir, isg_vectors, sources, BUILT_WITH)
    loaded = load_isg_vectors(vectors_dir, CONTEXT)

    assert loaded.columns == isg_vectors.columns
    assert loaded.settings == isg_vectors.settings
    assert not (tmp_path / "isg.partial").exists()
    for name in SPLITS:
        expected = isg_vectors.splits[name]
        split = loaded.splits[name]
        pd.testing.assert_frame_equal(split.pairs_df, expected.pairs_df)
        assert split.first.format == "csr"
        assert split.first.dtype == np.float32
        assert (split.first != expected.first).nnz == 0
        assert (split.second != expected.second).nnz == 0


def test_load_isg_vectors_accepts_a_run_at_another_commit(tmp_path: Path) -> None:
    """Only graph.py, uv.lock and the data files matter, not the commit of the run."""
    vectors_dir = tmp_path / "isg"
    isg_vectors = _isg_vectors()
    sources = _sources(CONTEXT)
    _write_isg_vectors(vectors_dir, isg_vectors, sources, BUILT_WITH)
    later_context = RunContext(
        environment={**CONTEXT.environment, "commit": "c" * 40},
        packages={},
    )

    loaded = load_isg_vectors(vectors_dir, later_context)

    assert loaded.columns == isg_vectors.columns


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("uv.lock", "uv_lock_sha256 differs"),
        ("data file", "data_sha256.test.jsonl differs"),
        ("data hash missing", "has no data_sha256.test.jsonl"),
        ("graph.py", "graph_py_sha256 differs"),
        ("settings", "were built with"),
        ("saved file", "was changed after it was written"),
    ],
)
def test_load_isg_vectors_refuses_vectors_of_other_sources(
    change: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vectors built with another graph.py, lock, data or settings, or edited later, are not loaded."""
    vectors_dir = tmp_path / "isg"
    isg_vectors = _isg_vectors()
    sources = _sources(CONTEXT)
    _write_isg_vectors(vectors_dir, isg_vectors, sources, BUILT_WITH)
    environment = dict(CONTEXT.environment)
    if change == "uv.lock":
        environment["uv_lock_sha256"] = "c" * 64
    elif change == "data file":
        environment["data_sha256.test.jsonl"] = "c" * 64
    elif change == "data hash missing":
        del environment["data_sha256.test.jsonl"]
    elif change == "graph.py":
        monkeypatch.setattr(graph, "_graph_module_sha256", lambda: "c" * 64)
    elif change == "settings":
        other_settings = {**graph.SETTINGS, "apply_prep": False}
        monkeypatch.setattr(graph, "SETTINGS", other_settings)
    elif change == "saved file":
        other_vectors = sp.csr_matrix(np.zeros((1, 3), np.float32))
        sp.save_npz(vectors_dir / "test_1.npz", other_vectors)
    context = RunContext(environment=environment, packages={})

    with pytest.raises(ValueError, match=message):
        load_isg_vectors(vectors_dir, context)


def test_load_isg_vectors_refuses_a_side_of_another_shape(tmp_path: Path) -> None:
    """A side without one row per pair and one column per feature is not loaded."""
    vectors_dir = tmp_path / "isg"
    isg_vectors = _isg_vectors()
    test = isg_vectors.splits["test"]
    narrow_first = test.first[:, :2]
    isg_vectors.splits["test"] = SplitMatrices(
        pairs_df=test.pairs_df,
        first=narrow_first,
        second=test.second,
    )
    sources = _sources(CONTEXT)
    _write_isg_vectors(vectors_dir, isg_vectors, sources, BUILT_WITH)

    with pytest.raises(ValueError, match="test has a side of shape"):
        load_isg_vectors(vectors_dir, CONTEXT)


@pytest.mark.parametrize(("fit_splits", "target_split"), [TUNING, FINAL, HELD_OUT])
def test_build_isg_features_returns_the_vectors_of_the_fit_and_target_pairs(
    fit_splits: list[str],
    target_split: str,
) -> None:
    """Fit rows are the fit splits in order (train first), the target is its own split."""
    isg_vectors = _isg_vectors()
    fit = [isg_vectors.splits[name] for name in fit_splits]
    target = isg_vectors.splits[target_split]
    expected_first = np.vstack([split.first.toarray() for split in fit])
    expected_second = np.vstack([split.second.toarray() for split in fit])
    expected_fit_ids = []
    for split in fit:
        expected_fit_ids.extend(split.pairs_df["id"])

    features = build_isg_features(isg_vectors, fit_splits, target_split)

    first_fit, second_fit = features.fit_sides
    assert np.array_equal(first_fit.toarray(), expected_first)
    assert np.array_equal(second_fit.toarray(), expected_second)
    assert features.fit_pairs_df["id"].tolist() == expected_fit_ids
    assert features.target_sides[0] is target.first
    assert features.target_sides[1] is target.second
    pd.testing.assert_frame_equal(features.target_pairs_df, target.pairs_df)
    assert features.vectorizer is None


def test_build_isg_features_returns_labels_the_caller_may_edit() -> None:
    """Editing the returned labels never changes the labels of the loaded vectors."""
    isg_vectors = _isg_vectors()
    fit_splits, target_split = TUNING
    expected_train = isg_vectors.splits["train"].pairs_df.copy()
    expected_validation = isg_vectors.splits["validation"].pairs_df.copy()

    features = build_isg_features(isg_vectors, fit_splits, target_split)
    features.fit_pairs_df.loc[0, "same"] = False
    features.target_pairs_df.loc[0, "same"] = False

    pd.testing.assert_frame_equal(isg_vectors.splits["train"].pairs_df, expected_train)
    pd.testing.assert_frame_equal(
        isg_vectors.splits["validation"].pairs_df,
        expected_validation,
    )


@pytest.mark.parametrize(
    ("fit_splits", "target_split"),
    [(["train", "test"], "validation"), (["validation"], "test"), (["train"], "train")],
)
def test_build_isg_features_rejects_other_split_combinations(
    fit_splits: list[str],
    target_split: str,
) -> None:
    """Only the tuning, the final and the held-out combination are allowed."""
    isg_vectors = _isg_vectors()

    with pytest.raises(ValueError, match="is not allowed"):
        build_isg_features(isg_vectors, fit_splits, target_split)
