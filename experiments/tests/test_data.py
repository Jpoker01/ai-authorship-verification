import json
from pathlib import Path

import pandas as pd
import pytest

from avcore.data import (
    PAIR_COLUMNS,
    TEST_CLEANED_FILE,
    TEST_PAIRS_FILE,
    TEST_TRUTH_FILE,
    TRAIN_CLEANED_FILE,
    TRAIN_PAIRS_FILE,
    TRAIN_TRUTH_FILE,
    VALIDATION_CLEANED_FILE,
    DatasetCounts,
    build_cleaned_datasets,
    clean_text,
    load_pan_pairs,
    read_jsonl,
    split_train_validation,
    write_pairs_jsonl,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Writes one JSON object per line, utf-8, like the PAN files."""
    lines = [json.dumps(record, ensure_ascii=False) + "\n" for record in records]
    path.write_text("".join(lines), encoding="utf-8")


def _write_raw_pan(
    pairs_path: Path, truth_path: Path, n_pairs: int, id_prefix: str = "p"
) -> None:
    """Writes a synthetic raw PAN pairs file and its truth file, with the extra columns.

    Pair i has the id f"{id_prefix}{i}", the texts "<b>t</b>{i}a" and "t_{i}b’"
    (cleaned "t{i}a" and "t{i}b") and is a same-author pair when i is even.
    """
    pairs = [
        {
            "id": f"{id_prefix}{i}",
            "fandoms": ["f1", "f2"],
            "pair": [f"<b>t</b>{i}a", f"t_{i}b’"],
        }
        for i in range(n_pairs)
    ]
    truths = [
        {"id": f"{id_prefix}{i}", "same": i % 2 == 0, "authors": ["a1", "a2"]}
        for i in range(n_pairs)
    ]
    _write_jsonl(pairs_path, pairs)
    _write_jsonl(truth_path, truths)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<b>bold</b> text", "bold text"),
        ("&amp; don&#39;t", " don't"),
        ("a [note] b", "a  b"),
        ("[[x]] y", " y"),
        *[(f"x{char}x", "xx") for char in "[\\]^_`"],
        ("Ab 9,\t'\n  \x1c", "Ab 9,\t'\n  \x1c"),
        *[(f"x{char}x", "xx") for char in '.!?;:-"()'],
        ("café", "caf"),
    ],
)
def test_clean_text(text: str, expected: str) -> None:
    """Removes HTML and [...] blocks and keeps only letters, digits, whitespace, comma and apostrophe."""
    assert clean_text(text) == expected


def test_read_jsonl_reads_utf8(tmp_path: Path) -> None:
    """Non-ASCII text is decoded as utf-8, whatever the default encoding of the OS is."""
    path = tmp_path / "pairs.jsonl"
    _write_jsonl(path, [{"id": "a", "pair": ["café", "naïve"], "same": True}])
    pairs_df = read_jsonl(path)
    assert pairs_df.to_dict(orient="records") == [
        {"id": "a", "pair": ["café", "naïve"], "same": True}
    ]


def test_load_pan_pairs_keeps_raw_texts_and_matches_labels_by_line(
    tmp_path: Path,
) -> None:
    """Without cleaning, the texts stay raw, only id / pair / same are kept and line i gets the label of truth line i."""
    pairs_path = tmp_path / "pairs.jsonl"
    truth_path = tmp_path / "truth.jsonl"
    _write_raw_pan(pairs_path, truth_path, n_pairs=10)
    pairs_df = load_pan_pairs(pairs_path, truth_path)
    assert list(pairs_df.columns) == PAIR_COLUMNS
    assert pairs_df["id"].tolist() == [f"p{i}" for i in range(10)]
    assert pairs_df["same"].tolist() == [i % 2 == 0 for i in range(10)]
    assert pairs_df["pair"][3] == ["<b>t</b>3a", "t_3b’"]


def test_load_pan_pairs_repeated_id_keeps_one_row_per_line(tmp_path: Path) -> None:
    """An id that repeats gives one row per line with that line's label, never a cross product."""
    pairs_path = tmp_path / "pairs.jsonl"
    truth_path = tmp_path / "truth.jsonl"
    _write_jsonl(
        pairs_path,
        [
            {"id": "a", "pair": ["a1", "a2"]},
            {"id": "b", "pair": ["b1", "b2"]},
            {"id": "a", "pair": ["a3", "a4"]},
        ],
    )
    _write_jsonl(
        truth_path,
        [
            {"id": "a", "same": True},
            {"id": "b", "same": False},
            {"id": "a", "same": False},
        ],
    )
    pairs_df = load_pan_pairs(pairs_path, truth_path)
    assert pairs_df.to_dict(orient="records") == [
        {"id": "a", "pair": ["a1", "a2"], "same": True},
        {"id": "b", "pair": ["b1", "b2"], "same": False},
        {"id": "a", "pair": ["a3", "a4"], "same": False},
    ]


def test_load_pan_pairs_cleans_both_texts(tmp_path: Path) -> None:
    """With clean_texts=True, BOTH texts of every pair go through clean_text."""
    pairs_path = tmp_path / "pairs.jsonl"
    truth_path = tmp_path / "truth.jsonl"
    _write_raw_pan(pairs_path, truth_path, n_pairs=10)
    pairs_df = load_pan_pairs(pairs_path, truth_path, clean_texts=True)
    assert pairs_df["pair"].tolist() == [[f"t{i}a", f"t{i}b"] for i in range(10)]


@pytest.mark.parametrize(
    ("truth_records", "error_message"),
    [
        (
            [{"id": "p0", "same": True}, {"id": "WRONG", "same": False}],
            "line 2: id p1 does not match truth id WRONG",
        ),
        ([{"id": "p0", "same": True}], "shorter than argument 1"),
    ],
)
def test_load_pan_pairs_rejects_misaligned_truth(
    tmp_path: Path,
    truth_records: list[dict],
    error_message: str,
) -> None:
    """A truth line with another id, or a truth file with fewer lines, stops the load."""
    pairs_path = tmp_path / "pairs.jsonl"
    truth_path = tmp_path / "truth.jsonl"
    _write_raw_pan(pairs_path, truth_path, n_pairs=10)
    _write_jsonl(truth_path, truth_records)
    with pytest.raises(ValueError, match=error_message):
        load_pan_pairs(pairs_path, truth_path)


@pytest.mark.parametrize("texts", [["a", "b", "c"], ["a"], "ab"])
def test_load_pan_pairs_rejects_three_texts(
    tmp_path: Path, texts: list[str] | str
) -> None:
    """3 texts, 1 text or a plain string instead of a list stops the load with its line number."""
    pairs_path = tmp_path / "pairs.jsonl"
    truth_path = tmp_path / "truth.jsonl"
    _write_jsonl(pairs_path, [{"id": "p0", "pair": texts}])
    _write_jsonl(truth_path, [{"id": "p0", "same": True}])
    with pytest.raises(ValueError, match="line 1: pair is not a list of 2 texts"):
        load_pan_pairs(pairs_path, truth_path)


def test_load_pan_pairs_workers_give_the_same_result(tmp_path: Path) -> None:
    """Cleaning in 2 worker processes gives the same frame as cleaning in this process."""
    pairs_path = tmp_path / "pairs.jsonl"
    truth_path = tmp_path / "truth.jsonl"
    _write_raw_pan(pairs_path, truth_path, n_pairs=2500)
    serial_df = load_pan_pairs(pairs_path, truth_path, clean_texts=True)
    parallel_df = load_pan_pairs(pairs_path, truth_path, clean_texts=True, workers=2)
    pd.testing.assert_frame_equal(serial_df, parallel_df)


def test_split_train_validation_pinned_ids() -> None:
    """The seeded split moves exactly these ids to validation and keeps the shuffled order."""
    pairs_df = pd.DataFrame(
        {
            "id": [f"p{i}" for i in range(10)],
            "pair": [[f"a{i}", f"b{i}"] for i in range(10)],
            "same": [i % 2 == 0 for i in range(10)],
        }
    )
    split = split_train_validation(pairs_df, validation_size=3, random_state=42)
    assert split.train_df["id"].tolist() == ["p0", "p7", "p2", "p9", "p4", "p3", "p6"]
    assert split.validation_df["id"].tolist() == ["p8", "p1", "p5"]


def test_write_pairs_jsonl_exact_bytes(tmp_path: Path) -> None:
    """One utf-8 JSON object per line in the order id, pair, same, ending in a single newline."""
    path = tmp_path / "pairs.jsonl"
    # columns in another order and an extra column, to check the selection and its order
    pairs_df = pd.DataFrame(
        {
            "same": [True, False],
            "extra": [1, 2],
            "id": ["a", "b"],
            "pair": [["é", "x"], ["y", "z"]],
        }
    )
    write_pairs_jsonl(pairs_df, path)
    assert path.read_bytes() == (
        b'{"id": "a", "pair": ["\xc3\xa9", "x"], "same": true}\n'
        b'{"id": "b", "pair": ["y", "z"], "same": false}\n'
    )


def test_write_pairs_jsonl_refuses_existing_file(tmp_path: Path) -> None:
    """An existing file is never overwritten, its content stays as it was."""
    path = tmp_path / "pairs.jsonl"
    path.write_text("old content\n", encoding="utf-8")
    pairs_df = pd.DataFrame({"id": ["a"], "pair": [["x", "y"]], "same": [True]})
    with pytest.raises(FileExistsError):
        write_pairs_jsonl(pairs_df, path)
    assert path.read_text(encoding="utf-8") == "old content\n"


def test_write_then_read_round_trip(tmp_path: Path) -> None:
    """A frame written by write_pairs_jsonl and read back by read_jsonl is unchanged."""
    path = tmp_path / "pairs.jsonl"
    pairs_df = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "pair": [["café", "x"], ["it's, ok", "y"], ["", "z"]],
            "same": [True, False, True],
        }
    )
    write_pairs_jsonl(pairs_df, path)
    pd.testing.assert_frame_equal(read_jsonl(path), pairs_df)


def test_build_cleaned_datasets_writes_the_three_files(tmp_path: Path) -> None:
    """Test and train are loaded cleaned, train is split, and the 3 files hold the right rows."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    # nested, so the missing parent folder is created too
    output_dir = tmp_path / "data" / "processed"
    _write_raw_pan(raw_dir / TRAIN_PAIRS_FILE, raw_dir / TRAIN_TRUTH_FILE, n_pairs=10)
    _write_raw_pan(
        raw_dir / TEST_PAIRS_FILE, raw_dir / TEST_TRUTH_FILE, n_pairs=4, id_prefix="t"
    )
    counts = build_cleaned_datasets(
        raw_dir,
        output_dir,
        validation_size=3,
        random_state=0,
        workers=2,
    )
    assert counts == DatasetCounts(train=7, validation=3, test=4)
    train_df = read_jsonl(output_dir / TRAIN_CLEANED_FILE)
    validation_df = read_jsonl(output_dir / VALIDATION_CLEANED_FILE)
    assert train_df["id"].tolist() == ["p9", "p1", "p6", "p7", "p3", "p0", "p5"]
    assert train_df["same"].tolist() == [False, False, True, False, False, True, False]
    assert train_df["pair"][0] == ["t9a", "t9b"]
    assert validation_df["id"].tolist() == ["p2", "p8", "p4"]
    assert validation_df["same"].tolist() == [True, True, True]
    # exact bytes pin the ids, the cleaned texts, the labels as true / false and the format
    assert (output_dir / TEST_CLEANED_FILE).read_bytes() == (
        b'{"id": "t0", "pair": ["t0a", "t0b"], "same": true}\n'
        b'{"id": "t1", "pair": ["t1a", "t1b"], "same": false}\n'
        b'{"id": "t2", "pair": ["t2a", "t2b"], "same": true}\n'
        b'{"id": "t3", "pair": ["t3a", "t3b"], "same": false}\n'
    )


@pytest.mark.parametrize(
    "existing_file", [TRAIN_CLEANED_FILE, VALIDATION_CLEANED_FILE, TEST_CLEANED_FILE]
)
def test_build_cleaned_datasets_refuses_before_writing(
    tmp_path: Path, existing_file: str
) -> None:
    """If any cleaned file exists, the build stops before it writes anything."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output_dir = tmp_path / "processed"
    output_dir.mkdir()
    _write_raw_pan(raw_dir / TRAIN_PAIRS_FILE, raw_dir / TRAIN_TRUTH_FILE, n_pairs=10)
    _write_raw_pan(
        raw_dir / TEST_PAIRS_FILE, raw_dir / TEST_TRUTH_FILE, n_pairs=4, id_prefix="t"
    )
    (output_dir / existing_file).write_text("old content\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        build_cleaned_datasets(raw_dir, output_dir, validation_size=3)
    assert sorted(path.name for path in output_dir.iterdir()) == [existing_file]


@pytest.mark.parametrize(
    "missing_file",
    [TRAIN_PAIRS_FILE, TRAIN_TRUTH_FILE, TEST_PAIRS_FILE, TEST_TRUTH_FILE],
)
def test_build_cleaned_datasets_needs_all_raw_files(
    tmp_path: Path, missing_file: str
) -> None:
    """A missing raw file stops the build before the output folder is even created."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output_dir = tmp_path / "processed"
    _write_raw_pan(raw_dir / TRAIN_PAIRS_FILE, raw_dir / TRAIN_TRUTH_FILE, n_pairs=10)
    _write_raw_pan(
        raw_dir / TEST_PAIRS_FILE, raw_dir / TEST_TRUTH_FILE, n_pairs=4, id_prefix="t"
    )
    (raw_dir / missing_file).unlink()
    with pytest.raises(FileNotFoundError, match=missing_file):
        build_cleaned_datasets(raw_dir, output_dir, validation_size=3)
    assert not output_dir.exists()
