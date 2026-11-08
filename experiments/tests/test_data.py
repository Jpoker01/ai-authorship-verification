import json
from pathlib import Path

import pytest

from avcore.data import clean_text


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Writes one JSON object per line, utf-8, like the PAN files."""
    lines = [json.dumps(record, ensure_ascii=False) + "\n" for record in records]
    path.write_text("".join(lines), encoding="utf-8")


def _write_raw_pan(
    pairs_path: Path, truth_path: Path, n_pairs: int, id_prefix: str = "p"
) -> None:
    """Writes a synthetic raw PAN pairs file and its truth file, with the extra columns.

    Pair i has the id f"{id_prefix}{i}", the texts "<b>t</b>{i}a" and "t_{i}b"
    (cleaned "t{i}a" and "t{i}b") and is a same-author pair when i is even.
    """
    pairs = [
        {
            "id": f"{id_prefix}{i}",
            "fandoms": ["f1", "f2"],
            "pair": [f"<b>t</b>{i}a", f"t_{i}b"],
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
