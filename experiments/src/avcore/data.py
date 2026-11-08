"""Loading, cleaning and the train / validation split of the PAN authorship verification pairs.

Builds the cleaned jsonl files that every notebook family except the LLMs reads.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

PAIR_COLUMNS = ["id", "pair", "same"]
VALIDATION_SIZE = 2_500
RANDOM_STATE = 42

TRAIN_PAIRS_FILE = "pan20-authorship-verification-training-large.jsonl"
TRAIN_TRUTH_FILE = "pan20-authorship-verification-training-large-truth.jsonl"

TEST_PAIRS_FILE = "pan21-authorship-verification-test.jsonl"
TEST_TRUTH_FILE = "pan21-authorship-verification-test-truth.jsonl"

TRAIN_CLEANED_FILE = "pan20-authorship-verification-training-large-cleaned.jsonl"
VALIDATION_CLEANED_FILE = "pan20-authorship-verification-validation-large-cleaned.jsonl"
TEST_CLEANED_FILE = "pan21-authorship-verification-test-cleaned.jsonl"

# [^]] = any character except "]", so every [...] block is removed
_BRACKET_BLOCK = re.compile(r"\[[^]]*\]")
_NOT_KEPT = re.compile(r"[^a-zA-Z0-9\s,']")


def clean_text(text: str) -> str:
    """Remove HTML markup, [...] blocks and every character outside the kept set.

    Args:
        text: One raw PAN text.

    Returns:
        The text with only ASCII letters, digits, whitespace, commas and apostrophes.
    """
    text_html_parsed = BeautifulSoup(text, "html.parser").get_text()
    text_no_brackets = _BRACKET_BLOCK.sub("", text_html_parsed)
    text_desired_characters = _NOT_KEPT.sub("", text_no_brackets)
    return text_desired_characters


def read_jsonl(path: Path) -> pd.DataFrame:
    """Read a jsonl file (one JSON object per line) into a DataFrame.

    Args:
        path: Path to the jsonl file.

    Returns:
        One row per line, one column per key.
    """
    with path.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file]
    return pd.DataFrame(records)


@dataclass(frozen=True)
class DatasetSplit:
    """Train and validation part of the PAN 2020 training pairs.

    Attributes:
        train_df: Training pairs, in the shuffled order of train_test_split.
        validation_df: Validation pairs, in the shuffled order of train_test_split.
    """

    train_df: pd.DataFrame
    validation_df: pd.DataFrame


@dataclass(frozen=True)
class DatasetCounts:
    """Number of pairs written to each cleaned file.

    Attributes:
        train: Pairs in the training file.
        validation: Pairs in the validation file.
        test: Pairs in the test file.
    """

    train: int
    validation: int
    test: int


def _read_pair_line(line: str, clean_texts: bool) -> dict[str, str | list[str]]:
    """Parse one line of a PAN pairs file, optionally cleaning both texts.

    Defined at module level, so worker processes can import it.

    Args:
        line: One JSON line of the pairs file.
        clean_texts: Whether to apply clean_text to both texts.

    Returns:
        The id and the texts of the pair; the fandoms are dropped.
    """
    record = json.loads(line)
    texts = record["pair"]
    if clean_texts:
        texts = [clean_text(text) for text in texts]
    return {"id": record["id"], "pair": texts}
