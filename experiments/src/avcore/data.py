"""Loading, cleaning and the train / validation split of the PAN authorship verification pairs.

Builds the cleaned jsonl files that every notebook family except the LLMs reads.
"""

import json
import re
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from itertools import islice
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

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

# pairs lines handed to the worker processes at once
_BATCH_LINES = 1_024


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


def _parse_pair_lines(
    lines: Iterable[str], clean_texts: bool, workers: int
) -> Iterator[dict[str, str | list[str]]]:
    """Yield _read_pair_line for every line, in the input order.

    With workers > 1 the lines are parsed in worker processes, one batch of
    _BATCH_LINES lines at a time, so only one batch of raw text is in memory.

    Args:
        lines: The lines of a pairs file (an open file or a list).
        clean_texts: Whether to apply clean_text to both texts.
        workers: Number of worker processes; 1 parses in this process.

    Yields:
        The id and the texts of every pair.
    """
    lines = iter(lines)
    read_line = partial(_read_pair_line, clean_texts=clean_texts)
    if workers == 1:
        yield from map(read_line, lines)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        while batch := list(islice(lines, _BATCH_LINES)):
            yield from executor.map(read_line, batch, chunksize=16)


def load_pan_pairs(
    pairs_path: Path,
    truth_path: Path,
    clean_texts: bool = False,
    workers: int = 1,
) -> pd.DataFrame:
    """Load a PAN pairs file and its truth file into one DataFrame.

    Line i of the truth file holds the label of line i of the pairs file, so both
    files are read in step and a repeated id never multiplies the rows.

    Args:
        pairs_path: The pairs jsonl file (id, fandoms, pair).
        truth_path: The truth jsonl file (id, same, authors).
        clean_texts: Whether to apply clean_text to both texts of every pair.
        workers: Number of worker processes that parse and clean the pairs lines.

    Returns:
        One row per pair with the columns id, pair and same, in the order of the pairs file.

    Raises:
        ValueError: If the files differ in length, the ids on a line differ
        or a pair does not have exactly 2 texts.
    """
    records = []
    with (
        pairs_path.open(encoding="utf-8") as pairs_file,
        truth_path.open(encoding="utf-8") as truth_file,
    ):
        pairs = _parse_pair_lines(pairs_file, clean_texts, workers)
        # strict=True raises if one file has more lines than the other
        lines_in_step = zip(pairs, truth_file, strict=True)
        line_number = 0
        for pair, truth_line in lines_in_step:
            line_number += 1
            truth = json.loads(truth_line)
            pair_id = pair["id"]
            texts = pair["pair"]

            if pair_id != truth["id"]:
                raise ValueError(
                    f"line {line_number}: id {pair_id} does not match truth id {truth['id']}"
                )
            if len(texts) != 2:
                raise ValueError(f"line {line_number}: {len(texts)} texts instead of 2")
            records.append({"id": pair_id, "pair": texts, "same": truth["same"]})
    return pd.DataFrame(records, columns=PAIR_COLUMNS)


def split_train_validation(
    pairs_df: pd.DataFrame,
    validation_size: int = VALIDATION_SIZE,
    random_state: int = RANDOM_STATE,
) -> DatasetSplit:
    """Split the training pairs into a training and a validation part.

    Args:
        pairs_df: All PAN 2020 training pairs.
        validation_size: Number of pairs in the validation part.
        random_state: Seed of the shuffle.

    Returns:
        The two parts, each in the shuffled order of train_test_split.
    """
    train_df, validation_df = train_test_split(
        pairs_df,
        test_size=validation_size,
        random_state=random_state,
        shuffle=True,
    )
    return DatasetSplit(train_df=train_df, validation_df=validation_df)


def write_pairs_jsonl(pairs_df: pd.DataFrame, path: Path) -> None:
    """Write the pairs as jsonl with the keys id, pair and same, one pair per line.

    Args:
        pairs_df: Pairs with the columns id, pair and same.
        path: Target file; it must not exist yet.

    Raises:
        FileExistsError: If the target file already exists.
    """
    with path.open("x", encoding="utf-8") as file:
        for record in pairs_df[PAIR_COLUMNS].to_dict(orient="records"):
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
