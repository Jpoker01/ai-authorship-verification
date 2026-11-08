"""Loading, cleaning and the train / validation split of the PAN authorship verification pairs.

Builds the cleaned jsonl files that every notebook family except the LLMs reads.
"""

import re

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
