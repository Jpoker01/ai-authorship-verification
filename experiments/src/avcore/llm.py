"""Prompting, decoding and strict answer parsing of the LLM runs, resumable after a crash.

Every pair gets one prompt - the messages of the prompt file with its two texts filled in, turned
into token ids by the chat template of the model. The pairs are generated in chunks, and every
finished chunk is written to a file of its own, so a restarted run generates only the chunks
that are missing and never repeats a finished one.

The module needs neither vLLM nor transformers - the engine and the tokenizer come as parameters.
"""

import hashlib
import json
import logging
import math
import os
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

logger = logging.getLogger(__name__)


MESSAGE_ROLES = ("system", "user", "assistant")
TEXT_PLACEHOLDERS = ("{text1}", "{text2}")
_PLACEHOLDER_PATTERN = re.compile(r"\{text[12]\}")

# 0 = different author, 1 = same author
SAME_AUTHOR = 1
DIFFERENT_AUTHOR = 0
ANSWERED = "answered"
NO_ANSWER = "no_answer"
CONFLICTING_ANSWERS = "conflicting_answers"

_ANSWER_PATTERN = re.compile(
    r"""
    "answer" \s* : \s*            # the key in double quotes and its colon
    (?P<quote>"?)                 # an optional quote before the value
    (?i:(?P<value>true|false))    # the value in any letter case - JSON or Python boolean
    (?P=quote)                    # the closing quote, only if the value had an opening one
    (?=\s*(?:[,.}\]]|$))          # the value ends here, so "True/False" is no answer
    """,
    re.MULTILINE | re.VERBOSE,
)

RUN_FILE = "run.json"
CHUNK_FILE_PATTERN = re.compile(r"chunk-(\d{5})\.json")
PARTIAL_SUFFIX = ".partial"
GENERATION_COLUMNS = [
    "id",
    "prompt_tokens",
    "generated_tokens",
    "finish_reason",
    "status",
]
UNANSWERED_COLUMNS = ["id", "status", "finish_reason"]


@dataclass(frozen=True)
class Prompt:
    """A prompt file - chat messages with the placeholders {text1} and {text2}.

    Attributes:
        messages: Role and content of every chat message.
        sha256: sha256 of the file bytes - the same value as sha256sum of the file.
    """

    messages: list[dict[str, str]]
    sha256: str


@dataclass(frozen=True)
class DecodingSettings:
    """Sampling settings of every request; a temperature of 0 is greedy decoding.

    Attributes:
        temperature: Sampling temperature.
        top_p: Share of the probability mass the next token is drawn from.
        max_tokens: Most tokens generated per answer.
        seed: Seed of the sampling, without effect on greedy decoding.
    """

    temperature: float
    top_p: float
    max_tokens: int
    seed: int | None = None


@dataclass(frozen=True)
class ParsedAnswer:
    """The answer read from one generated text.

    Attributes:
        pred: SAME_AUTHOR, DIFFERENT_AUTHOR or None when the text gives no single answer.
        status: ANSWERED, NO_ANSWER or CONFLICTING_ANSWERS.
    """

    pred: int | None
    status: str

@dataclass(frozen=True)
class GeneratedAnswers:
    """Every answer of a finished generation.

    Attributes:
        answers_df: One row per pair in pair order - id, prompt_tokens, generated_tokens,
            finish_reason, text, pred and status.
        generation_seconds: Seconds of the engine calls of all chunks, summed over restarts.
    """

    answers_df: pd.DataFrame
    generation_seconds: float


@dataclass(frozen=True)
class PredictionTables:
    """The tables a run folder holds.

    Attributes:
        predictions_df: id, same, score and pred of every pair.
        unanswered_df: id, status and finish_reason of every pair without an answer.
        generation_df: id, prompt_tokens, generated_tokens, finish_reason and status of every pair.
    """

    predictions_df: pd.DataFrame
    unanswered_df: pd.DataFrame
    generation_df: pd.DataFrame


class GenerationEngine(Protocol):
    """The part of vllm.LLM the runs use."""

    def generate(
        self,
        prompts: Sequence[Mapping[str, list[int]]],
        sampling_params: Any,
        *,
        use_tqdm: bool,
    ) -> list[Any]:
        """Return one request output per prompt, in the order of the prompts."""
        ...

def _sha256_text(text: str) -> str:
    """Return the sha256 hex digest of a text in UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _check_placeholders(templates: Sequence[str]) -> None:
    """Raise ValueError unless each text placeholder occurs exactly once in all templates."""
    # a placeholder has no newline, so no match can span two joined templates
    found = _PLACEHOLDER_PATTERN.findall("\n".join(templates))
    for placeholder in TEXT_PLACEHOLDERS:
        if found.count(placeholder) != 1:
            raise ValueError(
                f"The prompt has {placeholder} {found.count(placeholder)} times, not once."
            )

def load_prompt(path: Path) -> Prompt:
    """Read a prompt file - {"messages": [{"role": ..., "content": ...}, ...]}.

    Args:
        path: The JSON prompt file.

    Returns:
        The prompt and the sha256 of the file bytes.

    Raises:
        ValueError: For an unknown role, or a placeholder that is missing or repeated.
    """
    file_bytes = path.read_bytes()
    messages = json.loads(file_bytes.decode("utf-8"))["messages"]
    for message in messages:
        if message.get("role") not in MESSAGE_ROLES:
            raise ValueError(f"Unknown message role {message.get('role')!r}.")
    _check_placeholders([message["content"] for message in messages])
    return Prompt(
        messages=messages,
        sha256=hashlib.sha256(file_bytes).hexdigest(),
    )

def _fill(template: str, text1: str, text2: str) -> str:
    """Put the two texts in place of their placeholders; the inserted texts are not searched again."""
    texts = {"{text1}": text1, "{text2}": text2}
    return _PLACEHOLDER_PATTERN.sub(lambda match: texts[match.group(0)], template)


def truncate_text(tokenizer: Any, text: str, max_tokens: int | None) -> str:
    """Cut a text to its first max_tokens tokens and strip it, so a pair fits a short context.

    Args:
        tokenizer: Tokenizer of the model.
        text: One text of a pair.
        max_tokens: Most tokens kept; None keeps the text unchanged.

    Returns:
        The decoded first max_tokens tokens without surrounding whitespace, or the text itself.
    """
    if max_tokens is None:
        return text
    encoding = tokenizer(
        text, truncation=True, max_length=max_tokens, add_special_tokens=False
    )
    return tokenizer.decode(encoding["input_ids"], skip_special_tokens=True).strip()