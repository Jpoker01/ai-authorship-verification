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
from collections.abc import Callable, Mapping, Sequence
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
    (?<![\w'])                    # the key starts a word and is not in single quotes
    "? \**                        # an optional double quote and markdown bold before the key
    (?i:answer)                   # the key in any letter case - "answer", Answer, ANSWER
    \** "? \s* \** \s*            # the closing bold and quote, then spaces
    : \s* \** \s*                 # the colon, and bold after it as in **Answer:** True
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
# strings of the prompt templates (Llama-2 chat tags, the delimiter of the texts) - a text that
# holds one could close the delimiter or open a new turn inside the prompt
TEMPLATE_STRINGS = ("[INST]", "[/INST]", "<<SYS>>", "<</SYS>>", "```")


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
                f"The prompt has {placeholder} {found.count(placeholder)} times, not once.",
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

    def text_of(placeholder: re.Match[str]) -> str:
        return texts[placeholder.group(0)]

    return _PLACEHOLDER_PATTERN.sub(text_of, template)


def neutralize_template_strings(
    text: str,
    strings: Sequence[str] = TEMPLATE_STRINGS,
) -> tuple[str, int]:
    """Replace every occurrence of the given strings in a text by one space.

    Args:
        text: One text of a pair.
        strings: The strings to replace - the prompt template strings, or the special tokens
            of a tokenizer, which the cut to the token limit would delete and glue the words.

    Returns:
        The text without the strings and the number of strings replaced.
    """
    count = 0
    for string in strings:
        count += text.count(string)
        text = text.replace(string, " ")
    return text, count


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
        text,
        truncation=True,
        max_length=max_tokens,
        add_special_tokens=False,
    )
    return tokenizer.decode(encoding["input_ids"], skip_special_tokens=True).strip()


@dataclass(frozen=True)
class PromptBuilder:
    """Everything that turns the two texts of a pair into the token ids of its prompt.

    Attributes:
        tokenizer: Tokenizer of the model.
        prompt: The prompt file.
        text_token_limit: Most tokens kept of each text, None for the whole texts.
        chat_template: A chat template that replaces the tokenizer's own, None to use it.
    """

    tokenizer: Any
    prompt: Prompt
    text_token_limit: int | None = None
    chat_template: str | None = None

    def build_token_ids(self, text1: str, text2: str) -> list[int]:
        """Return the token ids of the prompt of one pair, with the generation prompt of a chat.

        Args:
            text1: The first text of the pair.
            text2: The second text of the pair.

        Returns:
            The token ids the engine gets, with the special tokens of the chat template.
        """
        text1 = truncate_text(self.tokenizer, text1, self.text_token_limit)
        text2 = truncate_text(self.tokenizer, text2, self.text_token_limit)
        messages = []
        for message in self.prompt.messages:
            content = _fill(message["content"], text1, text2)
            messages.append({"role": message["role"], "content": content})
        # tokenize=True encodes without extra special tokens - the template holds them
        token_ids = self.tokenizer.apply_chat_template(
            messages,
            chat_template=self.chat_template,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )
        return list(token_ids)


def count_prompt_tokens(builder: PromptBuilder, pairs_df: pd.DataFrame) -> pd.Series:
    """Return the number of prompt tokens of every pair, indexed by the pair id.

    Args:
        builder: Builds the prompt of a pair.
        pairs_df: Rows with id and pair (the two texts).

    Returns:
        The prompt length of every pair, in the order of pairs_df.
    """
    counts = []
    for text1, text2 in pairs_df["pair"]:
        prompt_ids = builder.build_token_ids(text1, text2)
        counts.append(len(prompt_ids))
    return pd.Series(counts, index=pairs_df["id"].tolist(), name="prompt_tokens")


def check_prompt_lengths(prompt_tokens: pd.Series, max_model_len: int) -> None:
    """Raise before any generation if a prompt leaves no room for one generated token.

    The engine refuses such a prompt and with it the whole chunk it belongs to.

    Args:
        prompt_tokens: Output of count_prompt_tokens.
        max_model_len: Context length of the engine.

    Raises:
        ValueError: If any prompt has max_model_len tokens or more.
    """
    too_long = prompt_tokens[prompt_tokens >= max_model_len]
    if not too_long.empty:
        raise ValueError(
            f"{len(too_long)} prompts leave no room for an answer in {max_model_len} tokens, "
            f"the longest has {too_long.max()} tokens (pair {too_long.idxmax()}).",
        )


def describe_run(
    engine_settings: Mapping[str, object],
    builder: PromptBuilder,
    decoding: DecodingSettings,
    chunk_size: int,
    pair_ids: Sequence[str],
) -> dict[str, object]:
    """Return everything that decides the generated texts - an answers folder resumes only an equal run.

    Args:
        engine_settings: The arguments of the engine that change its outputs (model, revision, ...).
        builder: Builds the prompt of a pair.
        decoding: Sampling settings of every request.
        chunk_size: Pairs per engine call.
        pair_ids: Ids of the pairs, in the order they are generated.

    Returns:
        A JSON-serializable description of the run.
    """
    if builder.chat_template is None:
        chat_template_sha256 = None
    else:
        chat_template_sha256 = _sha256_text(builder.chat_template)
    return {
        "engine_settings": dict(engine_settings),
        "prompt_sha256": builder.prompt.sha256,
        "chat_template_sha256": chat_template_sha256,
        "text_token_limit": builder.text_token_limit,
        "decoding": asdict(decoding),
        "chunk_size": chunk_size,
        "n_pairs": len(pair_ids),
        "pair_ids_sha256": _sha256_text("\n".join(pair_ids)),
    }


def _write_json_atomically(path: Path, content: object) -> None:
    """Write a JSON file through a temporary file, so a kill leaves the old file or none."""
    partial_path = path.with_name(f"{path.name}{PARTIAL_SUFFIX}")
    with partial_path.open("w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False)
        file.flush()
        # on the disk before the rename, also when the node crashes
        os.fsync(file.fileno())
    os.replace(partial_path, path)


def find_finished_chunks(answers_dir: Path) -> set[int]:
    """Return the index of every chunk whose file is complete.

    Args:
        answers_dir: The answers folder of the run.

    Returns:
        The chunk indexes of the chunk files; a temporary file of a killed write is no chunk.
    """
    finished = set()
    for path in answers_dir.glob("chunk-*.json"):
        match = CHUNK_FILE_PATTERN.fullmatch(path.name)
        if match:
            finished.add(int(match.group(1)))
    return finished


def _chunk_path(answers_dir: Path, index: int) -> Path:
    """Return the file of one chunk."""
    return answers_dir / f"chunk-{index:05d}.json"


def prepare_answers_dir(
    answers_dir: Path,
    run_description: Mapping[str, object],
) -> None:
    """Create the answers folder of a run, or check that an existing one belongs to this run.

    A temporary file a killed write left behind is removed - only after the check passed.

    Args:
        answers_dir: Folder of the answers of one run, outside the repository.
        run_description: Output of describe_run.

    Raises:
        RuntimeError: If the folder belongs to a run with other settings, or holds chunk files
            without the description of their run; nothing is changed then.
    """
    run_path = answers_dir / RUN_FILE
    # a JSON round trip, so a tuple and a list of the same values compare equal
    expected = json.loads(json.dumps(run_description))
    if run_path.is_file():
        stored = json.loads(run_path.read_text(encoding="utf-8"))
        changes = []
        for key in sorted(set(stored) | set(expected)):
            if stored.get(key) != expected.get(key):
                changes.append(f"{key}: {stored.get(key)!r} -> {expected.get(key)!r}")
        if changes:
            raise RuntimeError(
                f"{answers_dir} holds the answers of other settings - {'; '.join(changes)}. "
                "Move it away or restore the settings.",
            )
    elif find_finished_chunks(answers_dir):
        raise RuntimeError(
            f"{answers_dir} holds chunk files but no {RUN_FILE}, so their settings are "
            "unknown. Move it away.",
        )
    answers_dir.mkdir(parents=True, exist_ok=True)
    for partial_path in answers_dir.glob(f"*{PARTIAL_SUFFIX}"):
        partial_path.unlink()
    if not run_path.is_file():
        _write_json_atomically(run_path, expected)


def split_chunks(pair_ids: Sequence[str], chunk_size: int) -> list[list[str]]:
    """Split the pair ids into consecutive chunks; the last chunk may be shorter.

    Args:
        pair_ids: Ids of the pairs, in the order they are generated.
        chunk_size: Pairs per engine call.

    Returns:
        The ids of every chunk, in pair order.

    Raises:
        ValueError: If chunk_size is below 1, there are no ids, or an id occurs more than
            once - a repeated id would give two pairs one prompt.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size has to be at least 1, not {chunk_size}.")
    if not pair_ids:
        raise ValueError("There are no pair ids to split.")
    seen_ids = set()
    for pair_id in pair_ids:
        if pair_id in seen_ids:
            raise ValueError(f"The pair id {pair_id!r} occurs more than once.")
        seen_ids.add(pair_id)
    chunks = []
    for start in range(0, len(pair_ids), chunk_size):
        chunks.append(list(pair_ids[start : start + chunk_size]))
    return chunks


def _answer_record(
    pair_id: str,
    prompt_ids: list[int],
    output: Any,
) -> dict[str, object]:
    """Return what is kept of one request output.

    Args:
        pair_id: Id of the pair.
        prompt_ids: Token ids of the prompt of the pair.
        output: The request output of the engine for this prompt.

    Returns:
        id, prompt_tokens, generated_tokens, finish_reason and the generated text.

    Raises:
        RuntimeError: If the output belongs to another prompt.
    """
    if list(output.prompt_token_ids) != prompt_ids:
        raise RuntimeError(
            f"The engine returned the answer of another prompt for pair {pair_id}.",
        )
    completion = output.outputs[0]
    return {
        "id": pair_id,
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(completion.token_ids),
        "finish_reason": completion.finish_reason,
        "text": completion.text,
    }


def generate_answers(
    engine: GenerationEngine,
    sampling_params: Any,
    builder: PromptBuilder,
    pairs_df: pd.DataFrame,
    answers_dir: Path,
    chunk_size: int,
    on_chunk: Callable[[int, int], None] | None = None,
) -> int:
    """Generate the answers of every chunk that has no file yet, one engine call per chunk.

    Call prepare_answers_dir first. A chunk file is written only when its whole chunk is
    answered, so a kill loses at most the chunk that was running.

    Args:
        engine: The engine (vllm.LLM).
        sampling_params: The sampling parameters object of the engine.
        builder: Builds the prompt of a pair.
        pairs_df: Rows with id and pair (the two texts), in the order they are generated.
        answers_dir: The prepared answers folder of the run.
        chunk_size: Pairs per engine call.
        on_chunk: Called with the number of finished chunks and of all chunks after every
            chunk file is written, e.g. to update a progress file.

    Returns:
        The number of chunks generated by this call.

    Raises:
        RuntimeError: If the engine returns another number of outputs or another prompt's output.
    """
    texts_by_id = dict(zip(pairs_df["id"], pairs_df["pair"], strict=True))
    chunks = split_chunks(pairs_df["id"].tolist(), chunk_size)
    finished = find_finished_chunks(answers_dir)
    generated = 0
    for index, chunk_ids in enumerate(chunks):
        if index in finished:
            continue
        prompt_ids = []
        requests = []
        for pair_id in chunk_ids:
            text1, text2 = texts_by_id[pair_id]
            ids = builder.build_token_ids(text1, text2)
            prompt_ids.append(ids)
            requests.append({"prompt_token_ids": ids})
        start = time.monotonic()
        outputs = engine.generate(requests, sampling_params, use_tqdm=False)
        seconds = time.monotonic() - start
        if len(outputs) != len(chunk_ids):
            raise RuntimeError(
                f"Chunk {index}: {len(outputs)} outputs for {len(chunk_ids)} prompts.",
            )
        records = []
        for pair_id, ids, output in zip(chunk_ids, prompt_ids, outputs, strict=True):
            records.append(_answer_record(pair_id, ids, output))
        _write_json_atomically(
            _chunk_path(answers_dir, index),
            {"chunk": index, "seconds": seconds, "answers": records},
        )
        generated += 1
        finished.add(index)
        if on_chunk is not None:
            on_chunk(len(finished), len(chunks))
        logger.info(
            f"Chunk {index + 1} of {len(chunks)} - {len(chunk_ids)} answers in {seconds:.1f} s.",
        )
    return generated


def parse_answer(text: str) -> ParsedAnswer:
    """Read the answer of a generated text - only the value of the answer key counts.

    The key may be in any letter case, with or without double quotes or markdown bold around it.

    Args:
        text: The generated text alone, without the prompt.

    Returns:
        SAME_AUTHOR for true, DIFFERENT_AUTHOR for false; no pred without the key, with a
        value that is not a whole boolean, or with both values.
    """
    values = set()
    for match in _ANSWER_PATTERN.finditer(text):
        values.add(match.group("value").lower())
    if not values:
        return ParsedAnswer(pred=None, status=NO_ANSWER)
    if len(values) > 1:
        return ParsedAnswer(pred=None, status=CONFLICTING_ANSWERS)
    pred = SAME_AUTHOR if values.pop() == "true" else DIFFERENT_AUTHOR
    return ParsedAnswer(pred=pred, status=ANSWERED)


def read_answers(
    answers_dir: Path,
    pair_ids: Sequence[str],
    chunk_size: int,
) -> GeneratedAnswers:
    """Read every chunk file of a finished generation and parse its answers.

    Args:
        answers_dir: The answers folder of the run.
        pair_ids: Ids of the pairs, in the order they were generated.
        chunk_size: Pairs per engine call.

    Returns:
        The answers of all pairs in pair order and the summed generation time.

    Raises:
        RuntimeError: If a chunk file is missing or holds other pairs.
    """
    records = []
    generation_seconds = 0.0
    for index, chunk_ids in enumerate(split_chunks(pair_ids, chunk_size)):
        chunk_path = _chunk_path(answers_dir, index)
        if not chunk_path.is_file():
            raise RuntimeError(f"{chunk_path} is missing - generate the answers first.")
        chunk = json.loads(chunk_path.read_text(encoding="utf-8"))
        chunk_answer_ids = [answer["id"] for answer in chunk["answers"]]
        if chunk_answer_ids != chunk_ids:
            raise RuntimeError(f"{chunk_path} holds the answers of other pairs.")
        records.extend(chunk["answers"])
        generation_seconds += chunk["seconds"]
    answers_df = pd.DataFrame(
        records,
        columns=["id", "prompt_tokens", "generated_tokens", "finish_reason", "text"],
    )
    parsed = [parse_answer(text) for text in answers_df["text"]]
    answers_df["pred"] = pd.array([answer.pred for answer in parsed], dtype="Int64")
    answers_df["status"] = [answer.status for answer in parsed]
    return GeneratedAnswers(
        answers_df=answers_df,
        generation_seconds=generation_seconds,
    )


def build_prediction_tables(
    pairs_df: pd.DataFrame,
    answers_df: pd.DataFrame,
    unanswered_pred: int,
    unanswered_score: float,
) -> PredictionTables:
    """Return the prediction, unanswered and generation tables of a run.

    Args:
        pairs_df: Rows with id and same, the pairs that were generated.
        answers_df: The answers_df of read_answers for these pairs.
        unanswered_pred: Prediction of a pair without an answer.
        unanswered_score: Score of a pair without an answer.

    Returns:
        The three tables; an answered pair has the score 1,0 (same author) or 0,0.

    Raises:
        ValueError: If the answers are not exactly the pairs of pairs_df in their order, or the
            rule for unanswered pairs is not a 0 / 1 prediction with a finite score.
    """
    if answers_df["id"].tolist() != pairs_df["id"].tolist():
        raise ValueError("The answers do not hold every pair once, in the same order.")
    if unanswered_pred not in (SAME_AUTHOR, DIFFERENT_AUTHOR):
        raise ValueError(f"Unanswered pairs need a 0 / 1 pred, not {unanswered_pred}.")
    if not math.isfinite(unanswered_score):
        raise ValueError(
            f"Unanswered pairs need a finite score, not {unanswered_score}.",
        )
    answered = (answers_df["status"] == ANSWERED).to_numpy()
    pred = answers_df["pred"].fillna(unanswered_pred).astype(int).to_numpy()
    score = pred.astype(float)
    score[~answered] = unanswered_score
    predictions_df = pd.DataFrame(
        {
            "id": pairs_df["id"].to_numpy(),
            "same": pairs_df["same"].astype(int).to_numpy(),
            "score": score,
            "pred": pred,
        },
    )
    unanswered_df = answers_df.loc[~answered, UNANSWERED_COLUMNS].reset_index(drop=True)
    generation_df = answers_df[GENERATION_COLUMNS].reset_index(drop=True)
    return PredictionTables(
        predictions_df=predictions_df,
        unanswered_df=unanswered_df,
        generation_df=generation_df,
    )
