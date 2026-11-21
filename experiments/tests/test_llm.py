"""Tests of avcore.llm - prompt files, prompt rendering, strict answer parsing, chunked generation.

The engine is a fake with the generate interface of vllm.LLM and the tokenizer a fake with the
methods the module calls, so no model is loaded; the tests with a real tokenizer skip without
transformers.
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from avcore import llm, tracking

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
LLAMA2_TEMPLATE_FILE = "llama2_chat_template.jinja"
LLAMA2_PROMPT_FILE = "prompt_llama2_v1.json"
# the format text and the question of the v1 Llama-2 notebook, character for character
V1_RESPONSE_FORMAT = (
    "\nRespond with a JSON object including two key elements:\n"
    "{\n"
    '  "analysis": Reasoning behind your answer.,\n'
    '  "answer":  A boolean (True/False) answer.\n'
    "}\n"
)
V1_QUESTION = (
    "\nVerify if the input texts were written by the same author.\n"
    "Do not consider topic differences. Focus on grammatical styles. \n"
    "Analyze the writing styles of the input texts, disregarding the differences in topic "
    "and content. \n"
    "Focus on linguistic features such as phrasal verbs, modal verbs, punctuation, rare "
    "words, affixes, quantities, \n"
    "humor, sarcasm, typographical errors, and misspellings. \n"
)
V1_TEXTS = (
    " \n        The input texts (Text 1 and Text 2) are delimited with triple backticks. "
    "```Text 1: {text1}, \n\nText 2: {text2}```\n\n"
)
BOS = "<s>"
CHAT_PROMPT = {
    "messages": [
        {"role": "system", "content": "Answer in JSON."},
        {"role": "user", "content": "First {text1} second {text2} end"},
    ],
}
JSON_TRUE = '{"analysis": "alike", "answer": true}'
JSON_FALSE = '{"analysis": "unlike", "answer": false}'
# the shared settings of the generation tests - the fake engine ignores the sampling parameters
CHUNK_SIZE = 2
SAMPLING_PARAMS = None
# the rule of the runs for a pair without an answer
UNANSWERED_PRED = 0
UNANSWERED_SCORE = 0.5


class FakeTokenizer:
    """Word-level tokenizer with the methods and parameter names of a Hugging Face tokenizer."""

    def __init__(
        self, chat_template: str | None = "fake-template", decode_prefix: str = ""
    ) -> None:
        self.chat_template = chat_template
        # a sentencepiece tokenizer decodes the first word with its leading space
        self.decode_prefix = decode_prefix
        # the token id of a word is its position in this list
        self.words = [BOS]
        self.chat_calls: list[dict] = []

    def _token_id(self, word: str) -> int:
        if word not in self.words:
            self.words.append(word)
        return self.words.index(word)

    def __call__(
        self,
        text: str,
        truncation: bool = False,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, list[int]]:
        token_ids = [self._token_id(word) for word in text.split()]
        if truncation:
            token_ids = token_ids[:max_length]
        if add_special_tokens:
            token_ids.insert(0, self._token_id(BOS))
        return {"input_ids": token_ids}

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        words = [self.words[token_id] for token_id in token_ids]
        if skip_special_tokens:
            words = [word for word in words if word != BOS]
        return self.decode_prefix + " ".join(words)

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        chat_template: str | None = None,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        return_dict: bool = True,
    ) -> list[int] | dict[str, list[int]] | str:
        template = chat_template if chat_template is not None else self.chat_template
        if template is None:
            raise ValueError(
                "tokenizer.chat_template is not set and no template argument was passed"
            )
        self.chat_calls.append(
            {"template": template, "add_generation_prompt": add_generation_prompt}
        )
        words = [BOS]
        for message in conversation:
            words.append(f"<{message['role']}>")
            words.append(message["content"])
        if add_generation_prompt:
            words.append("<assistant>")
        rendered = " ".join(words)
        if not tokenize:
            return rendered
        token_ids = self(rendered, add_special_tokens=False)["input_ids"]
        # a dict by default, as a tokenizer whose default return type changed
        if return_dict:
            return {"input_ids": token_ids}
        return token_ids


def write_prompt_file(folder: Path, content: dict, name: str = "prompt.json") -> Path:
    """Write a prompt file and return its path."""
    path = folder / name
    # the layout of the prompt files of the repository - not the bytes of json.dumps
    path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    return path


def make_pairs(n_pairs: int) -> pd.DataFrame:
    """Return pairs whose first text holds the word pair-<i>, so the fake engine knows the pair."""
    pair_ids = []
    pairs = []
    same = []
    for index in range(n_pairs):
        pair_ids.append(f"id-{index}")
        pairs.append([f"pair-{index} alpha beta", f"gamma delta {index}"])
        same.append(index % 2 == 0)
    return pd.DataFrame({"id": pair_ids, "pair": pairs, "same": same})


def make_engine_answers(n_pairs: int) -> dict[str, str]:
    """Return a true answer for the even pairs and a false answer for the odd ones."""
    answers = {}
    for index in range(n_pairs):
        if index % 2 == 0:
            answers[f"pair-{index}"] = JSON_TRUE
        else:
            answers[f"pair-{index}"] = JSON_FALSE
    return answers


class FakeEngine:
    """Engine with the generate interface of vllm.LLM - the answer text of a pair comes from its id word."""

    def __init__(
        self,
        answers: dict[str, str],
        tokenizer: FakeTokenizer,
        fail_at_call: int | None = None,
    ) -> None:
        # the answer text of every pair, by the marker word in its first text (pair-0, pair-1, ...)
        self.answers = answers
        self.tokenizer = tokenizer
        # the number of prompts of every generate call so far
        self.prompts_per_call: list[int] = []
        # the failures a test can switch on
        self.fail_at_call = fail_at_call
        self.reverse_outputs = False
        self.drop_last_output = False

    def _answer_for(self, prompt_ids: list[int]) -> str:
        """Return the answer of the pair whose marker word is in the prompt.

        The engine gets only token ids, so the marker word is how it knows the pair of a prompt.
        """
        prompt_words = self.tokenizer.decode(prompt_ids).split()
        for marker_word, answer in self.answers.items():
            if marker_word in prompt_words:
                return answer
        raise KeyError("The prompt holds no marker word of a pair.")

    def _output_for(self, prompt_ids: list[int]) -> SimpleNamespace:
        # the attributes of a vllm.RequestOutput and its first completion that the module reads
        text = self._answer_for(prompt_ids)
        completion = SimpleNamespace(
            text=text, token_ids=text.split(), finish_reason="stop"
        )
        return SimpleNamespace(prompt_token_ids=prompt_ids, outputs=[completion])

    def generate(
        self,
        prompts: list[dict[str, list[int]]],
        sampling_params: object,
        *,
        use_tqdm: bool,
    ) -> list:
        self.prompts_per_call.append(len(prompts))
        call_number = len(self.prompts_per_call)
        if call_number == self.fail_at_call:
            raise RuntimeError("the engine died")
        outputs = []
        for prompt in prompts:
            outputs.append(self._output_for(prompt["prompt_token_ids"]))
        if self.drop_last_output:
            outputs.pop()
        if self.reverse_outputs:
            outputs.reverse()
        return outputs
