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
        self,
        chat_template: str | None = "fake-template",
        decode_prefix: str = "",
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
                "tokenizer.chat_template is not set and no template argument was passed",
            )
        self.chat_calls.append(
            {"template": template, "add_generation_prompt": add_generation_prompt},
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
            text=text,
            token_ids=text.split(),
            finish_reason="stop",
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


@pytest.fixture
def chat_builder(tmp_path: Path) -> llm.PromptBuilder:
    """A prompt builder of the chat prompt with a fake tokenizer."""
    prompt = llm.load_prompt(write_prompt_file(tmp_path, CHAT_PROMPT))
    return llm.PromptBuilder(tokenizer=FakeTokenizer(), prompt=prompt)


@pytest.fixture
def run_description(chat_builder: llm.PromptBuilder) -> dict:
    """The description of a run of four pairs in chunks of two."""
    pair_ids = make_pairs(4)["id"].tolist()
    decoding = llm.DecodingSettings(temperature=0.0, top_p=1.0, max_tokens=50)
    engine_settings = {"model": "fake"}
    return llm.describe_run(
        engine_settings,
        chat_builder,
        decoding,
        CHUNK_SIZE,
        pair_ids,
    )


def test_load_prompt_hashes_the_file_bytes(tmp_path: Path) -> None:
    """The prompt sha256 is the sha256 of the file bytes, the value sha256sum prints."""
    prompt_path = write_prompt_file(tmp_path, CHAT_PROMPT)
    prompt = llm.load_prompt(prompt_path)
    assert prompt.sha256 == hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    assert prompt.messages == CHAT_PROMPT["messages"]


@pytest.mark.parametrize(
    "user_content",
    ["Only {text1} here", "{text1} {text2} {text2}"],
)
def test_load_prompt_rejects_a_missing_or_repeated_placeholder(
    tmp_path: Path,
    user_content: str,
) -> None:
    """Each of {text1} and {text2} has to occur exactly once."""
    content = {"messages": [{"role": "user", "content": user_content}]}
    prompt_path = write_prompt_file(tmp_path, content)
    with pytest.raises(ValueError, match="not once"):
        llm.load_prompt(prompt_path)


def test_load_prompt_rejects_an_unknown_role(tmp_path: Path) -> None:
    """Only the roles system, user and assistant exist."""
    content = {"messages": [{"role": "narrator", "content": "{text1} {text2}"}]}
    prompt_path = write_prompt_file(tmp_path, content)
    with pytest.raises(ValueError, match="Unknown"):
        llm.load_prompt(prompt_path)


def test_shipped_prompt_has_the_v1_llama2_wording() -> None:
    """The prompt file of the runs holds the v1 Llama-2 format text and question unchanged."""
    prompt_path = PROMPT_DIR / LLAMA2_PROMPT_FILE
    prompt = llm.load_prompt(prompt_path)
    expected_messages = [
        {"role": "system", "content": V1_RESPONSE_FORMAT},
        {"role": "user", "content": V1_QUESTION + V1_TEXTS},
    ]
    assert prompt.messages == expected_messages


def test_build_token_ids_fills_each_text_once(chat_builder: llm.PromptBuilder) -> None:
    """A placeholder inside a text is kept as it is - the inserted texts are not searched again."""
    text1 = "uses {text2} literally"
    text2 = "plain"
    token_ids = chat_builder.build_token_ids(text1, text2)
    rendered = chat_builder.tokenizer.decode(token_ids)
    assert (
        rendered
        == f"{BOS} <system> Answer in JSON. <user> First uses {{text2}} literally second plain end <assistant>"
    )


def test_build_token_ids_asks_for_the_generation_prompt(
    chat_builder: llm.PromptBuilder,
) -> None:
    """The chat template is applied with add_generation_prompt=True, so the model starts its own turn."""
    text1 = "one"
    text2 = "two"
    chat_builder.build_token_ids(text1, text2)
    assert chat_builder.tokenizer.chat_calls == [
        {"template": "fake-template", "add_generation_prompt": True},
    ]


def test_build_token_ids_uses_a_given_chat_template(tmp_path: Path) -> None:
    """A given template replaces the tokenizer's own, which a tokenizer without one needs."""
    prompt = llm.load_prompt(write_prompt_file(tmp_path, CHAT_PROMPT))
    tokenizer = FakeTokenizer(chat_template=None)
    builder = llm.PromptBuilder(
        tokenizer=tokenizer,
        prompt=prompt,
        chat_template="given-template",
    )
    text1 = "one"
    text2 = "two"
    builder.build_token_ids(text1, text2)
    assert tokenizer.chat_calls[0]["template"] == "given-template"


def test_build_token_ids_truncates_both_texts(tmp_path: Path) -> None:
    """With a token limit each of the two texts is cut before it goes into the prompt."""
    prompt = llm.load_prompt(write_prompt_file(tmp_path, CHAT_PROMPT))
    builder = llm.PromptBuilder(
        tokenizer=FakeTokenizer(),
        prompt=prompt,
        text_token_limit=1,
    )
    text1 = "one two"
    text2 = "three four"
    token_ids = builder.build_token_ids(text1, text2)
    rendered = builder.tokenizer.decode(token_ids)
    assert (
        rendered
        == f"{BOS} <system> Answer in JSON. <user> First one second three end <assistant>"
    )


def test_truncate_text_keeps_the_first_tokens_and_strips() -> None:
    """A limit keeps the first max_tokens tokens, decoded without special tokens and stripped."""
    tokenizer = FakeTokenizer(decode_prefix=" ")
    text = f"  one {BOS} two three four  "
    max_tokens = 3
    assert llm.truncate_text(tokenizer, text, max_tokens) == "one two"


def test_truncate_text_without_a_limit_keeps_the_text() -> None:
    """Without a limit the text is passed on unchanged, surrounding whitespace included."""
    tokenizer = FakeTokenizer()
    text = "  one two three  "
    max_tokens = None
    assert llm.truncate_text(tokenizer, text, max_tokens) == text


@pytest.mark.parametrize(
    ("text", "expected_pred"),
    [
        (JSON_TRUE, llm.SAME_AUTHOR),
        (JSON_FALSE, llm.DIFFERENT_AUTHOR),
        ('{"analysis": "x", "answer": True}', llm.SAME_AUTHOR),
        ('{"answer":"False"}', llm.DIFFERENT_AUTHOR),
        ('```json\n{"analysis": "x", "answer": false}\n```', llm.DIFFERENT_AUTHOR),
        ('{"analysis": "x", "answer" : true}', llm.SAME_AUTHOR),
        ('{\n  "analysis": "x",\n  "answer": false.\n}', llm.DIFFERENT_AUTHOR),
        ('"answer": true\nThe texts share many phrases.', llm.SAME_AUTHOR),
    ],
)
def test_parse_answer_reads_the_answer_value(text: str, expected_pred: int) -> None:
    """A JSON or Python boolean of the answer key, quoted or not, up to the end of the value."""
    parsed = llm.parse_answer(text)
    assert parsed == llm.ParsedAnswer(pred=expected_pred, status=llm.ANSWERED)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "The texts were written by the same author.",
        '{"analysis": "x", "answer": "yes"}',
        '{"analysis": "x", "Answer": true}',
        '{"analysis": "x", "answer": trueish}',
        '"answer":  A boolean (True/False) answer.',
        '{"analysis": "x", "answer": "True/False"}',
        '{"analysis": "x", "answer": true or false}',
        '{"analysis": "x", "answer": True|False}',
        '{"analysis": "x", "answer": True (the texts share a style)}',
        '"answer": True - the same author.',
        '{"analysis": "x", "answer": "True."}',
        '{"analysis": "x", "answer": "True}',
    ],
)
def test_parse_answer_without_a_boolean_answer_value_is_unanswered(text: str) -> None:
    """Without "answer" followed by a whole boolean the pair has no answer."""
    parsed = llm.parse_answer(text)
    assert parsed == llm.ParsedAnswer(pred=None, status=llm.NO_ANSWER)


def test_parse_answer_with_agreeing_values_is_answered() -> None:
    """The same value twice is one answer."""
    text = f"{JSON_TRUE} and once more {JSON_TRUE}"
    parsed = llm.parse_answer(text)
    assert parsed == llm.ParsedAnswer(pred=llm.SAME_AUTHOR, status=llm.ANSWERED)


def test_parse_answer_with_both_values_is_unanswered() -> None:
    """Two different answer values are not an answer."""
    text = f"{JSON_TRUE} or maybe {JSON_FALSE}"
    parsed = llm.parse_answer(text)
    assert parsed == llm.ParsedAnswer(pred=None, status=llm.CONFLICTING_ANSWERS)


def test_parse_answer_ignores_true_outside_the_answer_value() -> None:
    """A "true" in the reasoning does not decide the answer."""
    text = '{"analysis": "It is true that both use commas, True.", "answer": false}'
    parsed = llm.parse_answer(text)
    assert parsed.pred == llm.DIFFERENT_AUTHOR


def test_check_prompt_lengths_rejects_a_prompt_without_room() -> None:
    """A prompt of max_model_len tokens leaves no room for one generated token."""
    prompt_tokens = pd.Series([10, 16], index=["id-0", "id-1"])
    max_model_len = 16
    with pytest.raises(ValueError, match="id-1"):
        llm.check_prompt_lengths(prompt_tokens, max_model_len)


def test_check_prompt_lengths_accepts_room_for_one_token() -> None:
    """A prompt of max_model_len - 1 tokens can still be answered."""
    prompt_tokens = pd.Series([10, 15], index=["id-0", "id-1"])
    max_model_len = 16
    llm.check_prompt_lengths(prompt_tokens, max_model_len)


def test_count_prompt_tokens_counts_every_pair(chat_builder: llm.PromptBuilder) -> None:
    """The count of a pair is the length of its prompt token ids."""
    pairs_df = make_pairs(2)
    counts = llm.count_prompt_tokens(chat_builder, pairs_df)
    expected = []
    for text1, text2 in pairs_df["pair"]:
        expected.append(len(chat_builder.build_token_ids(text1, text2)))
    assert counts.tolist() == expected
    assert counts.index.tolist() == ["id-0", "id-1"]


@pytest.mark.parametrize(
    "changed_setting",
    [
        "engine",
        "prompt",
        "chat_template",
        "text_token_limit",
        "decoding",
        "chunk_size",
        "pair_order",
    ],
)
def test_describe_run_changes_with_each_setting(
    tmp_path: Path,
    changed_setting: str,
) -> None:
    """Every setting that can change a generated text changes the description."""
    tokenizer = FakeTokenizer()
    prompt = llm.load_prompt(write_prompt_file(tmp_path, CHAT_PROMPT))
    user_only = {"messages": [CHAT_PROMPT["messages"][1]]}
    other_prompt = llm.load_prompt(write_prompt_file(tmp_path, user_only, "other.json"))
    settings = {
        "engine_settings": {"model": "fake"},
        "builder": llm.PromptBuilder(tokenizer=tokenizer, prompt=prompt),
        "decoding": llm.DecodingSettings(temperature=0.0, top_p=1.0, max_tokens=50),
        "chunk_size": 2,
        "pair_ids": ["a", "b", "c"],
    }
    changes = {
        "engine": {"engine_settings": {"model": "fake", "revision": "abc"}},
        "prompt": {
            "builder": llm.PromptBuilder(tokenizer=tokenizer, prompt=other_prompt),
        },
        "chat_template": {
            "builder": llm.PromptBuilder(
                tokenizer=tokenizer,
                prompt=prompt,
                chat_template="other",
            ),
        },
        "text_token_limit": {
            "builder": llm.PromptBuilder(
                tokenizer=tokenizer,
                prompt=prompt,
                text_token_limit=10,
            ),
        },
        "decoding": {
            "decoding": llm.DecodingSettings(temperature=0.0, top_p=1.0, max_tokens=60),
        },
        "chunk_size": {"chunk_size": 3},
        "pair_order": {"pair_ids": ["c", "b", "a"]},
    }
    description = llm.describe_run(**settings)
    changed = llm.describe_run(**{**settings, **changes[changed_setting]})
    assert changed != description
