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
        token_ids = []
        if add_special_tokens:
            token_ids.append(self._token_id(BOS))
        for word in text.split():
            token_ids.append(self._token_id(word))
        # as in a Hugging Face tokenizer, the begin token counts towards max_length
        if truncation:
            token_ids = token_ids[:max_length]
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
        # one token per word and the end token, so the token count is not the word count
        token_ids = text.split() + ["</s>"]
        completion = SimpleNamespace(
            text=text,
            token_ids=token_ids,
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
        engine_settings=engine_settings,
        builder=chat_builder,
        decoding=decoding,
        chunk_size=CHUNK_SIZE,
        pair_ids=pair_ids,
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


@pytest.mark.parametrize(
    ("text1", "text2", "expected_user_part"),
    [
        (
            "uses {text2} literally",
            "plain",
            "First uses {text2} literally second plain end",
        ),
        (
            "plain",
            "uses {text1} literally",
            "First plain second uses {text1} literally end",
        ),
    ],
)
def test_build_token_ids_fills_each_text_once(
    chat_builder: llm.PromptBuilder,
    text1: str,
    text2: str,
    expected_user_part: str,
) -> None:
    """A placeholder inside either text is kept as it is - the inserted texts are not searched again."""
    token_ids = chat_builder.build_token_ids(text1, text2)
    rendered = chat_builder.tokenizer.decode(token_ids)
    expected = f"{BOS} <system> Answer in JSON. <user> {expected_user_part} <assistant>"
    assert rendered == expected


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
        "engine_option",
        "max_tokens",
        "temperature",
        "top_p",
        "seed",
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
    prompt_path = write_prompt_file(tmp_path, CHAT_PROMPT)
    prompt = llm.load_prompt(prompt_path)
    engine_settings = {"model": "fake"}
    builder = llm.PromptBuilder(tokenizer=tokenizer, prompt=prompt)
    decoding = llm.DecodingSettings(temperature=0.0, top_p=1.0, max_tokens=50)
    chunk_size = 2
    pair_ids = ["a", "b", "c"]
    description = llm.describe_run(
        engine_settings=engine_settings,
        builder=builder,
        decoding=decoding,
        chunk_size=chunk_size,
        pair_ids=pair_ids,
    )
    # a case name without a branch changes nothing, so the assert below fails for it
    if changed_setting == "engine":
        engine_settings = {"model": "fake", "revision": "abc"}
    elif changed_setting == "engine_option":
        engine_settings = {"model": "fake", "max_num_seqs": 8}
    elif changed_setting == "prompt":
        user_only = {"messages": [CHAT_PROMPT["messages"][1]]}
        other_prompt_path = write_prompt_file(tmp_path, user_only, "other.json")
        other_prompt = llm.load_prompt(other_prompt_path)
        builder = llm.PromptBuilder(tokenizer=tokenizer, prompt=other_prompt)
    elif changed_setting == "chat_template":
        builder = llm.PromptBuilder(
            tokenizer=tokenizer,
            prompt=prompt,
            chat_template="other",
        )
    elif changed_setting == "text_token_limit":
        builder = llm.PromptBuilder(
            tokenizer=tokenizer,
            prompt=prompt,
            text_token_limit=10,
        )
    elif changed_setting == "max_tokens":
        decoding = llm.DecodingSettings(temperature=0.0, top_p=1.0, max_tokens=60)
    elif changed_setting == "temperature":
        decoding = llm.DecodingSettings(temperature=0.7, top_p=1.0, max_tokens=50)
    elif changed_setting == "top_p":
        decoding = llm.DecodingSettings(temperature=0.0, top_p=0.9, max_tokens=50)
    elif changed_setting == "seed":
        decoding = llm.DecodingSettings(
            temperature=0.0,
            top_p=1.0,
            max_tokens=50,
            seed=42,
        )
    elif changed_setting == "chunk_size":
        chunk_size = 3
    elif changed_setting == "pair_order":
        pair_ids = ["c", "b", "a"]
    changed_description = llm.describe_run(
        engine_settings=engine_settings,
        builder=builder,
        decoding=decoding,
        chunk_size=chunk_size,
        pair_ids=pair_ids,
    )
    assert changed_description != description


class TimedTokenizer(FakeTokenizer):
    """Fake tokenizer whose chat template takes one second of a fake clock."""

    def __init__(self, clock: dict[str, float]) -> None:
        super().__init__()
        self.clock = clock

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        chat_template: str | None = None,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        return_dict: bool = True,
    ) -> list[int] | dict[str, list[int]] | str:
        self.clock["now"] += 1.0
        return super().apply_chat_template(
            conversation,
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            return_dict=return_dict,
        )


class TimedEngine(FakeEngine):
    """Fake engine whose generate call takes ten seconds of a fake clock."""

    def __init__(
        self,
        answers: dict[str, str],
        tokenizer: FakeTokenizer,
        clock: dict[str, float],
    ) -> None:
        super().__init__(answers, tokenizer)
        self.clock = clock

    def generate(
        self,
        prompts: list[dict[str, list[int]]],
        sampling_params: object,
        *,
        use_tqdm: bool,
    ) -> list:
        self.clock["now"] += 10.0
        return super().generate(prompts, sampling_params, use_tqdm=use_tqdm)


def make_answers_df(pair_ids: list[str], texts: list[str]) -> pd.DataFrame:
    """Return an answers table as read_answers builds it."""
    preds = []
    statuses = []
    for text in texts:
        parsed = llm.parse_answer(text)
        preds.append(parsed.pred)
        statuses.append(parsed.status)
    return pd.DataFrame(
        {
            "id": pair_ids,
            "prompt_tokens": [5] * len(pair_ids),
            "generated_tokens": [3] * len(pair_ids),
            "finish_reason": ["stop"] * len(pair_ids),
            "text": texts,
            # the nullable integer type of read_answers - an unanswered pair keeps <NA>
            "pred": pd.array(preds, dtype="Int64"),
            "status": statuses,
        },
    )


def make_real_tokenizer() -> object:
    """Return a tiny real Hugging Face tokenizer without a chat template (skips without transformers)."""
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    words = [
        "<unk>",
        BOS,
        "</s>",
        "[INST]",
        "[/INST]",
        "<<SYS>>",
        "<</SYS>>",
        "Answer",
        "one",
        "two",
    ]
    vocabulary = {}
    for index, word in enumerate(words):
        vocabulary[word] = index
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocabulary, unk_token="<unk>"),
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    # like the Llama-2 tokenizer, put <s> (BOS, id 1) before every encoded text ($A)
    backend.post_processor = tokenizers.processors.TemplateProcessing(
        single=f"{BOS} $A",
        special_tokens=[(BOS, 1)],
    )
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token=BOS,
        eos_token="</s>",
        unk_token="<unk>",
    )


def test_prepare_answers_dir_stores_the_run_description(
    tmp_path: Path,
    run_description: dict,
) -> None:
    """A new folder gets the run description."""
    answers_dir = tmp_path / "answers"
    llm.prepare_answers_dir(answers_dir, run_description)
    stored = json.loads((answers_dir / llm.RUN_FILE).read_text(encoding="utf-8"))
    assert stored == run_description


def test_prepare_answers_dir_names_a_changed_setting_and_changes_nothing(
    tmp_path: Path,
    run_description: dict,
) -> None:
    """Another setting is refused, and the folder keeps every file, a leftover partial file too."""
    answers_dir = tmp_path / "answers"
    llm.prepare_answers_dir(answers_dir, run_description)
    leftover_path = answers_dir / f"chunk-00000.json{llm.PARTIAL_SUFFIX}"
    leftover_path.write_text("half", encoding="utf-8")
    changed_description = dict(run_description)
    changed_description["chunk_size"] = 3
    with pytest.raises(RuntimeError, match="chunk_size"):
        llm.prepare_answers_dir(answers_dir, changed_description)
    assert leftover_path.is_file()


def test_prepare_answers_dir_removes_a_leftover_partial_file(
    tmp_path: Path,
    run_description: dict,
) -> None:
    """A restart of the same run removes what a killed write left behind."""
    answers_dir = tmp_path / "answers"
    llm.prepare_answers_dir(answers_dir, run_description)
    leftover_path = answers_dir / f"chunk-00001.json{llm.PARTIAL_SUFFIX}"
    leftover_path.write_text("half", encoding="utf-8")
    llm.prepare_answers_dir(answers_dir, run_description)
    assert not leftover_path.exists()


def test_prepare_answers_dir_accepts_a_restart_with_a_tuple_setting(
    tmp_path: Path,
    run_description: dict,
) -> None:
    """A tuple setting is stored as a JSON list, and a restart with the same tuple is accepted."""
    answers_dir = tmp_path / "answers"
    description = dict(run_description)
    description["engine_settings"] = {"model": "fake", "stop": ("a", "b")}
    llm.prepare_answers_dir(answers_dir, description)
    llm.prepare_answers_dir(answers_dir, description)
    stored = json.loads((answers_dir / llm.RUN_FILE).read_text(encoding="utf-8"))
    assert stored["engine_settings"]["stop"] == ["a", "b"]


def test_prepare_answers_dir_refuses_chunk_files_without_a_run_description(
    tmp_path: Path,
    run_description: dict,
) -> None:
    """Chunk files whose run.json is gone belong to unknown settings and are not continued."""
    answers_dir = tmp_path / "answers"
    answers_dir.mkdir()
    chunk_path = answers_dir / "chunk-00000.json"
    chunk_path.write_text(
        '{"chunk": 0, "seconds": 1.0, "answers": []}',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="no run.json"):
        llm.prepare_answers_dir(answers_dir, run_description)
    assert not (answers_dir / llm.RUN_FILE).exists()


def test_split_chunks_keeps_the_order_and_a_short_last_chunk() -> None:
    """Consecutive chunks of chunk_size ids, the last one shorter."""
    pair_ids = ["a", "b", "c", "d", "e"]
    chunk_size = 2
    assert llm.split_chunks(pair_ids, chunk_size) == [["a", "b"], ["c", "d"], ["e"]]


def test_split_chunks_rejects_a_repeated_id() -> None:
    """A repeated id would give two pairs the prompt of one of them."""
    pair_ids = ["a", "b", "a"]
    chunk_size = 2
    with pytest.raises(ValueError, match="more than once"):
        llm.split_chunks(pair_ids, chunk_size)


def test_split_chunks_rejects_no_ids() -> None:
    """A run without pairs stops before anything is generated."""
    pair_ids = []
    chunk_size = 2
    with pytest.raises(ValueError, match="no pair ids"):
        llm.split_chunks(pair_ids, chunk_size)


def test_split_chunks_with_a_chunk_larger_than_the_pairs_gives_one_chunk() -> None:
    """Fewer pairs than chunk_size are one chunk."""
    pair_ids = ["a", "b", "c"]
    chunk_size = 256
    assert llm.split_chunks(pair_ids, chunk_size) == [["a", "b", "c"]]


def test_generate_answers_writes_one_file_per_chunk(
    tmp_path: Path,
    chat_builder: llm.PromptBuilder,
    run_description: dict,
) -> None:
    """Four pairs in chunks of two are two engine calls and two chunk files."""
    pairs_df = make_pairs(4)
    engine = FakeEngine(make_engine_answers(4), chat_builder.tokenizer)
    answers_dir = tmp_path / "answers"
    llm.prepare_answers_dir(answers_dir, run_description)
    generated = llm.generate_answers(
        engine,
        SAMPLING_PARAMS,
        chat_builder,
        pairs_df,
        answers_dir,
        CHUNK_SIZE,
    )
    assert generated == 2
    assert engine.prompts_per_call == [2, 2]
    assert llm.find_finished_chunks(answers_dir) == {0, 1}


def test_generate_answers_generates_only_the_missing_chunks(
    tmp_path: Path,
    chat_builder: llm.PromptBuilder,
    run_description: dict,
) -> None:
    """After the engine died in the second chunk, the restart generates only that chunk."""
    pairs_df = make_pairs(4)
    answers = make_engine_answers(4)
    answers_dir = tmp_path / "answers"
    llm.prepare_answers_dir(answers_dir, run_description)
    failing_engine = FakeEngine(answers, chat_builder.tokenizer, fail_at_call=2)
    with pytest.raises(RuntimeError, match="died"):
        llm.generate_answers(
            failing_engine,
            SAMPLING_PARAMS,
            chat_builder,
            pairs_df,
            answers_dir,
            CHUNK_SIZE,
        )
    restarted_engine = FakeEngine(answers, chat_builder.tokenizer)
    generated = llm.generate_answers(
        restarted_engine,
        SAMPLING_PARAMS,
        chat_builder,
        pairs_df,
        answers_dir,
        CHUNK_SIZE,
    )
    assert generated == 1
    assert restarted_engine.prompts_per_call == [2]


def test_generate_answers_after_a_finished_run_calls_no_engine(
    tmp_path: Path,
    chat_builder: llm.PromptBuilder,
    run_description: dict,
) -> None:
    """A complete answers folder makes no engine call."""
    pairs_df = make_pairs(4)
    answers = make_engine_answers(4)
    answers_dir = tmp_path / "answers"
    llm.prepare_answers_dir(answers_dir, run_description)
    first_engine = FakeEngine(answers, chat_builder.tokenizer)
    llm.generate_answers(
        first_engine,
        SAMPLING_PARAMS,
        chat_builder,
        pairs_df,
        answers_dir,
        CHUNK_SIZE,
    )
    engine = FakeEngine(answers, chat_builder.tokenizer)
    generated = llm.generate_answers(
        engine,
        SAMPLING_PARAMS,
        chat_builder,
        pairs_df,
        answers_dir,
        CHUNK_SIZE,
    )
    assert generated == 0
    assert engine.prompts_per_call == []


def test_generate_answers_rejects_outputs_of_other_prompts(
    tmp_path: Path,
    chat_builder: llm.PromptBuilder,
    run_description: dict,
) -> None:
    """Outputs in another order than the prompts stop the run before the chunk is written."""
    pairs_df = make_pairs(4)
    engine = FakeEngine(make_engine_answers(4), chat_builder.tokenizer)
    engine.reverse_outputs = True
    answers_dir = tmp_path / "answers"
    llm.prepare_answers_dir(answers_dir, run_description)
    with pytest.raises(RuntimeError, match="another prompt"):
        llm.generate_answers(
            engine,
            SAMPLING_PARAMS,
            chat_builder,
            pairs_df,
            answers_dir,
            CHUNK_SIZE,
        )
    assert llm.find_finished_chunks(answers_dir) == set()


def test_generate_answers_rejects_a_missing_output(
    tmp_path: Path,
    chat_builder: llm.PromptBuilder,
    run_description: dict,
) -> None:
    """An engine that returns fewer outputs than prompts stops the run before the chunk is written."""
    pairs_df = make_pairs(4)
    engine = FakeEngine(make_engine_answers(4), chat_builder.tokenizer)
    engine.drop_last_output = True
    answers_dir = tmp_path / "answers"
    llm.prepare_answers_dir(answers_dir, run_description)
    with pytest.raises(RuntimeError, match="outputs for"):
        llm.generate_answers(
            engine,
            SAMPLING_PARAMS,
            chat_builder,
            pairs_df,
            answers_dir,
            CHUNK_SIZE,
        )
    assert llm.find_finished_chunks(answers_dir) == set()


def test_generate_answers_leaves_no_chunk_file_when_the_write_fails(
    tmp_path: Path,
    chat_builder: llm.PromptBuilder,
    run_description: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A write that fails before its rename leaves only the temporary file, never a chunk file."""
    pairs_df = make_pairs(4)
    engine = FakeEngine(make_engine_answers(4), chat_builder.tokenizer)
    answers_dir = tmp_path / "answers"
    llm.prepare_answers_dir(answers_dir, run_description)

    def fail_to_replace(source: Path, target: Path) -> None:
        raise OSError("no space left on the device")

    monkeypatch.setattr(llm.os, "replace", fail_to_replace)
    with pytest.raises(OSError, match="no space"):
        llm.generate_answers(
            engine,
            SAMPLING_PARAMS,
            chat_builder,
            pairs_df,
            answers_dir,
            CHUNK_SIZE,
        )
    assert llm.find_finished_chunks(answers_dir) == set()
    assert list(answers_dir.glob(f"*{llm.PARTIAL_SUFFIX}")) != []


def test_generate_answers_times_only_the_engine_call(
    tmp_path: Path,
    run_description: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The seconds of a chunk are the time of its engine call, not of building its prompts."""
    clock = {"now": 0.0}
    tokenizer = TimedTokenizer(clock)
    prompt = llm.load_prompt(write_prompt_file(tmp_path, CHAT_PROMPT))
    builder = llm.PromptBuilder(tokenizer=tokenizer, prompt=prompt)
    engine = TimedEngine(make_engine_answers(4), tokenizer, clock)
    pairs_df = make_pairs(4)
    answers_dir = tmp_path / "answers"
    chunk_size = 2
    llm.prepare_answers_dir(answers_dir, run_description)

    def read_fake_clock() -> float:
        return clock["now"]

    monkeypatch.setattr(llm.time, "monotonic", read_fake_clock)
    llm.generate_answers(
        engine,
        SAMPLING_PARAMS,
        builder,
        pairs_df,
        answers_dir,
        chunk_size,
    )
    chunk = json.loads((answers_dir / "chunk-00000.json").read_text(encoding="utf-8"))
    assert chunk["seconds"] == 10.0


def test_read_answers_parses_every_pair_in_order(
    tmp_path: Path,
    chat_builder: llm.PromptBuilder,
    run_description: dict,
) -> None:
    """The answers come back in pair order with their parsed prediction."""
    pairs_df = make_pairs(4)
    answers_dir = tmp_path / "answers"
    llm.prepare_answers_dir(answers_dir, run_description)
    engine = FakeEngine(make_engine_answers(4), chat_builder.tokenizer)
    llm.generate_answers(
        engine,
        SAMPLING_PARAMS,
        chat_builder,
        pairs_df,
        answers_dir,
        CHUNK_SIZE,
    )
    pair_ids = pairs_df["id"].tolist()
    generated = llm.read_answers(answers_dir, pair_ids, CHUNK_SIZE)
    assert generated.answers_df["id"].tolist() == pairs_df["id"].tolist()
    assert generated.answers_df["pred"].tolist() == [1, 0, 1, 0]
    assert generated.answers_df["finish_reason"].tolist() == ["stop"] * 4
    # the words of each answer and the end token of the fake engine
    expected_generated_tokens = [
        len(JSON_TRUE.split()) + 1,
        len(JSON_FALSE.split()) + 1,
    ]
    assert (
        generated.answers_df["generated_tokens"].tolist()
        == expected_generated_tokens * 2
    )
    expected_prompt_tokens = llm.count_prompt_tokens(chat_builder, pairs_df).tolist()
    assert generated.answers_df["prompt_tokens"].tolist() == expected_prompt_tokens


def test_read_answers_raises_for_a_missing_chunk(
    tmp_path: Path,
    chat_builder: llm.PromptBuilder,
    run_description: dict,
) -> None:
    """An unfinished generation can not be read."""
    pairs_df = make_pairs(4)
    answers_dir = tmp_path / "answers"
    llm.prepare_answers_dir(answers_dir, run_description)
    pair_ids = pairs_df["id"].tolist()
    engine = FakeEngine(make_engine_answers(4), chat_builder.tokenizer, fail_at_call=2)
    with pytest.raises(RuntimeError, match="died"):
        llm.generate_answers(
            engine,
            SAMPLING_PARAMS,
            chat_builder,
            pairs_df,
            answers_dir,
            CHUNK_SIZE,
        )
    with pytest.raises(RuntimeError, match="missing"):
        llm.read_answers(answers_dir, pair_ids, CHUNK_SIZE)


def test_read_answers_rejects_chunks_of_other_pairs(
    tmp_path: Path,
    chat_builder: llm.PromptBuilder,
    run_description: dict,
) -> None:
    """An answers folder is not read for pairs it holds no answers of."""
    pairs_df = make_pairs(4)
    answers_dir = tmp_path / "answers"
    llm.prepare_answers_dir(answers_dir, run_description)
    engine = FakeEngine(make_engine_answers(4), chat_builder.tokenizer)
    llm.generate_answers(
        engine,
        SAMPLING_PARAMS,
        chat_builder,
        pairs_df,
        answers_dir,
        CHUNK_SIZE,
    )
    # the folder holds the answers of id-0 to id-3 only
    other_pair_ids = ["other-0", "other-1", "other-2", "other-3"]
    with pytest.raises(RuntimeError, match="other pairs"):
        llm.read_answers(answers_dir, other_pair_ids, CHUNK_SIZE)


def test_read_answers_sums_the_seconds_of_every_chunk(tmp_path: Path) -> None:
    """The generation time is the sum of the stored chunk times."""
    answers_dir = tmp_path / "answers"
    answers_dir.mkdir()
    pair_ids = ["a", "b"]
    chunk_seconds = [1.5, 2.25]
    for index in range(len(pair_ids)):
        record = {
            "id": pair_ids[index],
            "prompt_tokens": 3,
            "generated_tokens": 2,
            "finish_reason": "stop",
            "text": JSON_TRUE,
        }
        chunk = {"chunk": index, "seconds": chunk_seconds[index], "answers": [record]}
        chunk_path = answers_dir / f"chunk-{index:05d}.json"
        chunk_path.write_text(json.dumps(chunk), encoding="utf-8")
    chunk_size = 1
    generated = llm.read_answers(answers_dir, pair_ids, chunk_size)
    assert generated.generation_seconds == 3.75


@pytest.mark.parametrize(
    ("unanswered_pred", "unanswered_score", "expected_pred", "expected_score"),
    [(0, 0.5, [1, 0, 0], [1.0, 0.0, 0.5]), (1, 0.0, [1, 0, 1], [1.0, 0.0, 0.0])],
)
def test_build_prediction_tables_applies_the_unanswered_rule(
    unanswered_pred: int,
    unanswered_score: float,
    expected_pred: list[int],
    expected_score: list[float],
) -> None:
    """An unanswered pair gets the given pred and score, an answered one 1,0 or 0,0."""
    pairs_df = pd.DataFrame({"id": ["a", "b", "c"], "same": [True, False, True]})
    answers_df = make_answers_df(["a", "b", "c"], [JSON_TRUE, JSON_FALSE, "no idea"])
    tables = llm.build_prediction_tables(
        pairs_df,
        answers_df,
        unanswered_pred,
        unanswered_score,
    )
    assert tables.predictions_df["pred"].tolist() == expected_pred
    assert tables.predictions_df["score"].tolist() == expected_score
    assert tables.unanswered_df["id"].tolist() == ["c"]


def test_build_prediction_tables_keeps_an_empty_unanswered_table() -> None:
    """Without unanswered pairs the table exists with its columns and no row."""
    pairs_df = pd.DataFrame({"id": ["a"], "same": [True]})
    answers_df = make_answers_df(["a"], [JSON_TRUE])
    tables = llm.build_prediction_tables(
        pairs_df,
        answers_df,
        UNANSWERED_PRED,
        UNANSWERED_SCORE,
    )
    assert tables.unanswered_df.empty
    assert tables.unanswered_df.columns.tolist() == llm.UNANSWERED_COLUMNS


def test_build_prediction_tables_keep_no_text_in_the_generation_table() -> None:
    """The generation table holds token counts and short codes, never the generated text."""
    pairs_df = pd.DataFrame({"id": ["a"], "same": [True]})
    answers_df = make_answers_df(["a"], [JSON_TRUE])
    tables = llm.build_prediction_tables(
        pairs_df,
        answers_df,
        UNANSWERED_PRED,
        UNANSWERED_SCORE,
    )
    assert tables.generation_df.columns.tolist() == [
        "id",
        "prompt_tokens",
        "generated_tokens",
        "finish_reason",
        "status",
    ]


def test_build_prediction_tables_names_the_unanswered_columns() -> None:
    """The unanswered table has the columns id, status and finish_reason, in this order."""
    pairs_df = pd.DataFrame({"id": ["a"], "same": [True]})
    answers_df = make_answers_df(["a"], ["no idea"])
    tables = llm.build_prediction_tables(
        pairs_df,
        answers_df,
        UNANSWERED_PRED,
        UNANSWERED_SCORE,
    )
    assert tables.unanswered_df.columns.tolist() == ["id", "status", "finish_reason"]
    assert tables.unanswered_df["status"].tolist() == [llm.NO_ANSWER]


def test_build_prediction_tables_rejects_answers_of_other_pairs() -> None:
    """Answers in another order are refused - the tables match answers to pairs by row position."""
    pairs_df = pd.DataFrame({"id": ["a", "b"], "same": [True, False]})
    answers_df = make_answers_df(["b", "a"], [JSON_TRUE, JSON_FALSE])
    with pytest.raises(ValueError, match="every pair once"):
        llm.build_prediction_tables(
            pairs_df,
            answers_df,
            UNANSWERED_PRED,
            UNANSWERED_SCORE,
        )


@pytest.mark.parametrize(
    ("unanswered_pred", "unanswered_score"),
    [(2, 0.5), (0, float("nan"))],
)
def test_build_prediction_tables_rejects_a_bad_unanswered_rule(
    unanswered_pred: int,
    unanswered_score: float,
) -> None:
    """The unanswered rule has to be a 0 / 1 prediction with a finite score."""
    pairs_df = pd.DataFrame({"id": ["a"], "same": [True]})
    answers_df = make_answers_df(["a"], ["no idea"])
    with pytest.raises(ValueError, match="Unanswered"):
        llm.build_prediction_tables(
            pairs_df,
            answers_df,
            unanswered_pred,
            unanswered_score,
        )


def test_build_prediction_tables_pass_the_checks_of_the_run_folder(
    tmp_path: Path,
) -> None:
    """write_run_files accepts the three tables as they are."""
    pairs_df = pd.DataFrame({"id": ["a", "b"], "same": [True, False]})
    answers_df = make_answers_df(["a", "b"], [JSON_TRUE, "no idea"])
    tables = llm.build_prediction_tables(
        pairs_df,
        answers_df,
        UNANSWERED_PRED,
        UNANSWERED_SCORE,
    )
    context = tracking.RunContext(environment={"commit": "abc"}, packages={})
    run_dir = tmp_path / "run_v2"
    tracking.write_run_files(
        run_dir,
        context=context,
        params={},
        metrics={},
        tables={
            "predictions-test": tables.predictions_df,
            "unanswered-test": tables.unanswered_df,
            "generation-test": tables.generation_df,
        },
        figures={},
    )
    assert (run_dir / "unanswered-test.csv").is_file()


def test_real_chat_template_gives_one_begin_token(tmp_path: Path) -> None:
    """The template's own begin token is the only one - the rendered ids get no second one."""
    tokenizer = make_real_tokenizer()
    chat_template = (PROMPT_DIR / LLAMA2_TEMPLATE_FILE).read_text(encoding="utf-8")
    prompt = llm.load_prompt(
        write_prompt_file(
            tmp_path,
            {
                "messages": [
                    {"role": "system", "content": "Answer"},
                    {"role": "user", "content": "{text1} {text2}"},
                ],
            },
        ),
    )
    builder = llm.PromptBuilder(
        tokenizer=tokenizer,
        prompt=prompt,
        chat_template=chat_template,
    )
    text1 = "one"
    text2 = "two"
    token_ids = builder.build_token_ids(text1, text2)
    assert tokenizer.convert_ids_to_tokens(token_ids) == [
        BOS,
        "[INST]",
        "<<SYS>>",
        "Answer",
        "<</SYS>>",
        "one",
        "two",
        "[/INST]",
    ]


def test_real_tokenizer_without_a_template_raises(tmp_path: Path) -> None:
    """A tokenizer without a chat template and no given template stops the prompt building."""
    tokenizer = make_real_tokenizer()
    prompt = llm.load_prompt(write_prompt_file(tmp_path, CHAT_PROMPT))
    builder = llm.PromptBuilder(tokenizer=tokenizer, prompt=prompt)
    text1 = "one"
    text2 = "two"
    with pytest.raises(ValueError, match="chat_template"):
        builder.build_token_ids(text1, text2)
