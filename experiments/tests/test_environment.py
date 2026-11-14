"""Tests that the transformer environment loads the tokenizers of the three models."""

import pytest

transformers = pytest.importorskip("transformers")

pytestmark = pytest.mark.cerit


@pytest.mark.parametrize(
    "model_id",
    [
        "allenai/longformer-base-4096",
        "google/bigbird-roberta-base",
        "answerdotai/ModernBERT-large",
    ],
)
def test_tokenizer_loads_as_fast_tokenizer(model_id: str) -> None:
    """The tokenizer of every notebook model loads as a fast tokenizer and encodes a text pair."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    encoding = tokenizer(["first text"], ["second text"])

    assert tokenizer.is_fast
    assert len(encoding["input_ids"][0]) > 0
    assert encoding["input_ids"][0].count(tokenizer.sep_token_id) >= 2

def test_bigbird_tokenizer_gives_the_expected_ids() -> None:
    """The BigBird tokenizer converted from spiece.model gives the ids of the pinned environment."""
    tokenizer = transformers.AutoTokenizer.from_pretrained("google/bigbird-roberta-base")

    encoding = tokenizer(["first text"], ["second text"])

    # [CLS] ▁first ▁text [SEP] ▁second ▁text [SEP]
    assert encoding["input_ids"][0] == [65, 818, 2521, 66, 1319, 2521, 66]

