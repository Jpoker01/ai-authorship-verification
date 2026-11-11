"""Tests of avcore.transformer_training on a tiny random BERT - CPU only, no downloads."""

import pytest

pytestmark = pytest.mark.cerit
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import optuna  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers, processors  # noqa: E402

from avcore import transformer_training  # noqa: E402
from avcore.tuning import open_study, run_study  # noqa: E402

WORDS = [f"w{index}" for index in range(40)]
# 8 pairs / batch size 3 = 3 steps per epoch (the last batch has 2 pairs)
SETTINGS = transformer_training.TrainingSettings(
    learning_rate=1e-3,
    weight_decay=0.01,
    epochs=2,
    batch_size=3,
    warmup_ratio=0.1,
)

@pytest.fixture
def tokenizer():
    """Word-level tokenizer built in memory, "[CLS] a [SEP] b [SEP]" for a pair."""
    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    vocab = {token: index for index, token in enumerate(special + WORDS)}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    # defines how special tokens around words are to be used
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )

@pytest.fixture
def optimizer_steps(monkeypatch):
    """Count the AdamW steps and their learning rates; "crash_at" raises KeyboardInterrupt instead of that step."""
    counter = {"done": 0, "crash_at": None, "learning_rates": []}
    original_step = torch.optim.AdamW.step

    def counted_step(self, *args, **kwargs):
        if counter["done"] + 1 == counter["crash_at"]:
            counter["crash_at"] = None  # one crash only, the restart runs through
            raise KeyboardInterrupt
        counter["done"] += 1
        counter["learning_rates"].append(self.param_groups[0]["lr"])
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", counted_step)
    return counter

def _model(dropout=0.1, num_labels=2):
    """Tiny BERT classifier with the same weights on every call."""
    torch.manual_seed(0)
    config = transformers.BertConfig(
        vocab_size=len(WORDS) + 4,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        num_labels=num_labels,
    )
    # from_pretrained returns the model in eval mode
    return transformers.BertForSequenceClassification(config).eval()

def _pairs(n_pairs, offset=0):
    """Pairs of short random texts, same = True for every second pair."""
    generator = torch.Generator().manual_seed(offset)
    rows = []
    for index in range(n_pairs):
        words = torch.randint(0, len(WORDS), (2, 6), generator=generator).tolist()
        rows.append(
            {
                "id": f"id{offset}-{index}",
                "same": index % 2 == 1,
                "pair": [" ".join(WORDS[word] for word in text) for text in words],
            }
        )
    return pd.DataFrame(rows, columns=["id", "same", "pair"])

def _train(tokenizer, checkpoint_dir, key="final", settings=SETTINGS, train_df=None, dropout=0.1, **kwargs):
    """Train a fresh tiny model, return it and the TrainingResult."""
    model = _model(dropout=dropout)
    result = transformer_training.train_classifier(
        model,
        tokenizer,
        _pairs(8) if train_df is None else train_df,
        settings,
        checkpoint_dir,
        key,
        **kwargs,
    )
    return model, result

def _same_weights(model_a, model_b):
    """True if both models hold exactly the same weights."""
    state_b = model_b.state_dict()
    return all(torch.equal(value, state_b[name]) for name, value in model_a.state_dict().items())

@pytest.mark.parametrize("training", [True, False])
def test_score_pairs_returns_the_softmax_score_and_the_argmax(tokenizer, training):
    """score is the softmax probability of label 1, argmax the label with the higher logit; the mode stays."""
    model = _model()
    # larger random weights than the BERT init, so the pairs get different scores and both labels
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.5)
    pairs_df = _pairs(6, offset=1)
    model.train(training)
    scores_df = transformer_training.score_pairs(model, tokenizer, pairs_df)
    assert model.training == training  # training goes on with dropout after the validation pass

    model.eval()
    with torch.no_grad():
        encoding = tokenizer(list(pairs_df["pair"].str[0]), list(pairs_df["pair"].str[1]), return_tensors="pt", padding=True)
        logits = model(**encoding).logits
    assert list(scores_df.columns) == ["id", "same", "score", "argmax"]
    assert scores_df["id"].tolist() == pairs_df["id"].tolist()
    assert scores_df["score"].tolist() == pytest.approx(torch.softmax(logits, dim=-1)[:, 1].tolist())
    assert scores_df["argmax"].tolist() == logits.argmax(dim=-1).tolist()
    assert set(scores_df["argmax"]) == {0, 1}

def test_train_classifier_follows_the_linear_warmup_schedule(tokenizer, tmp_path, optimizer_steps):
    """Warmup over the first half of the 6 steps from 0, then linear decay - as v1."""
    settings = transformer_training.TrainingSettings(1e-3, 0.01, epochs=2, batch_size=3, warmup_ratio=0.5)
    _train(tokenizer, tmp_path, settings=settings)
    assert optimizer_steps["learning_rates"] == pytest.approx([0, 1e-3 / 3, 2e-3 / 3, 1e-3, 2e-3 / 3, 1e-3 / 3])