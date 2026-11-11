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

def test_train_classifier_reports_the_weighted_f1_at_argmax_of_every_epoch(tokenizer, tmp_path, monkeypatch):
    """Each epoch's pruning F1 is the weighted F1 of that epoch's argmax labels (v1); the last scores are returned."""
    validation_df = _pairs(5, offset=1)  # 3 different and 2 same - the weighted F1 differs from the macro F1
    argmaxes = iter([[1, 1, 0, 0, 0], [0, 1, 0, 1, 1]])
    scored = []

    def fake_score_pairs(model, tokenizer, pairs_df):
        scores_df = pd.DataFrame(
            {
                "id": pairs_df["id"].to_numpy(),
                "same": pairs_df["same"].to_numpy(),
                "score": 0.5,  # not used by the pruning F1
                "argmax": next(argmaxes),
            }
        )
        scored.append(scores_df)
        return scores_df

    monkeypatch.setattr(transformer_training, "score_pairs", fake_score_pairs)
    #executes normal training but returns fake scores from fake_score_pairs
    _, result = _train(tokenizer, tmp_path, validation_df=validation_df)
    labels = validation_df["same"].astype(int)
    assert len(scored) == 2  # scored after every epoch
    assert result.pruning_f1s == [f1_score(labels, df["argmax"], average="weighted") for df in scored]
    assert result.validation_scores_df is scored[-1]


def test_train_classifier_reports_the_mean_loss_of_every_epoch(tokenizer, tmp_path):
    """epoch_losses holds the mean of the batch losses of each epoch, trained in train mode (dropout on)."""
    model = _model()
    batch_losses = []
    original_forward = model.forward

    def recording_forward(*args, **kwargs):
        outputs = original_forward(*args, **kwargs)
        assert model.training
        batch_losses.append(outputs.loss.item())
        return outputs

    model.forward = recording_forward
    # 10 pairs / batch size 3 = 4 steps per epoch, so the mean is not the sum / batch size
    result = transformer_training.train_classifier(model, tokenizer, _pairs(10), SETTINGS, tmp_path, "final")
    assert len(batch_losses) == 8
    assert result.epoch_losses == pytest.approx([sum(batch_losses[:4]) / 4, sum(batch_losses[4:]) / 4])


def test_epoch_order_changes_every_epoch():
    """Every epoch is shuffled anew, as with the v1 DataLoader; the same epoch always the same way."""
    first = transformer_training._epoch_order(20, seed=42, epoch=0)
    assert sorted(first) == list(range(20))
    assert first == transformer_training._epoch_order(20, seed=42, epoch=0)
    assert first != transformer_training._epoch_order(20, seed=42, epoch=1)
    assert first != transformer_training._epoch_order(20, seed=7, epoch=0)

def test_train_classifier_saves_a_checkpoint_only_after_checkpoint_minutes(tokenizer, tmp_path, monkeypatch):
    """Inside an epoch a checkpoint is written once checkpoint_minutes have passed, and after every epoch."""
    clock = {"now": 0.0}
    monkeypatch.setattr(transformer_training.time, "monotonic", lambda: clock["now"])
    saved_steps = []
    original_save = transformer_training._save_checkpoint
    original_step = torch.optim.AdamW.step

    def recording_save(checkpoint_dir, state):
        saved_steps.append((state["epoch"], state["step"]))
        original_save(checkpoint_dir, state)

    def timed_step(self, *args, **kwargs):
        clock["now"] += 40  # every step takes 40 seconds
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", timed_step)
    monkeypatch.setattr(transformer_training, "_save_checkpoint", recording_save)
    # 15 pairs / batch size 3 = 5 steps per epoch; 2 minutes are reached exactly after step 3
    _train(tokenizer, tmp_path, train_df=_pairs(15), checkpoint_minutes=2)
    assert saved_steps == [(0, 3), (1, 0), (1, 3), (2, 0)]

def test_train_classifier_resumes_after_a_crash(tokenizer, tmp_path, optimizer_steps):
    """A crash in the middle of epoch 2 resumes at the saved step and ends with the same weights."""
    reference_model, reference_result = _train(tokenizer, tmp_path / "reference")
    optimizer_steps.update(done=0, crash_at=5)
    with pytest.raises(KeyboardInterrupt):
        _train(tokenizer, tmp_path / "run", checkpoint_minutes=0)
    optimizer_steps["done"] = 0
    resumed_model, resumed_result = _train(tokenizer, tmp_path / "run", checkpoint_minutes=0)
    assert optimizer_steps["done"] == 2  # 6 steps, 4 done before the crash
    assert _same_weights(resumed_model, reference_model)
    assert resumed_result.epoch_losses == reference_result.epoch_losses

def test_finished_run_returns_its_scores_without_training_again(tokenizer,
                                                                tmp_path,
                                                                optimizer_steps):
    """A run repeated after it finished (killed before the trial was stored) trains nothing."""
    validation_df = _pairs(6, offset=1)
    _, first_result = _train(tokenizer, tmp_path,
                             validation_df=validation_df)
    optimizer_steps["done"] = 0
    _, repeated_result = _train(tokenizer, tmp_path,
                                validation_df=validation_df)
    assert optimizer_steps["done"] == 0
    pd.testing.assert_frame_equal(repeated_result.validation_scores_df,
                                  first_result.validation_scores_df)
    assert repeated_result.pruning_f1s == first_result.pruning_f1s

def test_a_kill_during_a_save_keeps_the_previous_checkpoint(tokenizer, tmp_path, monkeypatch, optimizer_steps):
    """A save killed half-way leaves the last complete checkpoint, the run resumes from it."""
    reference_model, _ = _train(tokenizer, tmp_path / "reference")
    original_save = torch.save
    saves = {"n": 0}

    def half_save(obj, file, *args, **kwargs):
        saves["n"] += 1
        if saves["n"] == 2:
            file.write(b"half a checkpoint")
            raise KeyboardInterrupt
        return original_save(obj, file, *args, **kwargs)

    with monkeypatch.context() as patch, pytest.raises(KeyboardInterrupt):
        patch.setattr(torch, "save", half_save)
        _train(tokenizer, tmp_path / "run", checkpoint_minutes=0)
    optimizer_steps["done"] = 0
    resumed_model, _ = _train(tokenizer, tmp_path / "run", checkpoint_minutes=0)
    assert optimizer_steps["done"] == 5  # resumed after step 1, the step of the first save
    assert _same_weights(resumed_model, reference_model)

def test_train_classifier_ignores_a_checkpoint_of_another_run(tokenizer, tmp_path, optimizer_steps):
    """A checkpoint with another key is not loaded - the run trains all its steps."""
    _train(tokenizer, tmp_path, key="position-3")
    optimizer_steps["done"] = 0
    _train(tokenizer, tmp_path, key="position-4")
    assert optimizer_steps["done"] == 6

@pytest.mark.parametrize(
    "changes",
    [
        {"settings": transformer_training.TrainingSettings(2e-3, 0.01, epochs=2, batch_size=3, warmup_ratio=0.1)},
        {"train_df": _pairs(8, offset=9)},
        {"dropout": 0.2},
    ],
)
def test_train_classifier_refuses_a_checkpoint_of_the_same_key_with_other_inputs(tokenizer, tmp_path, changes):
    """The same key with other hyperparameters or other pairs means a changed study - it raises."""
    _train(tokenizer, tmp_path, key="position-3")
    with pytest.raises(ValueError, match="other settings or pairs"):
        _train(tokenizer, tmp_path, key="position-3", **changes)

def test_repeated_trial_resumes_and_reports_its_epochs_again(tokenizer, tmp_path, optimizer_steps):
    """A trial killed in epoch 2 is repeated at its position, resumes and reports both epochs."""
    validation_df = _pairs(4, offset=2)

    def objective(trial):
        trial.suggest_float("x", 0, 1)
        result = transformer_training.train_classifier(
            _model(),
            tokenizer,
            _pairs(8),
            SETTINGS,
            tmp_path / "checkpoints",
            transformer_training.checkpoint_key_for(trial),
            validation_df=validation_df,
            trial=trial,
        )
        return result.pruning_f1s[-1]

    reference_study = open_study(tmp_path / "reference.db", pruner=optuna.pruners.NopPruner())
    run_study(reference_study, objective, n_trials=1)
    optimizer_steps.update(done=0, crash_at=4)  # the first step of epoch 2
    study = open_study(tmp_path / "study.db", pruner=optuna.pruners.NopPruner())
    with pytest.raises(KeyboardInterrupt):
        run_study(study, objective, n_trials=1)
    optimizer_steps["done"] = 0
    study = open_study(tmp_path / "study.db", pruner=optuna.pruners.NopPruner())
    run_study(study, objective, n_trials=1)

    failed, repeated = study.trials
    assert [failed.state.name, repeated.state.name] == ["FAIL", "COMPLETE"]
    assert repeated.params == failed.params
    assert optimizer_steps["done"] == 3  # epoch 2 only
    assert repeated.intermediate_values == reference_study.trials[0].intermediate_values

def test_repeated_trial_gets_the_pruning_decisions_of_an_uninterrupted_study(tokenizer, tmp_path, monkeypatch):
    """Killed before the pruning decision of its last epoch, a repeated trial is still pruned (v1 pruner)."""
    # scripted pruning F1 per position and epoch - position 1 is pruned after epoch 2, its last
    scripted_f1s = {0: [0.5, 0.6], 1: [0.5, 0.4]}
    computed = {0: 0, 1: 0}
    current = {"position": 0}

    def scripted_f1(*args, **kwargs):
        position = current["position"]
        computed[position] += 1
        return scripted_f1s[position][computed[position] - 1]

    def objective(trial):
        current["position"] = trial.user_attrs["position"]
        result = transformer_training.train_classifier(
            _model(),
            tokenizer,
            _pairs(8),
            SETTINGS,
            tmp_path / trial.study.study_name,
            transformer_training.checkpoint_key_for(trial),
            validation_df=_pairs(4, offset=2),
            trial=trial,
        )
        return result.pruning_f1s[-1]

    def pruner():
        return optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=3, min_early_stopping_rate=0)

    monkeypatch.setattr(transformer_training, "f1_score", scripted_f1)
    reference_study = open_study(tmp_path / "reference.db", pruner=pruner())
    run_study(reference_study, objective, n_trials=2)

    computed.update({0: 0, 1: 0})
    study = open_study(tmp_path / "study.db", pruner=pruner())
    original_should_prune = optuna.Trial.should_prune

    def kill_before_the_last_decision_of_position_1(self):
        if self.user_attrs["position"] == 1 and computed[1] == 2:
            raise KeyboardInterrupt
        return original_should_prune(self)

    with monkeypatch.context() as patch, pytest.raises(KeyboardInterrupt):
        patch.setattr(optuna.Trial, "should_prune", kill_before_the_last_decision_of_position_1)
        run_study(study, objective, n_trials=2)
    study = open_study(tmp_path / "study.db", pruner=pruner())
    run_study(study, objective, n_trials=2)

    def summary(trials):
        return [(trial.state.name, trial.intermediate_values) for trial in trials if not trial.user_attrs.get("crashed")]

    assert [state for state, _ in summary(reference_study.trials)] == ["COMPLETE", "PRUNED"]
    assert summary(study.trials) == summary(reference_study.trials)

def test_train_classifier_prunes_after_an_epoch(tokenizer, tmp_path):
    """A pruner that always prunes stops the trial after epoch 1 with its F1 reported."""

    class AlwaysPrune(optuna.pruners.BasePruner):
        def prune(self, study, trial):
            return True

    study = optuna.create_study(pruner=AlwaysPrune(), direction="maximize")
    trial = study.ask()
    trial.set_user_attr("position", 0)
    with pytest.raises(optuna.TrialPruned):
        _train(tokenizer, tmp_path, key="position-0", validation_df=_pairs(4, offset=2), trial=trial)
    assert list(study.trials[0].intermediate_values) == [0]


def _raise_keyboard_interrupt(*args, **kwargs):
    raise KeyboardInterrupt


def test_resumed_trial_asks_the_pruner_before_training_on(tokenizer, tmp_path, optimizer_steps, monkeypatch):
    """A kill between the epoch checkpoint and the pruning decision - the repeated trial is pruned at once."""

    class AlwaysPrune(optuna.pruners.BasePruner):
        def prune(self, study, trial):
            return True

    study = optuna.create_study(pruner=AlwaysPrune(), direction="maximize")
    killed_trial = study.ask()
    killed_trial.set_user_attr("position", 0)
    with monkeypatch.context() as patch, pytest.raises(KeyboardInterrupt):
        patch.setattr(optuna.Trial, "should_prune", _raise_keyboard_interrupt)
        _train(tokenizer, tmp_path, key="position-0", validation_df=_pairs(4, offset=2), trial=killed_trial)
    optimizer_steps["done"] = 0
    repeated_trial = study.ask()
    repeated_trial.set_user_attr("position", 0)
    with pytest.raises(optuna.TrialPruned):
        _train(tokenizer, tmp_path, key="position-0", validation_df=_pairs(4, offset=2), trial=repeated_trial)
    assert optimizer_steps["done"] == 0

