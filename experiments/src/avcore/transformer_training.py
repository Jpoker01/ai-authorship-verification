"""Training and scoring of the transformer classification heads, resumable after a crash.

A Kubernetes Job pod that is restarted runs the notebook from the top again. The trial that
was running is repeated with the same hyperparameters (tuning.py), and train_classifier
continues it from its last checkpoint instead of from the first step.
"""

import hashlib
import json
import logging
import math
import os
import pickle
import struct
import time
import zipfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.optimization import get_linear_schedule_with_warmup

from avcore.data import RANDOM_STATE

logger = logging.getLogger(__name__)

# v1 limit of all three transformers (ModernBERT could take 8 192)
MAX_LENGTH = 4096
CHECKPOINT_FILE = "checkpoint.pt"
CHECKPOINT_MINUTES = 60
TRAINING_SECONDS_ATTR = "training_seconds"


@dataclass(frozen=True)
class TrainingSettings:
    """Hyperparameters of one training run (one Optuna trial or the final training).

    Attributes:
        learning_rate: Peak learning rate of AdamW, reached after the warmup.
        weight_decay: Weight decay of AdamW.
        epochs: Number of passes over the training pairs.
        batch_size: Number of pairs per training step.
        warmup_ratio: Share of all steps in which the learning rate rises linearly.
        seed: Seed of the order of the pairs in every epoch.
    """

    learning_rate: float
    weight_decay: float
    epochs: int
    batch_size: int
    warmup_ratio: float
    seed: int = RANDOM_STATE


@dataclass(frozen=True)
class TrainingResult:
    """What a training run reports.

    Attributes:
        epoch_losses: Mean training loss of every epoch.
        pruning_f1s: Weighted F1 at argmax on validation after every epoch (empty without validation).
        validation_scores_df: score_pairs output of the last epoch, or None without validation.
        training_seconds: Seconds of the training steps and of the validation scoring after each
            epoch up to the last checkpoint, summed over restarts - without loading the model or
            the checkpoint, the saves and the final re-scoring.
    """

    epoch_losses: list[float]
    pruning_f1s: list[float]
    validation_scores_df: pd.DataFrame | None
    training_seconds: float


def checkpoint_key_for(trial: optuna.Trial) -> str:
    """Name the checkpoint of a trial by its study and position, which a repeated trial keeps.

    Raises:
        ValueError: If the trial has no position, i.e. it was not started by tuning.run_study.
    """
    if "position" not in trial.user_attrs:
        raise ValueError(
            "The trial has no position - start the study with tuning.run_study.",
        )
    # trial.number changes when a crashed trial is repeated, the position does not
    return f"{trial.study.study_name}-position-{trial.user_attrs['position']}"


def _encode(
    tokenizer: PreTrainedTokenizerBase,
    pairs: Sequence[Sequence[str]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Tokenize text pairs as one padded batch, truncated to MAX_LENGTH tokens."""
    encoding = tokenizer(
        [pair[0] for pair in pairs],
        [pair[1] for pair in pairs],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    return encoding.to(device)


def score_pairs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    pairs_df: pd.DataFrame,
) -> pd.DataFrame:
    """Score every pair alone, as v1 did, without gradients.

    Args:
        model: Classifier with 2 labels, 0 = different author, 1 = same author.
        tokenizer: Tokenizer of the model.
        pairs_df: Rows with id, same and pair (the two texts).

    Returns:
        One row per pair, in order, with id, same, score (softmax probability of "same author")
        and argmax (the label with the higher logit, the v1 pruning prediction).
    """
    device = next(model.parameters()).device
    was_training = model.training
    # None for BERT, Longformer and ModernBERT - BigBird switches to full attention for good at a short input
    configured_attention = getattr(model.config, "attention_type", None)
    attention_before = getattr(model.base_model, "attention_type", None)
    # a switch of the BigBird attention builds new layers, and their initialisation draws from this RNG
    rng_state = torch.get_rng_state()
    model.eval()
    with torch.inference_mode():
        # the logits are read once at the end - reading them after every pair would make the
        # CPU wait for the GPU instead of tokenizing the next pair in the meantime
        logits = torch.empty((len(pairs_df), 2), device=device)
        for row, pair in enumerate(pairs_df["pair"]):
            if configured_attention is not None:
                # every pair in the configured attention, as the saved model scores it - a score
                # depends only on the weights and its own pair, not on the pairs before it
                model.base_model.set_attention_type(configured_attention)
            logits[row] = model(**_encode(tokenizer, [pair], device)).logits[0]
        scores = F.softmax(logits, dim=-1)[:, 1].tolist()
        argmaxes = logits.argmax(dim=-1).tolist()
        if attention_before is not None:
            # the training goes on in the attention it had before the scoring
            model.base_model.set_attention_type(attention_before)
            # the dropout of the training goes on as if no pair had been scored
        torch.set_rng_state(rng_state)
    model.train(was_training)
    return pd.DataFrame(
        {
            "id": pairs_df["id"].to_numpy(),
            "same": pairs_df["same"].to_numpy(),
            "score": scores,
            "argmax": argmaxes,
        },
    )


def _epoch_order(n_pairs: int, seed: int, epoch: int) -> list[int]:
    """Return the shuffled row order of an epoch, the same for the same seed and epoch."""
    generator = torch.Generator().manual_seed(seed + epoch)
    return torch.randperm(n_pairs, generator=generator).tolist()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_pairs(pairs_df: pd.DataFrame) -> str:
    """Hash the id, label and both texts of every pair, in order."""
    digest = hashlib.sha256()
    for pair_id, same, pair in pairs_df[["id", "same", "pair"]].itertuples(index=False):
        digest.update(
            json.dumps([str(pair_id), bool(same), list(pair)]).encode("utf-8"),
        )
    return digest.hexdigest()


def _describe_run(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame | None,
    settings: TrainingSettings,
    trial: optuna.Trial | None,
) -> dict:
    """Return everything that decides the trained weights - a checkpoint resumes only an equal run."""
    run = {
        **asdict(settings),
        "model": model.name_or_path,
        # the vocabulary, not the folder it was loaded from (another path in every pod)
        "tokenizer_vocab_sha256": _sha256(
            json.dumps(tokenizer.get_vocab(), sort_keys=True)),
        "truncation_side": tokenizer.truncation_side,
        "padding_side": tokenizer.padding_side,
        "max_length": MAX_LENGTH,
        # holds the dropout too, when it is set through from_pretrained
        "model_config_sha256": _sha256(model.config.to_json_string()),
        # every submodule with its settings, incl. the p of every nn.Dropout (also one set after loading)
        "model_modules_sha256": _sha256(repr(model)),
        "train_pairs": len(train_df),
        # the labels and texts too - a rebuilt file with the same ids is another run
        "train_pairs_sha256": _sha256_pairs(train_df),
        # the replayed pruning F1 values come from the validation pairs
        "validation_pairs_sha256": None
        if validation_df is None
        else _sha256_pairs(validation_df),
        "trial_params": {} if trial is None else dict(trial.params),
    }
    try:
        # plain Python values only - torch.load(weights_only=True) refuses a checkpoint with a
        # numpy number in it, which torch.save writes without an error
        return json.loads(json.dumps(run, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"The trial params have to be plain Python values: {error}") from error


def _save_checkpoint(checkpoint_dir: Path, state: dict) -> None:
    """Remove leftover temporary files, then write the state and replace the old checkpoint in one step."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # one process per folder, so a temporary file found here was left by a killed save
    for stale_path in checkpoint_dir.glob(f"{CHECKPOINT_FILE}.*.partial"):
        stale_path.unlink()
    partial_path = checkpoint_dir / f"{CHECKPOINT_FILE}.{os.getpid()}.partial"
    with partial_path.open("wb") as file:
        torch.save(state, file)
        file.flush()
        # on the disk before the rename, also when the node crashes
        os.fsync(file.fileno())
    # a kill during torch.save leaves the previous checkpoint intact
    os.replace(partial_path, checkpoint_dir / CHECKPOINT_FILE)


def _load_checkpoint(checkpoint_dir: Path, key: str, run: dict) -> dict | None:
    """Return the checkpoint of this run, None if there is none or it belongs to another run.

    Raises:
        ValueError: If the checkpoint can not be read, or has this key but describes another run.
    """
    checkpoint_path = checkpoint_dir / CHECKPOINT_FILE
    if not checkpoint_path.is_file():
        return None
    try:
        # torch.save always writes a zip - opening anything else as one raises BadZipFile
        with zipfile.ZipFile(checkpoint_path):
            pass
        # on the CPU - torch.set_rng_state takes only CPU tensors
        state = torch.load(checkpoint_path, map_location="cpu",
                           weights_only=True)
        # a damaged pickle can load as another object - no dict, or a dict without these entries
        state_key, state_run = state["key"], state["run"]
        # an OSError (disk, permissions) is not caught - the file may be fine, "delete it" would be wrong
    except (
            RuntimeError,
            KeyError,
            EOFError,
            AttributeError,
            ValueError,
            AssertionError,
            struct.error,
            IndexError,
            TypeError,
            pickle.UnpicklingError,
            zipfile.BadZipFile,
    ) as error:
        raise ValueError(
            f"{checkpoint_path} can not be read - delete it: {error}",
        ) from error


@dataclass
class _Progress:
    """Where a training run stands - saved in every checkpoint next to the weights."""

    epoch: int = 0
    step: int = 0
    epoch_loss: float = 0.0
    epoch_losses: list[float] = field(default_factory=list)
    pruning_f1s: list[float] = field(default_factory=list)
    training_seconds: float = 0.0


def _check_inputs(
    model: PreTrainedModel,
    train_df: pd.DataFrame,
    settings: TrainingSettings,
    validation_df: pd.DataFrame | None,
    trial: optuna.Trial | None,
) -> None:
    """Raise a ValueError for an input that can not be trained on."""
    # torch.save writes a numpy number (also into the optimizer state), but
    # torch.load(weights_only=True) then refuses the whole checkpoint
    if any(type(value) not in (int, float) for value in
           asdict(settings).values()):
        raise ValueError(
            f"The settings have to be plain Python numbers: {settings}")
    if train_df.empty:
        raise ValueError("train_df is empty.")
    if settings.epochs < 1 or settings.batch_size < 1:
        raise ValueError(f"epochs and batch_size have to be at least 1: {settings}")
    if validation_df is not None and validation_df.empty:
        raise ValueError("validation_df is empty.")
    if trial is not None and validation_df is None:
        raise ValueError("Pruning needs validation_df.")
    if model.config.num_labels != 2:
        raise ValueError(f"The model has {model.config.num_labels} labels, not 2.")


def _build_optimizer_and_scheduler(
    model: PreTrainedModel,
    settings: TrainingSettings,
    steps_per_epoch: int,
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR]:
    """Return AdamW and its linear schedule with warmup over all steps of the run."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    total_steps = steps_per_epoch * settings.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * settings.warmup_ratio),
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def _collect_checkpoint_state(
    key: str,
    run: dict,
    model: PreTrainedModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    progress: _Progress,
) -> dict:
    """Collect everything a killed run needs to continue as if it had never stopped."""
    device = next(model.parameters()).device
    return {
        "key": key,
        "run": run,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        # BigBird switches to full attention for good after a short input; not in state_dict
        "attention_type": getattr(model.base_model, "attention_type", None),
        # epoch, step, epoch_loss, epoch_losses, pruning_f1s and training_seconds
        **asdict(progress),
    }


def _restore(
    state: dict,
    model: PreTrainedModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> _Progress:
    """Load a checkpoint into the model, optimizer, scheduler and RNG, return where the run stood."""
    device = next(model.parameters()).device
    if state["attention_type"] is not None:
        model.base_model.set_attention_type(state["attention_type"])
    model.load_state_dict(state["model"])
    # moves the AdamW state to the device of the parameters
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["rng_cpu"])
    if device.type == "cuda" and state["rng_cuda"] is not None:
        torch.cuda.set_rng_state(state["rng_cuda"], device)
    return _Progress(
        epoch=state["epoch"],
        step=state["step"],
        epoch_loss=state["epoch_loss"],
        epoch_losses=state["epoch_losses"],
        pruning_f1s=state["pruning_f1s"],
        training_seconds=state["training_seconds"],
    )


def _report_and_prune(
    trial: optuna.Trial | None,
    pruning_f1: float,
    epoch: int,
    training_seconds: float,
) -> None:
    """Report the pruning F1 of an epoch to the trial, if there is one.
            The training time so far is stored in the trial first - a pruned trial returns no TrainingResult.

    Raises:
        optuna.TrialPruned: If the pruner stops the trial.
    """
    if trial is None:
        return
    trial.set_user_attr(TRAINING_SECONDS_ATTR, training_seconds)
    trial.report(pruning_f1, epoch)
    if trial.should_prune():
        raise optuna.TrialPruned()


def _train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    batch_df: pd.DataFrame,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> float:
    """Train on one batch and return its loss."""
    device = next(model.parameters()).device
    # long - cross_entropy rejects the int32 that astype(int) gives on Windows
    labels = torch.tensor(batch_df["same"].to_numpy(), dtype=torch.long, device=device)
    loss = model(
        **_encode(tokenizer, batch_df["pair"].tolist(), device),
        labels=labels,
    ).loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item()


def _compute_pruning_f1(validation_scores_df: pd.DataFrame) -> float:
    """Return the weighted F1 of the argmax labels, the value the pruner compares."""
    pruning_f1 = f1_score(
        validation_scores_df["same"].astype(int),
        validation_scores_df["argmax"],
        average="weighted",
    )
    return float(pruning_f1)


def train_classifier(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_df: pd.DataFrame,
    settings: TrainingSettings,
    checkpoint_dir: Path,
    checkpoint_key: str,
    *,
    validation_df: pd.DataFrame | None = None,
    trial: optuna.Trial | None = None,
    checkpoint_minutes: float = CHECKPOINT_MINUTES,
) -> TrainingResult:
    """Fine-tune the classifier with AdamW and a linear warmup schedule, resuming a checkpoint.

    The state is saved every checkpoint_minutes and after every epoch into ONE file in
    checkpoint_dir; a checkpoint with another key is overwritten, so only the last run is kept.
    One process per checkpoint_dir.

    Args:
        model: Freshly loaded classifier with 2 labels, already on its device, before its first
            forward pass (a short input changes the attention modules of BigBird). Seed it with
            transformers.set_seed before from_pretrained - this module seeds only the epoch order.
        tokenizer: Tokenizer of the model.
        train_df: Training pairs with id, same and pair.
        settings: Hyperparameters of the run.
        checkpoint_dir: Folder of the checkpoints of one notebook, outside the repo.
        checkpoint_key: Name of the run - checkpoint_key_for(trial), or e.g. "final".
        validation_df: Pairs scored after every epoch for pruning; None for the final training.
        trial: Optuna trial the pruning F1 is reported to; needs validation_df. The training time
            so far is stored in its user attribute TRAINING_SECONDS_ATTR before every report.
        checkpoint_minutes: Time between two checkpoints inside an epoch.

    Returns:
        The losses, the pruning F1 values, the last validation scores and the training time.

    Raises:
        ValueError: If train_df or validation_df is empty, a setting is out of range, trial is
            given without validation_df, the model has not 2 labels, or the checkpoint of this
            key can not be read or describes another run.
        optuna.TrialPruned: If the pruner stops the trial after an epoch.
    """
    _check_inputs(model, train_df, settings, validation_df, trial)
    # the last batch is smaller, as with the v1 DataLoader (drop_last=False)
    steps_per_epoch = -(-len(train_df) // settings.batch_size)
    optimizer, scheduler = _build_optimizer_and_scheduler(
        model,
        settings,
        steps_per_epoch,
    )
    run = _describe_run(model, tokenizer, train_df, validation_df, settings, trial)
    progress = _Progress()
    state = _load_checkpoint(checkpoint_dir, checkpoint_key, run)
    if state is not None:
        progress = _restore(
            state=state,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        del state  # one copy of the weights, not two
        logger.info(
            f"{checkpoint_key} resumed at epoch {progress.epoch + 1}, step {progress.step}.",
        )
        # the reports of the crashed trial are hidden from the pruner - they are replayed with
        # their pruning decisions, as in the uninterrupted run
        for done_epoch, pruning_f1 in enumerate(progress.pruning_f1s):
            _report_and_prune(trial, pruning_f1, done_epoch, progress.training_seconds)

    def save() -> None:
        nonlocal last_save
        # only the time up to a save counts - a restart repeats the work done after it
        progress.training_seconds += time.monotonic() - last_save
        state = _collect_checkpoint_state(
            key=checkpoint_key,
            run=run,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            progress=progress,
        )
        _save_checkpoint(checkpoint_dir, state)
        last_save = time.monotonic()

    validation_scores_df = None
    model.train()
    last_save = time.monotonic()
    while progress.epoch < settings.epochs:
        # the order of an epoch depends only on the seed, so a resumed epoch skips its done batches
        order = _epoch_order(len(train_df), settings.seed, progress.epoch)
        while progress.step < steps_per_epoch:
            start = progress.step * settings.batch_size
            batch_df = train_df.iloc[order[start : start + settings.batch_size]]
            progress.epoch_loss += _train_step(
                model=model,
                tokenizer=tokenizer,
                batch_df=batch_df,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            progress.step += 1
            if time.monotonic() - last_save >= checkpoint_minutes * 60:
                save()
        progress.epoch_losses.append(progress.epoch_loss / steps_per_epoch)
        if validation_df is not None:
            validation_scores_df = score_pairs(model, tokenizer, validation_df)
            progress.pruning_f1s.append(_compute_pruning_f1(validation_scores_df))
        logger.info(
            f"{checkpoint_key} epoch {progress.epoch + 1} / {settings.epochs} - "
            f"loss {progress.epoch_losses[-1]:.4f}",
        )
        finished_epoch = progress.epoch
        progress.epoch += 1
        progress.step = 0
        progress.epoch_loss = 0.0
        # saved before the report - a trial killed between the two replays the report on resume
        save()
        if trial is not None:
            _report_and_prune(
                trial,
                progress.pruning_f1s[-1],
                finished_epoch,
                progress.training_seconds,
            )
    if validation_df is not None and validation_scores_df is None:
        # resumed after the last epoch - the scores of the restored model
        validation_scores_df = score_pairs(model, tokenizer, validation_df)
    return TrainingResult(
        epoch_losses=progress.epoch_losses,
        pruning_f1s=progress.pruning_f1s,
        validation_scores_df=validation_scores_df,
        training_seconds=progress.training_seconds,
    )
