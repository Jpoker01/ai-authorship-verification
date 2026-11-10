"""Training and scoring of the transformer classification heads, resumable after a crash.

A Kubernetes Job pod that is restarted runs the notebook from the top again. The trial that
was running is repeated with the same hyperparameters (tuning.py), and train_classifier
continues it from its last checkpoint instead of from the first step.
"""

import hashlib
import logging
import os
import pickle
import time
import zipfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
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

@dataclass(frozen=True)
class TrainingSettings:
    """Hyperparameters of one training run (one Optuna trial or the final training)."""

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
    """

    epoch_losses: list[float]
    pruning_f1s: list[float]
    validation_scores_df: pd.DataFrame | None

def checkpoint_key_for(trial: optuna.Trial) -> str:
    """Name the checkpoint of a trial by its study and position, which a repeated trial keeps.

    Raises:
        ValueError: If the trial has no position, i.e. it was not started by tuning.run_study.
    """
    if "position" not in trial.user_attrs:
        raise ValueError("The trial has no position - start the study with tuning.run_study.")
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
    model.eval()
    scores = []
    argmaxes = []
    with torch.no_grad():
        for pair in pairs_df["pair"]:
            logits = model(**_encode(tokenizer, [pair], device)).logits
            scores.append(F.softmax(logits, dim=-1)[0, 1].item())
            argmaxes.append(torch.argmax(logits, dim=-1).item())
    model.train(was_training)
    return pd.DataFrame(
        {
            "id": pairs_df["id"].to_numpy(),
            "same": pairs_df["same"].to_numpy(),
            "score": scores,
            "argmax": argmaxes,
        }
    )

def _epoch_order(n_pairs: int, seed: int, epoch: int) -> list[int]:
    """Return the shuffled row order of an epoch, the same for the same seed and epoch."""
    generator = torch.Generator().manual_seed(seed + epoch)
    return torch.randperm(n_pairs, generator=generator).tolist()

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _describe_run(
    model: PreTrainedModel,
    train_df: pd.DataFrame,
    settings: TrainingSettings,
    trial: optuna.Trial | None,
) -> dict:
    """Return everything that decides the trained weights - a checkpoint resumes only an equal run."""
    return {
        **asdict(settings),
        "model": model.name_or_path,
        # holds the dropout too, when it is set through from_pretrained
        "model_config_sha256": _sha256(model.config.to_json_string()),
        "train_pairs": len(train_df),
        "train_ids_sha256": _sha256("\n".join(train_df["id"].astype(str))),
        "trial_params": {} if trial is None else dict(trial.params),
    }

def _save_checkpoint(checkpoint_dir: Path, state: dict) -> None:
    """Write the state to a temporary file, then replace the old checkpoint in one step."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
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
    # torch.save always writes a zip; anything else would fail in torch.load with an arbitrary error
    if not zipfile.is_zipfile(checkpoint_path):
        raise ValueError(f"{checkpoint_path} can not be read - delete it: not a torch.save file")
    try:
        # on the CPU - torch.set_rng_state takes only CPU tensors
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (RuntimeError, EOFError, IndexError, pickle.UnpicklingError, zipfile.BadZipFile) as error:
        raise ValueError(f"{checkpoint_path} can not be read - delete it: {error}") from error
    if state["key"] != key:
        logger.info(f"Checkpoint of {state['key']} ignored, it is replaced by {key}.")
        return None
    if state["run"] != run:
        raise ValueError(
            f"{checkpoint_path} ({key}) was written for other settings or pairs - "
            f"delete it if its study file was deleted: {state['run']}"
        )
    return state

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
        model: Freshly loaded classifier with 2 labels, already on its device.
        tokenizer: Tokenizer of the model.
        train_df: Training pairs with id, same and pair.
        settings: Hyperparameters of the run.
        checkpoint_dir: Folder of the checkpoints of one notebook, outside the repo.
        checkpoint_key: Name of the run - checkpoint_key_for(trial), or e.g. "final".
        validation_df: Pairs scored after every epoch for pruning; None for the final training.
        trial: Optuna trial the pruning F1 is reported to; needs validation_df.
        checkpoint_minutes: Time between two checkpoints inside an epoch.

    Returns:
        The losses, the pruning F1 values and the last validation scores.

    Raises:
        ValueError: If train_df is empty, a setting is out of range, trial is given without
            validation_df, the model has not 2 labels, or the checkpoint of this key can not be
            read or describes another run.
        optuna.TrialPruned: If the pruner stops the trial after an epoch.
    """
    if train_df.empty:
        raise ValueError("train_df is empty.")
    if settings.epochs < 1 or settings.batch_size < 1:
        raise ValueError(f"epochs and batch_size have to be at least 1: {settings}")
    if trial is not None and validation_df is None:
        raise ValueError("Pruning needs validation_df.")
    if model.config.num_labels != 2:
        raise ValueError(f"The model has {model.config.num_labels} labels, not 2.")