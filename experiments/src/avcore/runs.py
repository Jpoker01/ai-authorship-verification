"""Checks and tables that every notebook run needs around its study and its result folder."""

import json
import shutil
import sqlite3

from collections.abc import Mapping
from pathlib import Path

import optuna
import pandas as pd
from optuna.trial import FrozenTrial, TrialState

from avcore import tracking, tuning

# name of the study attribute that keeps the settings the study was started with
STUDY_SETTINGS_ATTR = "settings"
# shown for a setting that exists on one side only
MISSING = "<missing>"
# exact types only - numpy scalars print differently inside a tuple
_PLAIN_TYPES = (str, int, float, bool, type(None))


def _is_plain(value: object) -> bool:
    """Say whether a value has a stable text - a str, int, float, bool or None.

    The exact type is compared, so a numpy scalar is not plain.
    """
    return type(value) in _PLAIN_TYPES

def _settings_as_text(settings: Mapping[str, object]) -> dict[str, str]:
    """Return the settings as text, the form they are stored and compared in.

    Args:
        settings: Everything that is fixed before the first trial.

    Returns:
        Every setting with its value as str.

    Raises:
        TypeError: If a key is not a str, or a value is neither plain nor a
            tuple / list of plain values.
    """
    for key, value in settings.items():
        if type(key) is not str:
            raise TypeError(f"Setting {key!r} needs a str name.")
        if type(value) in (tuple, list):
            is_stable = all(_is_plain(item) for item in value)
        else:
            is_stable = _is_plain(value)
        if not is_stable:
            raise TypeError(
                f"Setting {key} has a value whose text is not stable: {value!r}.")
    return {key: str(value) for key, value in settings.items()}


def _read_stored_settings(study_path: Path) -> Mapping[str, str] | None:
    """Return the settings stored in the study file; the file is only read.

    Args:
        study_path: SQLite file of the study; its stem is the study name.

    Returns:
        The stored settings, or None if there is nothing to compare yet - no file, an
        empty file, no study in the file, or a study without trials and settings.

    Raises:
        RuntimeError: If the study holds trials but no settings.
    """
    if not study_path.is_file() or study_path.stat().st_size == 0:
        return None
    # read-only, so Optuna can not create or migrate tables in the file; as_uri gives
    # the SQLite URI form also for a Windows path (forward slashes, /C:/...)
    read_only_uri = f"{study_path.as_uri()}?mode=ro"
    storage = optuna.storages.RDBStorage(
        "sqlite://",
        engine_kwargs={"creator": lambda: sqlite3.connect(read_only_uri, uri=True)},
    )
    try:
        stored_study = optuna.load_study(study_name=study_path.stem, storage=storage)
    except KeyError:
        # the file exists, the study in it was not created yet
        return None
    stored_settings = stored_study.user_attrs.get(STUDY_SETTINGS_ATTR)
    if stored_settings is None and stored_study.get_trials(deepcopy=False):
        raise RuntimeError(
            f"Study {study_path.stem} holds trials but no settings - move it away."
        )
    return stored_settings

def _list_changes(stored: Mapping[str, str], current: Mapping[str, str]) -> list[str]:
    """Return one "key: stored -> current" line per setting that differs.

    Args:
        stored: Settings the study was started with.
        current: Settings of this run.

    Returns:
        The differences in key order; a key on one side only shows MISSING on the other.
    """
    return [
        f"{key}: {stored.get(key, MISSING)} -> {current.get(key, MISSING)}"
        for key in sorted(set(stored) | set(current))
        if stored.get(key, MISSING) != current.get(key, MISSING)
    ]

def check_study_settings(study_path: Path, settings: Mapping[str, object]) -> None:
    """Stop when the study file was started with other settings; the file is only read.

    Args:
        study_path: SQLite file of the study; its stem is the study name.
        settings: Everything that is fixed before the first trial.

    Raises:
        RuntimeError: If the stored settings differ, or the study holds trials but no settings.
        TypeError: If a setting is not plain (see _settings_as_text).
    """
    current_settings = _settings_as_text(settings)
    stored_settings = _read_stored_settings(study_path)
    if stored_settings is None:
        return
    changes = _list_changes(stored_settings, current_settings)
    if changes:
        raise RuntimeError(
            f"Study {study_path.stem} was started with other settings - "
            f"{'; '.join(changes)}. Move the study away or use another run name."
        )

def store_study_settings(study: optuna.Study, settings: Mapping[str, object]) -> None:
    """Keep the settings in a study that holds none yet.

    Args:
        study: Study returned by tuning.open_study.
        settings: Everything that is fixed before the first trial.

    Raises:
        TypeError: If a setting is not plain (see _settings_as_text).
    """
    current_settings = _settings_as_text(settings)
    if STUDY_SETTINGS_ATTR not in study.user_attrs:
        study.set_user_attr(STUDY_SETTINGS_ATTR, current_settings)

def select_best_trial(study: optuna.Study) -> FrozenTrial:
    """Return the complete trial with the highest value, the earliest one if two are equal.

    Args:
        study: Finished study.

    Returns:
        The best trial.

    Raises:
        ValueError: If the study holds no complete trial.
    """
    complete_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    if not complete_trials:
        raise ValueError(f"Study {study.study_name} holds no complete trial.")
    return max(complete_trials, key=lambda trial: (trial.value, -trial.number))

def sum_training_seconds(study: optuna.Study, seconds_attr: str) -> float:
    """Return the training time of the search in seconds.

    A crashed attempt is repeated, so its time is inside the repeated trial and not added.

    Args:
        study: Finished study whose trials hold their training time.
        seconds_attr: Name of the user attribute that holds the training time of a trial.

    Returns:
        The sum over the complete, pruned and failed trials that did not crash.

    Raises:
        ValueError: If no such trial holds seconds_attr.
    """
    counted_trials = [
        trial
        for trial in study.get_trials(deepcopy=False, states=tuning.ENDED_STATES)
        if not trial.user_attrs.get("crashed")
    ]
    if not any(seconds_attr in trial.user_attrs for trial in counted_trials):
        raise ValueError(f"Study {study.study_name} holds no trial with {seconds_attr}.")
    return sum(
        (trial.user_attrs.get(seconds_attr, 0.0) for trial in counted_trials), start=0.0
    )

def find_finished_run(run_dir: Path, params: Mapping[str, object]) -> bool:
    """Say whether the result folder of the run exists - a finished run is not repeated.

    Args:
        run_dir: Result folder of the run.
        params: The params of the run that are known so far; the folder has to hold their values.

    Returns:
        True if the folder exists and holds these params.

    Raises:
        RuntimeError: If run_dir exists without a params file, or holds another value of one
            of these params.
    """
    if not run_dir.exists():
        return False
    params_path = run_dir / tracking.PARAMS_FILE
    if not params_path.is_file():
        raise RuntimeError(f"{run_dir} is not a finished run folder - move it away.")
    saved_params = json.loads(params_path.read_text(encoding="utf-8"))
    changes = []
    for key, value in params.items():
        saved_value = saved_params.get(key, MISSING)
        if saved_value != str(value):
            changes.append(f"{key}: {saved_value} -> {value}")
    if changes:
        raise RuntimeError(
            f"{run_dir} holds the results of other settings - {'; '.join(changes)}. "
            "Move it away."
        )
    return True