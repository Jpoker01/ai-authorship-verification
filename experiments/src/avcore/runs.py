"""Checks and tables that every notebook run needs around its study and its result folder."""

import json
import shutil
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