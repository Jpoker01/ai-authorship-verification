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