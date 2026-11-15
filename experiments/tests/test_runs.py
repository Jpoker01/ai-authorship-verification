"""Tests of avcore.runs on constructed studies and folders; nothing is trained."""

import hashlib
import json
import sqlite3
from pathlib import Path

import optuna
import pandas as pd
import pytest
import sqlalchemy
from optuna.trial import TrialState, create_trial

from avcore import runs, tracking, tuning

optuna.logging.set_verbosity(optuna.logging.ERROR)

SEED = 42
SECONDS_ATTR = "training_seconds"


def build_study(trials: list[optuna.trial.FrozenTrial]) -> optuna.Study:
    """Return an in-memory study that maximizes and holds the given trials."""
    study = optuna.create_study(direction="maximize")
    for trial in trials:
        study.add_trial(trial)
    return study


def sha256_of(path: Path) -> str:
    """Return the sha256 hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_run_dir(run_dir: Path, params: dict[str, str]) -> None:
    """Write a result folder that holds only its params file."""
    run_dir.mkdir(parents=True)
    (run_dir / tracking.PARAMS_FILE).write_text(json.dumps(params), encoding="utf-8")


def test_select_best_trial_takes_the_highest_value() -> None:
    """The complete trial with the highest value is the best trial."""
    trials = [
        create_trial(state=TrialState.COMPLETE, value=54.55),
        create_trial(state=TrialState.COMPLETE, value=57.14),
        create_trial(state=TrialState.COMPLETE, value=50.0),
    ]
    study = build_study(trials)
    assert runs.select_best_trial(study).number == 1


def test_select_best_trial_takes_the_earliest_trial_of_equal_values() -> None:
    """Of trials with the same value the one with the lowest number wins."""
    trials = [
        create_trial(state=TrialState.COMPLETE, value=54.55),
        create_trial(state=TrialState.COMPLETE, value=57.14),
        create_trial(state=TrialState.COMPLETE, value=57.14),
        create_trial(state=TrialState.COMPLETE, value=57.14),
    ]
    study = build_study(trials)
    assert runs.select_best_trial(study).number == 1


def test_select_best_trial_ignores_pruned_and_failed_trials() -> None:
    """A pruned or failed trial is never the best trial, whatever it reported."""
    trials = [
        create_trial(
            state=TrialState.PRUNED,
            value=99.0,
            intermediate_values={0: 99.0},
        ),
        create_trial(state=TrialState.FAIL, intermediate_values={0: 98.0}),
        create_trial(state=TrialState.COMPLETE, value=40.0),
    ]
    study = build_study(trials)
    assert runs.select_best_trial(study).number == 2


def test_select_best_trial_without_a_complete_trial_raises() -> None:
    """A study with pruned trials only has no best trial."""
    trials = [create_trial(state=TrialState.PRUNED, intermediate_values={0: 0.5})]
    study = build_study(trials)
    with pytest.raises(ValueError, match="holds no complete trial"):
        runs.select_best_trial(study)


def test_sum_training_seconds_adds_complete_pruned_and_failed_trials() -> None:
    """The search time is the sum over the complete, pruned and failed trials."""
    trials = [
        create_trial(
            state=TrialState.COMPLETE,
            value=1.0,
            user_attrs={SECONDS_ATTR: 10.0},
        ),
        create_trial(
            state=TrialState.PRUNED,
            intermediate_values={0: 0.5},
            user_attrs={SECONDS_ATTR: 5.0},
        ),
        create_trial(state=TrialState.FAIL, user_attrs={SECONDS_ATTR: 3.0}),
    ]
    study = build_study(trials)
    assert runs.sum_training_seconds(study, SECONDS_ATTR) == 18.0


def test_sum_training_seconds_leaves_out_a_crashed_attempt() -> None:
    """The time of a crashed attempt is not added - the repeated trial holds it."""
    trials = [
        create_trial(
            state=TrialState.FAIL,
            user_attrs={SECONDS_ATTR: 7.0, "crashed": True},
        ),
        create_trial(
            state=TrialState.COMPLETE,
            value=1.0,
            user_attrs={SECONDS_ATTR: 10.0},
        ),
    ]
    study = build_study(trials)
    assert runs.sum_training_seconds(study, SECONDS_ATTR) == 10.0


def test_sum_training_seconds_leaves_out_a_running_trial() -> None:
    """A trial that is still running has not ended, so its time is not counted yet."""
    trials = [
        create_trial(state=TrialState.RUNNING, user_attrs={SECONDS_ATTR: 99.0}),
        create_trial(
            state=TrialState.COMPLETE,
            value=1.0,
            user_attrs={SECONDS_ATTR: 10.0},
        ),
    ]
    study = build_study(trials)
    assert runs.sum_training_seconds(study, SECONDS_ATTR) == 10.0


def test_sum_training_seconds_counts_a_trial_without_a_time_as_zero() -> None:
    """A trial that failed before its first epoch has no stored time."""
    trials = [
        create_trial(state=TrialState.FAIL),
        create_trial(
            state=TrialState.COMPLETE,
            value=1.0,
            user_attrs={SECONDS_ATTR: 4.0},
        ),
    ]
    study = build_study(trials)
    assert runs.sum_training_seconds(study, SECONDS_ATTR) == 4.0


def test_sum_training_seconds_without_any_stored_time_raises() -> None:
    """A study whose trials never stored the attribute does not report 0 seconds."""
    trials = [create_trial(state=TrialState.COMPLETE, value=1.0)]
    study = build_study(trials)
    with pytest.raises(ValueError, match="holds no trial with training_seconds"):
        runs.sum_training_seconds(study, SECONDS_ATTR)


def test_build_predictions_score_equal_to_the_threshold_is_same_author() -> None:
    """A pair is predicted as the same author when its score is at least the threshold."""
    pairs_df = pd.DataFrame({"id": ["a", "b", "c"]})
    scores_df = pd.DataFrame(
        {"id": ["a", "b", "c"], "same": [0, 1, 1], "score": [0.3, 0.5, 0.7]},
    )
    threshold = 0.5
    predictions_df = runs.build_predictions(pairs_df, scores_df, threshold)
    assert predictions_df["pred"].tolist() == [0, 1, 1]


def test_build_predictions_has_the_prediction_columns_and_integer_labels() -> None:
    """The table has exactly the columns of a prediction table, both labels as integers."""
    pairs_df = pd.DataFrame({"id": ["a", "b"]})
    scores_df = pd.DataFrame(
        {
            "id": ["a", "b"],
            "same": [True, False],
            "score": [0.9, 0.2],
            "argmax": [1, 0],
        },
    )
    threshold = 0.5
    predictions_df = runs.build_predictions(pairs_df, scores_df, threshold)
    assert list(predictions_df.columns) == tracking.PREDICTION_COLUMNS
    assert pd.api.types.is_integer_dtype(predictions_df["same"])
    assert pd.api.types.is_integer_dtype(predictions_df["pred"])
    assert predictions_df["same"].tolist() == [1, 0]
    assert predictions_df["score"].tolist() == [0.9, 0.2]


@pytest.mark.parametrize(
    "score_ids",
    [["x", "y", "z"], ["a", "b"], ["b", "a", "c"]],
    ids=["other_pairs", "a_missing_pair", "another_order"],
)
def test_build_predictions_rejects_scores_that_are_not_the_pairs(
    score_ids: list[str],
) -> None:
    """Scores of other pairs, of fewer pairs or in another order are refused."""
    pairs_df = pd.DataFrame({"id": ["a", "b", "c"]})
    scores_df = pd.DataFrame(
        {
            "id": score_ids,
            "same": [0] * len(score_ids),
            "score": [0.5] * len(score_ids),
        },
    )
    threshold = 0.5
    with pytest.raises(ValueError, match="do not hold every pair once"):
        runs.build_predictions(pairs_df, scores_df, threshold)


def test_build_predictions_rejects_a_repeated_id() -> None:
    """A pair that appears twice is refused, also when the scores repeat it the same way."""
    pairs_df = pd.DataFrame({"id": ["a", "b", "a"]})
    scores_df = pd.DataFrame(
        {"id": ["a", "b", "a"], "same": [0, 1, 0], "score": [0.3, 0.7, 0.3]},
    )
    threshold = 0.5
    with pytest.raises(ValueError, match="repeated id"):
        runs.build_predictions(pairs_df, scores_df, threshold)


def test_build_predictions_rejects_a_nan_score() -> None:
    """A NaN score is refused instead of being predicted as a different author."""
    pairs_df = pd.DataFrame({"id": ["a", "b"]})
    scores_df = pd.DataFrame(
        {"id": ["a", "b"], "same": [0, 1], "score": [float("nan"), 0.7]},
    )
    threshold = 0.5
    with pytest.raises(ValueError, match="NaN"):
        runs.build_predictions(pairs_df, scores_df, threshold)


def test_check_study_settings_passes_without_a_study_file(tmp_path: Path) -> None:
    """A study that does not exist yet has no settings to compare."""
    study_path = tmp_path / "optuna_study_new.db"
    settings = {"n_trials": 12}
    runs.check_study_settings(study_path, settings)
    assert not study_path.exists()


def test_check_study_settings_passes_for_the_stored_settings(tmp_path: Path) -> None:
    """The settings the study was started with are accepted, also as numbers and tuples."""
    study_path = tmp_path / "optuna_study_equal.db"
    settings = {"n_trials": 12, "search.lr": (1e-6, 5e-5), "smoke": True}
    study = tuning.open_study(study_path, seed=SEED)
    runs.store_study_settings(study, settings)
    runs.check_study_settings(study_path, settings)


def test_check_study_settings_names_a_changed_setting(tmp_path: Path) -> None:
    """A setting that differs from the stored one stops the run and is named."""
    study_path = tmp_path / "optuna_study_changed.db"
    stored_settings = {"n_trials": 12, "model_id": "model-a"}
    changed_settings = {"n_trials": 13, "model_id": "model-a"}
    study = tuning.open_study(study_path, seed=SEED)
    runs.store_study_settings(study, stored_settings)
    with pytest.raises(RuntimeError, match="n_trials: 12 -> 13"):
        runs.check_study_settings(study_path, changed_settings)


def test_check_study_settings_names_a_setting_that_is_no_longer_there(
    tmp_path: Path,
) -> None:
    """A stored setting without a current value stops the run too."""
    study_path = tmp_path / "optuna_study_removed.db"
    stored_settings = {"n_trials": 12, "model_id": "model-a"}
    shorter_settings = {"n_trials": 12}
    study = tuning.open_study(study_path, seed=SEED)
    runs.store_study_settings(study, stored_settings)
    with pytest.raises(RuntimeError, match="model_id: model-a -> <missing>"):
        runs.check_study_settings(study_path, shorter_settings)


def test_check_study_settings_names_a_new_setting(tmp_path: Path) -> None:
    """A current setting the study was not started with stops the run too."""
    study_path = tmp_path / "optuna_study_added.db"
    stored_settings = {"n_trials": 12}
    longer_settings = {"n_trials": 12, "model_id": "model-a"}
    study = tuning.open_study(study_path, seed=SEED)
    runs.store_study_settings(study, stored_settings)
    with pytest.raises(RuntimeError, match="model_id: <missing> -> model-a"):
        runs.check_study_settings(study_path, longer_settings)


@pytest.mark.parametrize(
    "settings",
    [{"search.ids": {1, 2}}, {7: "seven"}, {"search.lr": (pd.NA,)}],
    ids=["a_set", "an_int_key", "a_non_plain_item"],
)
def test_check_study_settings_rejects_a_setting_whose_text_is_not_stable(
    tmp_path: Path,
    settings: dict,
) -> None:
    """A value whose text can change between runs, or a key that is not a str, is refused."""
    study_path = tmp_path / "optuna_study_unstable.db"
    with pytest.raises(TypeError):
        runs.check_study_settings(study_path, settings)


def test_check_study_settings_leaves_the_study_file_unchanged(tmp_path: Path) -> None:
    """The refusal does not touch the file - a trial that was running stays running."""
    study_path = tmp_path / "optuna_study_untouched.db"
    stored_settings = {"n_trials": 12}
    changed_settings = {"n_trials": 13}
    study = tuning.open_study(study_path, seed=SEED)
    runs.store_study_settings(study, stored_settings)
    study.ask()
    sha256_before = sha256_of(study_path)
    with pytest.raises(RuntimeError):
        runs.check_study_settings(study_path, changed_settings)
    stored_study = optuna.load_study(
        study_name=study_path.stem,
        storage=f"sqlite:///{study_path}",
    )
    assert sha256_of(study_path) == sha256_before
    assert [trial.state for trial in stored_study.trials] == [TrialState.RUNNING]


def test_check_study_settings_leaves_an_empty_file_empty(tmp_path: Path) -> None:
    """A file that a killed start left empty is not filled with tables by the check."""
    study_path = tmp_path / "optuna_study_zero.db"
    study_path.touch()
    settings = {"n_trials": 12}
    runs.check_study_settings(study_path, settings)
    assert study_path.stat().st_size == 0


def test_check_study_settings_never_writes_into_the_file(tmp_path: Path) -> None:
    """A file Optuna would migrate is refused on the read-only connection, not changed."""
    study_path = tmp_path / "optuna_study_unstamped.db"
    settings = {"n_trials": 12}
    study = tuning.open_study(study_path, seed=SEED)
    runs.store_study_settings(study, settings)
    connection = sqlite3.connect(study_path)
    connection.execute("DELETE FROM alembic_version")
    connection.commit()
    connection.close()
    sha256_before = sha256_of(study_path)
    with pytest.raises(sqlalchemy.exc.OperationalError, match="readonly"):
        runs.check_study_settings(study_path, settings)
    assert sha256_of(study_path) == sha256_before


def test_check_study_settings_rejects_trials_without_settings(tmp_path: Path) -> None:
    """A study that holds trials but no settings was not started by the notebook."""
    study_path = tmp_path / "optuna_study_foreign.db"
    settings = {"n_trials": 12}
    study = tuning.open_study(study_path, seed=SEED)
    study.ask()
    with pytest.raises(RuntimeError, match="holds trials but no settings"):
        runs.check_study_settings(study_path, settings)


def test_check_study_settings_passes_for_a_study_without_trials_and_settings(
    tmp_path: Path,
) -> None:
    """A run killed between creating the study and storing the settings can start again."""
    study_path = tmp_path / "optuna_study_empty.db"
    settings = {"n_trials": 12}
    tuning.open_study(study_path, seed=SEED)
    runs.check_study_settings(study_path, settings)


def test_check_study_settings_passes_for_a_file_without_the_study(
    tmp_path: Path,
) -> None:
    """A run killed before the study was created in its file can start again."""
    study_path = tmp_path / "optuna_study_no_study.db"
    settings = {"n_trials": 12}
    optuna.storages.RDBStorage(f"sqlite:///{study_path}")
    runs.check_study_settings(study_path, settings)


def test_store_study_settings_stores_every_value_as_text(tmp_path: Path) -> None:
    """The settings are stored as text, the form they are compared in."""
    study_path = tmp_path / "optuna_study_text.db"
    settings = {"n_trials": 12, "search.lr": (1e-6, 5e-5), "smoke": True}
    study = tuning.open_study(study_path, seed=SEED)
    runs.store_study_settings(study, settings)
    assert study.user_attrs[runs.STUDY_SETTINGS_ATTR] == {
        "n_trials": "12",
        "search.lr": "(1e-06, 5e-05)",
        "smoke": "True",
    }


def test_store_study_settings_keeps_the_first_settings(tmp_path: Path) -> None:
    """Settings that are stored once are never replaced."""
    study_path = tmp_path / "optuna_study_first.db"
    first_settings = {"n_trials": 12}
    second_settings = {"n_trials": 13}
    study = tuning.open_study(study_path, seed=SEED)
    runs.store_study_settings(study, first_settings)
    runs.store_study_settings(study, second_settings)
    assert study.user_attrs[runs.STUDY_SETTINGS_ATTR] == {"n_trials": "12"}


def test_find_finished_run_without_a_folder_is_false(tmp_path: Path) -> None:
    """Without a result folder the run is not finished."""
    run_dir = tmp_path / "run_v2"
    params = {"seed": 42}
    assert runs.find_finished_run(run_dir, params) is False


def test_find_finished_run_with_the_saved_params_is_true(tmp_path: Path) -> None:
    """A result folder with the same params marks a finished run."""
    run_dir = tmp_path / "run_v2"
    saved_params = {"seed": "42", "final_epochs": "2"}
    params = {"seed": 42, "final_epochs": 2}
    write_run_dir(run_dir, saved_params)
    assert runs.find_finished_run(run_dir, params) is True


def test_find_finished_run_compares_only_the_given_params(tmp_path: Path) -> None:
    """Before the tuning only a part of the params is known - the rest is not compared."""
    run_dir = tmp_path / "run_v2"
    saved_params = {"seed": "42", "final_epochs": "2"}
    known_params = {"seed": 42}
    write_run_dir(run_dir, saved_params)
    assert runs.find_finished_run(run_dir, known_params) is True


def test_find_finished_run_with_other_params_raises(tmp_path: Path) -> None:
    """A result folder of other params stops the run and names the difference."""
    run_dir = tmp_path / "run_v2"
    saved_params = {"seed": "42", "final_epochs": "2"}
    params = {"seed": 42, "final_epochs": 3}
    write_run_dir(run_dir, saved_params)
    with pytest.raises(RuntimeError, match="final_epochs: 2 -> 3"):
        runs.find_finished_run(run_dir, params)

def test_find_finished_run_with_a_param_the_folder_lacks_raises(tmp_path: Path) -> None:
    """A result folder without one of the given params is not this run."""
    run_dir = tmp_path / "run_v2"
    saved_params = {"seed": "42"}
    params = {"seed": 42, "final_epochs": 2}
    write_run_dir(run_dir, saved_params)
    with pytest.raises(RuntimeError, match="final_epochs: <missing> -> 2"):
        runs.find_finished_run(run_dir, params)

def test_find_finished_run_without_a_params_file_raises(tmp_path: Path) -> None:
    """A folder of the run name without a params file is not a finished run."""
    run_dir = tmp_path / "run_v2"
    run_dir.mkdir()
    params = {"seed": 42}
    with pytest.raises(RuntimeError, match="is not a finished run folder"):
        runs.find_finished_run(run_dir, params)


def test_remove_unfinished_run_dir_removes_only_the_partial_folder(
    tmp_path: Path,
) -> None:
    """The half-written folder goes; the finished folder and every other folder stay."""
    run_dir = tmp_path / "run_v2"
    partial_dir = tmp_path / "run_v2.partial"
    other_dir = tmp_path / "other_v2"
    for folder in (run_dir, partial_dir, other_dir):
        folder.mkdir()
        (folder / "metrics.json").write_text("{}", encoding="utf-8")
    runs.remove_unfinished_run_dir(run_dir)
    assert not partial_dir.exists()
    assert (run_dir / "metrics.json").is_file()
    assert (other_dir / "metrics.json").is_file()


def test_remove_unfinished_run_dir_without_a_partial_folder_does_nothing(
    tmp_path: Path,
) -> None:
    """A run that was never killed inside a save has nothing to remove."""
    run_dir = tmp_path / "run_v2"
    runs.remove_unfinished_run_dir(run_dir)
    assert list(tmp_path.iterdir()) == []
