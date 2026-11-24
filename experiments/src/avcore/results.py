"""Result tables of the v2 runs, read from the run folders of tracking.write_run_files.

The four test metrics are recomputed from predictions-test.csv, because metrics.json
holds them rounded to two decimals.
"""

import json
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from avcore import tracking
from avcore.evaluation import METRIC_COLUMNS

IDENTITY_COLUMNS = ["family", "model", "representation", "combination", "classifier"]
TEST_PREDICTIONS_FILE = "predictions-test.csv"
TRIAL_COUNT_METRICS = [
    "n_trials_complete",
    "n_trials_pruned",
    "n_trials_failed",
    "n_trials_crashed",
]
LLM_COUNT_METRICS = ["n_unanswered", "n_cut_by_context", "n_cut_by_budget"]
REPLICATE_COLUMNS = ["run_name", "quantity", "run_1", "run_2", "mean", "abs_delta"]


class TableDefinition(NamedTuple):
    """Which runs enter one result table.

    Attributes:
        families: Experiment families whose runs are ranked in the table.
        max_rows: The best runs by F1 that are kept; None keeps every run.
    """

    families: tuple[str, ...]
    max_rows: int | None


RESULT_TABLES = {
    "table1_traditional": TableDefinition(families=("traditional",), max_rows=5),
    "table2_graph": TableDefinition(families=("graph",), max_rows=5),
    "table3_transformer": TableDefinition(families=("transformer",), max_rows=None),
    "table4_overall": TableDefinition(
        families=("traditional", "graph", "transformer", "llm"),
        max_rows=5,
    ),
    "table5_llm": TableDefinition(families=("llm",), max_rows=None),
}


def compute_test_metrics(predictions_df: pd.DataFrame) -> dict[str, float]:
    """Compute accuracy, precision, recall and F1 of the same-author class in percent.

    Args:
        predictions_df: Test predictions with the columns same and pred
            (1 = same author).

    Returns:
        The four metrics in percent, not rounded.
    """
    labels = predictions_df["same"].astype(int)
    predictions = predictions_df["pred"].astype(int)
    return {
        "accuracy": 100 * accuracy_score(labels, predictions),
        "precision": 100 * precision_score(labels, predictions, zero_division=0),
        "recall": 100 * recall_score(labels, predictions, zero_division=0),
        "f1": 100 * f1_score(labels, predictions, zero_division=0),
    }


def read_test_predictions(run_dir: Path) -> pd.DataFrame:
    """Read the test predictions of a run folder.

    Args:
        run_dir: Folder written by tracking.write_run_files.

    Returns:
        The columns id (text, so leading zeros stay), same, score and pred.
    """
    return pd.read_csv(run_dir / TEST_PREDICTIONS_FILE, dtype={"id": str})


def read_run(run_dir: Path) -> dict[str, object]:
    """Read one finished run folder into one row of the runs table.

    Args:
        run_dir: Folder written by tracking.write_run_files.

    Returns:
        The run name, the five identity params, the recomputed test metrics, the
        threshold and the trial and LLM counts; a value the run did not log is None.
    """
    params = json.loads((run_dir / tracking.PARAMS_FILE).read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / tracking.METRICS_FILE).read_text(encoding="utf-8"))
    run = {"run_name": run_dir.name}
    for column in IDENTITY_COLUMNS:
        run[column] = params[column]
    run.update(compute_test_metrics(read_test_predictions(run_dir)))
    run["threshold"] = metrics.get("threshold")
    for name in [*TRIAL_COUNT_METRICS, *LLM_COUNT_METRICS]:
        run[name] = metrics.get(name)
    return run


def collect_runs(results_dir: Path) -> pd.DataFrame:
    """Read every finished run folder of results_dir into the runs table.

    Args:
        results_dir: Folder with one sub-folder per run (~/results_v2).

    Returns:
        One row per run, sorted by run name.

    Raises:
        ValueError: If the folder holds no finished run.
    """
    runs = []
    for run_dir in sorted(results_dir.iterdir()):
        # write_run_files writes into <run>.partial and renames it once complete
        if run_dir.name.endswith(".partial"):
            continue
        if not (run_dir / tracking.MANIFEST_FILE).is_file():
            continue
        runs.append(read_run(run_dir))
    if not runs:
        raise ValueError(f"{results_dir} holds no finished run folder.")
    return pd.DataFrame(runs)


def average_runs(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Merge the runs of one configuration into one row with their mean metrics.

    Args:
        runs_df: Output of collect_runs.

    Returns:
        One row per configuration with the mean metrics, the number of runs n_runs
        and the run names joined by "+".
    """
    grouped = runs_df.groupby(IDENTITY_COLUMNS, sort=False)
    averaged_df = grouped[METRIC_COLUMNS].mean()
    averaged_df["n_runs"] = grouped.size()
    averaged_df["run_name"] = grouped["run_name"].agg("+".join)
    return averaged_df.reset_index()


def build_result_table(
    averaged_df: pd.DataFrame,
    table: TableDefinition,
) -> pd.DataFrame:
    """Rank the configurations of the table's families by F1.

    Args:
        averaged_df: Output of average_runs.
        table: The families and the number of rows of the table.

    Returns:
        The best configurations, best first.
    """
    table_df = averaged_df[averaged_df["family"].isin(table.families)]
    table_df = table_df.sort_values("f1", ascending=False, kind="stable")
    if table.max_rows is not None:
        table_df = table_df.head(table.max_rows)
    columns = [*IDENTITY_COLUMNS, *METRIC_COLUMNS, "n_runs", "run_name"]
    return table_df[columns].reset_index(drop=True)


def _name_length(run_names: pd.Series) -> pd.Series:
    """Return the length of every run name; an untagged run has the shorter one."""
    return run_names.str.len()


def build_replicate_table(runs_df: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    """Put the two runs of every configuration that was run twice side by side.

    Args:
        runs_df: Output of collect_runs.
        results_dir: Folder of the run folders.

    Returns:
        Per metric and LLM count - both runs, their mean and their absolute
        difference; per configuration - the share of test pairs with the same
        prediction in both runs, in the column mean.
    """
    rows = []
    for _, group_df in runs_df.groupby(IDENTITY_COLUMNS, sort=False):
        if len(group_df) != 2:
            continue
        # the run without RUN_TAG becomes run_1, its repeat run_2
        group_df = group_df.sort_values("run_name", key=_name_length, kind="stable")
        first_run = group_df.iloc[0]
        second_run = group_df.iloc[1]
        run_name = f"{first_run['run_name']}+{second_run['run_name']}"
        quantities = list(METRIC_COLUMNS)
        # only the LLM runs log the counts of unanswered and cut pairs
        if first_run["family"] == "llm":
            quantities.extend(LLM_COUNT_METRICS)
        for quantity in quantities:
            first_value = first_run[quantity]
            second_value = second_run[quantity]
            rows.append(
                {
                    "run_name": run_name,
                    "quantity": quantity,
                    "run_1": first_value,
                    "run_2": second_value,
                    "mean": (first_value + second_value) / 2,
                    "abs_delta": abs(first_value - second_value),
                },
            )
        first_df = read_test_predictions(results_dir / first_run["run_name"])
        second_df = read_test_predictions(results_dir / second_run["run_name"])
        pairs_df = first_df.merge(second_df, on="id", suffixes=("_1", "_2"))
        same_share = 100 * (pairs_df["pred_1"] == pairs_df["pred_2"]).mean()
        rows.append(
            {"run_name": run_name, "quantity": "same_prediction", "mean": same_share},
        )
    return pd.DataFrame(rows, columns=REPLICATE_COLUMNS)


def build_confusion_table(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Count the predictions of each actual class, also as a share of that class.

    Args:
        predictions_df: Test predictions with the columns same and pred
            (1 = same author).

    Returns:
        One row per actual class (1 first) with the counts pred_1 and pred_0 and
        their percent of the row.
    """
    labels = predictions_df["same"].astype(int)
    predictions = predictions_df["pred"].astype(int)
    rows = []
    for actual in [1, 0]:
        class_predictions = predictions[labels == actual]
        predicted_same = int((class_predictions == 1).sum())
        predicted_different = len(class_predictions) - predicted_same
        rows.append(
            {
                "actual": actual,
                "pred_1": predicted_same,
                "pred_0": predicted_different,
                "pred_1_percent": 100 * predicted_same / len(class_predictions),
                "pred_0_percent": 100 * predicted_different / len(class_predictions),
            },
        )
    return pd.DataFrame(rows)
