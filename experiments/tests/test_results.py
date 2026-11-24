"""Tests of avcore.results on constructed prediction and runs tables."""

from pathlib import Path

import pandas as pd
import pytest

from avcore import results
from avcore.evaluation import METRIC_COLUMNS


def make_run(run_name: str, family: str, model: str, f1: float) -> dict[str, object]:
    """Return one runs-table row whose four metrics all equal f1."""
    run = {
        "run_name": run_name,
        "family": family,
        "model": model,
        "representation": "none",
        "combination": "none",
        "classifier": "prompt",
    }
    for metric in METRIC_COLUMNS:
        run[metric] = f1
    return run


def write_predictions(run_dir: Path, predictions: list[int]) -> None:
    """Write a predictions-test.csv of ten pairs, five same-author ones first."""
    run_dir.mkdir()
    predictions_df = pd.DataFrame(
        {
            "id": [f"{number:05d}" for number in range(10)],
            "same": [1] * 5 + [0] * 5,
            "score": [0.5] * 10,
            "pred": predictions,
        },
    )
    predictions_df.to_csv(run_dir / results.TEST_PREDICTIONS_FILE, index=False)


def test_compute_test_metrics_gives_unrounded_percent() -> None:
    """Two of three right per class give 200 / 3 % for every metric, not 66.67."""
    predictions_df = pd.DataFrame(
        {"same": [1, 1, 1, 0, 0, 0], "pred": [1, 1, 0, 0, 0, 1]},
    )
    metrics = results.compute_test_metrics(predictions_df)
    for metric in METRIC_COLUMNS:
        assert metrics[metric] == pytest.approx(200 / 3, abs=1e-9)
        assert metrics[metric] != 66.67


def test_average_runs_means_two_runs_of_one_identity() -> None:
    """Two runs with the same identity become one row with the mean and n_runs 2."""
    runs_df = pd.DataFrame(
        [
            make_run("granite_r2_v2", "llm", "granite", 50.0),
            make_run("granite_v2", "llm", "granite", 60.0),
            make_run("gemma_v2", "llm", "gemma", 40.0),
        ],
    )
    averaged_df = results.average_runs(runs_df).set_index("model")
    assert averaged_df.loc["granite", "f1"] == 55.0
    assert averaged_df.loc["granite", "n_runs"] == 2
    assert averaged_df.loc["granite", "run_name"] == "granite_r2_v2+granite_v2"
    assert averaged_df.loc["gemma", "n_runs"] == 1


def test_build_result_table_ranks_by_f1_and_keeps_max_rows() -> None:
    """Only the table's families enter, best F1 first, cut to max_rows."""
    averaged_df = pd.DataFrame(
        [
            make_run("a_v2", "traditional", "none", 70.0),
            make_run("b_v2", "traditional", "none", 90.0),
            make_run("c_v2", "traditional", "none", 80.0),
            make_run("d_v2", "graph", "none", 99.0),
        ],
    ).assign(n_runs=1)
    table = results.TableDefinition(families=("traditional",), max_rows=2)
    table_df = results.build_result_table(averaged_df, table)
    assert table_df["run_name"].tolist() == ["b_v2", "c_v2"]


def test_build_replicate_table_gives_share_of_same_predictions(tmp_path: Path) -> None:
    """The untagged run is run_1; 2 of 10 differing pairs give 80 % same predictions."""
    first_predictions = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    second_predictions = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]
    write_predictions(tmp_path / "granite_r2_v2", first_predictions)
    write_predictions(tmp_path / "granite_v2", second_predictions)
    runs_df = pd.DataFrame(
        [
            make_run("granite_r2_v2", "llm", "granite", 60.0),
            make_run("granite_v2", "llm", "granite", 50.0),
        ],
    ).assign(n_unanswered=[7.0, 5.0], n_cut_by_context=0.0, n_cut_by_budget=0.0)
    replicate_df = results.build_replicate_table(runs_df, tmp_path)
    replicate_df = replicate_df.set_index("quantity")
    f1_row = replicate_df.loc["f1"]
    assert f1_row[["run_1", "run_2", "mean", "abs_delta"]].tolist() == [50, 60, 55, 10]
    assert replicate_df.loc["n_unanswered", "mean"] == 6.0
    assert replicate_df.loc["same_prediction", "mean"] == 80.0


def test_build_confusion_table_normalizes_per_actual_class() -> None:
    """Each actual class sums to 100 %, also when the classes differ in size."""
    predictions_df = pd.DataFrame(
        {"same": [1, 1, 1, 1, 0, 0], "pred": [1, 1, 1, 0, 0, 1]},
    )
    confusion_df = results.build_confusion_table(predictions_df)
    assert confusion_df.to_dict("records") == [
        {
            "actual": 1,
            "pred_1": 3,
            "pred_0": 1,
            "pred_1_percent": 75.0,
            "pred_0_percent": 25.0,
        },
        {
            "actual": 0,
            "pred_1": 1,
            "pred_0": 1,
            "pred_1_percent": 50.0,
            "pred_0_percent": 50.0,
        },
    ]


def test_build_replicate_table_skips_llm_counts_of_other_families(
    tmp_path: Path,
) -> None:
    """A repeated traditional run gives the four metric rows and no LLM count rows."""
    predictions = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    write_predictions(tmp_path / "svc_r2_v2", predictions)
    write_predictions(tmp_path / "svc_v2", predictions)
    runs_df = pd.DataFrame(
        [
            make_run("svc_r2_v2", "traditional", "none", 80.0),
            make_run("svc_v2", "traditional", "none", 82.0),
        ],
    ).assign(n_unanswered=None, n_cut_by_context=None, n_cut_by_budget=None)
    replicate_df = results.build_replicate_table(runs_df, tmp_path)
    expected_quantities = [*METRIC_COLUMNS, "same_prediction"]
    assert replicate_df["quantity"].tolist() == expected_quantities
