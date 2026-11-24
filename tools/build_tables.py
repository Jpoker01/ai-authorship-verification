"""Build the result tables of the thesis from the finished run folders, as CSV files.

On dt, from ~/diploma_thesis, once the runs are finished:
    PYTHONDONTWRITEBYTECODE=1 experiments/.venv/bin/python tools/build_tables.py \\
        --results-dir ~/results_v2 --output-dir experiments/results/v2/tables
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd
from avcore import results

THRESHOLDS_FILE = "thresholds-validation.csv"

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse the folder of the run folders and the folder the CSV files go to."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def write_csv(table_df: pd.DataFrame, path: Path) -> None:
    """Write a table as UTF-8 CSV with LF line ends and log its number of rows."""
    table_df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    logger.info(f"written {path.name} ({len(table_df)} rows)")


def write_figure_data(
    table_name: str,
    table_df: pd.DataFrame,
    results_dir: Path,
    output_dir: Path,
) -> None:
    """Write the confusion matrix and the F1-vs-threshold data of the table's best run.

    Args:
        table_name: Name of the table, the prefix of the file names.
        table_df: Output of results.build_result_table, best run first.
        results_dir: Folder of the run folders.
        output_dir: Folder the CSV files go to.
    """
    if table_df.empty:
        return
    best_run = table_df.iloc[0]
    if best_run["n_runs"] > 1:
        logger.warning(f"{table_name} - the best row is a mean of runs, no figure data")
        return
    run_dir = results_dir / best_run["run_name"]
    predictions_df = results.read_test_predictions(run_dir)
    confusion_df = results.build_confusion_table(predictions_df)
    write_csv(confusion_df, output_dir / f"{table_name}_confusion.csv")
    thresholds_path = run_dir / THRESHOLDS_FILE
    # only the threshold classifiers write a threshold sweep
    if thresholds_path.is_file():
        shutil.copy2(thresholds_path, output_dir / f"{table_name}_f1_threshold.csv")
        logger.info(f"written {table_name}_f1_threshold.csv")


def main() -> int:
    """Write the runs table, the result tables, their figure data and the replicates.

    Returns:
        0 once every file is written.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    arguments = parse_arguments()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    runs_df = results.collect_runs(arguments.results_dir)
    count_parts = []
    for family, count in runs_df["family"].value_counts().sort_index().items():
        count_parts.append(f"{family} {count}")
    logger.info(f"{len(runs_df)} runs - {', '.join(count_parts)}")
    averaged_df = results.average_runs(runs_df)
    for run_name in averaged_df.loc[averaged_df["n_runs"] > 1, "run_name"]:
        logger.warning(f"averaged into one row - {run_name}")

    write_csv(runs_df, arguments.output_dir / "runs.csv")
    for table_name, table in results.RESULT_TABLES.items():
        table_df = results.build_result_table(averaged_df, table)
        write_csv(table_df, arguments.output_dir / f"{table_name}.csv")
        write_figure_data(
            table_name=table_name,
            table_df=table_df,
            results_dir=arguments.results_dir,
            output_dir=arguments.output_dir,
        )
    replicate_df = results.build_replicate_table(runs_df, arguments.results_dir)
    if not replicate_df.empty:
        write_csv(replicate_df, arguments.output_dir / "replicates.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
