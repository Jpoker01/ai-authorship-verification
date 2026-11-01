"""Check what the Databricks MLflow server actually holds for the thesis experiment.

Run from experiments/notebooks/traditional on Metacentrum:
    python ../../../check_mlflow.py
"""
from dotenv import load_dotenv

load_dotenv("../../conf/local/.env")

import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("databricks")
client = MlflowClient()

print("== all experiments visible to this token:")
for e in client.search_experiments():
    print(f"  id={e.experiment_id}  name={e.name}  stage={e.lifecycle_stage}")

NAME = "/Users/xpokor17@mendelu.cz/JP_authorship_verification_dt"
exp = client.get_experiment_by_name(NAME)
print(f"\n== experiment {NAME}: {'FOUND id=' + exp.experiment_id if exp else 'NOT FOUND'}")
if exp:
    runs = client.search_runs([exp.experiment_id], run_view_type=3)  # ALL, incl. deleted
    print(f"   runs (incl. deleted): {len(runs)}")
    for r in runs:
        print(f"   - {r.info.run_name:70s} status={r.info.status:9s} stage={r.info.lifecycle_stage}  "
              f"metrics={ {k: v for k, v in r.data.metrics.items()} }")

print("\n== writing a fresh test run now ...")
mlflow.set_experiment(NAME)
with mlflow.start_run(run_name="check_mlflow_script") as run:
    mlflow.log_metric("top_f1", 12.34)
    print("   logged run_id", run.info.run_id, "-> experiment", run.info.experiment_id)
