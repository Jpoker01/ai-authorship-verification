# HANDOFF - diploma_thesis_v2

2026-11-06 09:30 (replaces the version from 2026-11-01)

Purpose: fix the bugs found in the experiment code of the diploma thesis (authorship verification), rebuild and re-run ALL experiments with one unified methodology and report the new results. Deadline of the final thesis version: 6.1.2027.

## WORKFLOW

- Repository: the ORIGINAL repository github.com/Jpoker01/diploma_thesis, branch v2. The submitted state stays on main (commit 59d98ab) until v2 is merged.
- Environment: uv. experiments/pyproject.toml + experiments/uv.lock define the exact versions; "uv sync --frozen" (add "--group transformer" or "--group llm") installs them. The transformer and llm groups conflict (different torch builds) and are NEVER installed together.
- Every notebook is first rebuilt and smoke tested locally in PyCharm (small sample, 1 - 2 trials), only then run on Metacentrum.
- Metacentrum = CERIT-SC Kubernetes Jupyter servers (A40 46GB). Each server has its OWN home volume, meaning uv, the repository and the data have to be set up on each server.
- Runs are started from the server terminal with nohup + papermill, so they continue after the Jupyter tab is closed. One server per long notebook for maximum parallelism.

## SCOPE AND ORDER

1. Faulty, longest first - the 3 transformer classification_head notebooks (full re-tuning, 30 trials).
2. Faulty, CPU - the 7 traditional notebooks and the 2 ISG classifier notebooks.
3. Unaffected, last - the 6 transformer embedding notebooks, the ISG embedding notebook and the 6 LLM notebooks.

- A notebook that is NOT re-run in time is reported with its v1 result. The thesis has to mark every result as v1 or v2, and the methodology has to state that the v1 runs used an unrecorded environment and an unseeded sampler.
- Therefore results_v1/, the v1 MLflow runs and the v1 notebook outputs (main branch) stay UNTOUCHED!!!

## FIXES

**1. Swapped accuracy and F1 (traditional and graph notebooks)** - DONE

- evaluate_results returns (accuracy, precision, recall, f1), but was unpacked as (top_f1, top_accuracy, top_precision, top_recall), meaning Optuna maximized accuracy instead of F1.

**2. Validation run on test data (word TF-IDF/BoW embedding notebooks)** - DONE

- compute_similarities ignored its test_df parameter and iterated over the global test_data_df, meaning the Optuna objective scored the TEST set.

**3. Duplicated training pairs (transformer classification_head)** - DONE

- The final training set was pd.concat([train_data_df, train_tuning_data_df, val_data_df]), while train_tuning_data_df is a sample of train_data_df, meaning 10 000 (Longformer) or 19 999 (BigBird, ModernBERT) pairs were present twice.

**4. Classifier dropout never applied (transformer classification_head)** - TO DO

- The tuned classifier_dropout was written into the config AFTER the model was built and the config was never passed to the model; the objective even assigned the range (0.0, 0.3) instead of the sampled value. Dropout therefore stayed at the pretrained default in ALL trials and final runs.
- Fix: apply the sampled value to the classification head after loading the model, in the objective AND in the final training.

**5. Wrong model name (BigBird)**

- The notebooks load google/bigbird-roberta-base, the thesis states large. Only the thesis text (and the notebook file names) is wrong.

**6. Missing preprocessing in the web application**

- webapp/backend/core/ml.py sends the raw user text to the vectorizer; clean_text has to be applied to both texts first. Does not affect any experiment.

## UNIFIED RE-RUN SETUP (ALL notebooks)

- Fresh Optuna studies - study name and SQLite file with the suffix "_v2", so the old study is NEVER resumed.
- Seeded sampler - optuna.samplers.TPESampler(seed=42); transformers.set_seed(42) in the transformer notebooks.
- 30 trials counted as COMPLETE + PRUNED only (MaxTrialsCallback), failed trials (e.g. OOM) are recorded but do not count.
- MLflow run names with the suffix "_v2"; library versions logged as tags and uv.lock as an artifact.
- Per-pair test predictions (id, actual, predicted) logged as an MLflow table in every notebook (confidence intervals, McNemar test at the defense).
- Everything else (data splits, preprocessing, parameter ranges) stays EXACTLY as in v1.

## DATA

- The notebooks expect the cleaned datasets and the ISG vectors (*.npz) in experiments/data/01_processed/. They are NOT in git and have to be copied to every server.

## COMPARISON

- Export the v2 MLflow runs the same way as prilohaC_MLFlow_all_experiments_export.xlsx.
- v1 reference values (old export, correctly relabelled):

| Model | F1 | Accuracy |
|---|---|---|
| TF-IDF abs. diff. + LinearSVC | 91,32 | 91,50 |
| TF-IDF abs. diff. + logistic regression | 91,08 | 91,28 |
| TF-IDF abs. diff. + XGBoost | 90,05 | 90,17 |
| BoW abs. diff. + logistic regression | 88,73 | 88,91 |
| BoW abs. diff. + LinearSVC | 87,17 | 87,65 |
| ISG abs. diff. + logistic regression | 86,26 | 86,69 |

## TO VERIFY

- warmup_ratio choices are [0.0, 0.5, 0.1] in the transformer notebooks; 0.5 looks like a typo for 0.05. It has to match what the thesis states BEFORE the re-tuning starts.
