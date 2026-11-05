# HANDOFF - diploma_thesis_v2

2026-11-01 15:30

Purpose: fix the bugs found in the experiment code of the diploma thesis (authorship verification), re-run the affected experiments on Metacentrum and compare the results with the submitted thesis.

## WORKFLOW

1. This folder (diploma_thesis_v2) is a copy of the original repository without git history.
2. The fixes are implemented here and pushed to a NEW GitHub repository (e.g. Jpoker01/diploma_thesis_v2).
3. The new repository is cloned to Metacentrum, where the affected experiments are re-run.
4. The new results are compared with the submitted ones (see COMPARISON).

## FIXES TO IMPLEMENT

**1. Swapped accuracy and F1 (traditional and graph notebooks)**

- The function evaluate_results returns (accuracy, precision, recall, f1), but the result is unpacked as (top_f1, top_accuracy, top_precision, top_recall). Values are therefore stored under the wrong names and Optuna maximizes accuracy instead of F1.
- Fix: in EVERY call replace the left side with:

```txt
top_accuracy, top_precision, top_recall, top_f1 = evaluate_results(y_test, y_pred)
```

- The same line with y_val instead of y_test exists in the graph notebooks.
- Files (7 occurrences in each, i.e. one in objective() and six final evaluations):
    - experiments/notebooks/traditional/20251208_JP_traditional_word_TF-IDF_absolute_difference.ipynb
    - experiments/notebooks/traditional/20251208_JP_traditional_word_TF-IDF_element-wise.ipynb
    - experiments/notebooks/traditional/20251208_JP_traditional_word_BOW_absolute_difference.ipynb
    - experiments/notebooks/traditional/20251208_JP_traditional_word_BOW_element-wise.ipynb
    - experiments/notebooks/traditional/20251208_JP_traditional_char_TF-IDF_absolute_difference.ipynb
    - experiments/notebooks/graph/20251127_JP_integrated_syntactic_graph_absolute_difference.ipynb
    - experiments/notebooks/graph/20251127_JP_integrated_syntactic_graph_element-wise.ipynb
- The patch fix_metric_unpacking.patch (49 lines) does exactly this; it can be applied with "git apply" once the folder is a git repository.

**2. Validation run on test data (cosine similarity notebooks)**

- The function compute_similarities(train_df, test_df, vectorizer) ignores the parameter test_df and iterates over the global test_data_df, meaning the Optuna objective scored the TEST set instead of the validation set.
- Fix: inside compute_similarities replace every test_data_df with test_df (the loop header and the four .loc lines).
- Files:
    - experiments/notebooks/traditional/20251208_JP_traditional_word_TF-IDF_embedding.ipynb
    - experiments/notebooks/traditional/20251208_JP_traditional_word_BOW_embedding.ipynb

**3. Duplicated training pairs (transformers)**

- train_tuning_data_df is sampled from train_data_df, but the final training set is built as pd.concat([train_data_df, train_tuning_data_df, val_data_df]), meaning 10 000 (Longformer) or 19 999 (BigBird, ModernBERT) pairs are present twice.
- Fix: remove train_tuning_data_df from the concat.
- Files:
    - experiments/notebooks/transformer/20251127_JP_ longformer-base-4096_classification_head.ipynb
    - experiments/notebooks/transformer/20251127_JP_ bigbird-roberta-large_classification_head.ipynb
    - experiments/notebooks/transformer/20251127_JP_ ModernBERT-large_classification_head.ipynb

**4. Wrong model name (BigBird)**

- The notebooks load google/bigbird-roberta-base, the thesis states large. The code is correct, ONLY the thesis text is wrong. Optionally rename the notebook files to bigbird-roberta-base so the names match the code.

**5. Missing preprocessing in the web application**

- webapp/backend/core/ml.py sends the raw user text to the vectorizer. The clean_text function from experiments/notebooks/dataset/20251128_JP-transformer_dataset_preprocessing.ipynb has to be applied to both texts before transform.
- NOTE: this does not affect any experiment and is not part of the re-run. It should be fixed before the defense anyway.

## CHANGES REQUIRED FOR THE RE-RUN

These are not bug fixes, however without them the re-run would NOT produce a valid comparison:

- **Fresh Optuna studies.** Every notebook uses create_study(..., load_if_exists=True) with the same study name and SQLite file, and some use n_trials=30 - len(study.trials). Run as is, the notebook would resume the OLD study and run zero new trials. Either delete/move the old optuna_study_* files from the notebook folders, or change the study name (e.g. suffix "_v2").
- **Seed the sampler.** Add sampler=optuna.samplers.TPESampler(seed=42) to create_study. Without it the re-run is not reproducible.
- **New MLflow run names.** Add the suffix "_v2" to every run_name, so the old and new runs can be compared side by side in MLflow.
- **Log per-pair predictions.** Log the test predictions (id, actual, predicted) as an MLflow table in the traditional and graph notebooks, the same way the transformer notebooks already do. This is needed for confidence intervals and a McNemar test at the defense.
- Everything else (validation split, preprocessing, TF-IDF parameter ranges, classifier parameter ranges, 30 trials) stays EXACTLY the same, so that any difference comes from the fix only.

## WHAT TO RE-RUN AND IN WHICH ORDER

1. traditional_word_TF-IDF_absolute_difference - the deployed model; logistic regression first (~13h for 30 trials), then the other classifiers.
2. traditional_word_TF-IDF_element-wise, traditional_word_BOW_absolute_difference, traditional_word_BOW_element-wise.
3. integrated_syntactic_graph_absolute_difference, integrated_syntactic_graph_element-wise - the ISG vectors (*.npz) do NOT have to be rebuilt, only the classifiers.
4. traditional_word_TF-IDF_embedding, traditional_word_BOW_embedding - cheap, only threshold and vectorizer tuning.
5. Transformers - OPTIONAL. 2 - 4 days each on H100. The submitted numbers are valid as produced; the duplication only has to be disclosed. Re-run only if there is spare time, Longformer first.

Data on Metacentrum: the notebooks expect the cleaned datasets in experiments/data/01_processed/ and the ISG vectors in the same folder. Neither is in git (data/ is ignored), so they have to be copied from the original Metacentrum storage.

## COMPARISON

- Export the new MLflow runs the same way as prilohaC_MLFlow_all_experiments_export.xlsx.
- Compare per model: old F1 (correctly relabelled, i.e. the old top_recall column) vs. new F1, old best hyperparameters vs. new.
- Expected outcome: differences within about 0,5 p. b.; the selected hyperparameters may differ because the sampler was not seeded before.
- Reference values (old export, correctly relabelled):

| Model | F1 | Accuracy |
|---|---|---|
| TF-IDF abs. diff. + LinearSVC | 91,32 | 91,50 |
| TF-IDF abs. diff. + logistic regression | 91,08 | 91,28 |
| TF-IDF abs. diff. + XGBoost | 90,05 | 90,17 |
| BoW abs. diff. + logistic regression | 88,73 | 88,91 |
| BoW abs. diff. + LinearSVC | 87,17 | 87,65 |
| ISG abs. diff. + logistic regression | 86,26 | 86,69 |

## DO NOT FORGET

- The original repository (github.com/Jpoker01/diploma_thesis, commit 59d98ab) and the electronic appendix of the thesis stay UNTOUCHED until the supervisor replies!!!
- The re-run results go to the errata ONLY if the supervisor agrees; otherwise they serve as backup material for the defense.
- experiments/requirements.txt and conf/ are missing from the repository (gitignored). Export them from the Metacentrum environment this time (pip freeze) and commit them.
