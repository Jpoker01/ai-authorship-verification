import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import optuna
import pandas as pd
import pytest
import scipy.sparse as sp
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from avcore import classifiers, runs, tracking
from avcore.classifiers import (
    CLASSIFIER_FIXED_PARAMS,
    CLASSIFIER_NAMES,
    CLASSIFIER_SEARCH_SPACES,
    SEED,
    PairPredictions,
    build_classifier,
    build_prediction_table,
    predict_pairs,
    suggest_params,
)

# the scikit-learn method that gives the score of every classifier
SCORE_METHODS = {
    "multinomial_naive_bayes": "predict_proba",
    "logistic_regression": "predict_proba",
    "linear_svc": "decision_function",
    "k-nearest_neighbors": "predict_proba",
    "random_forest": "predict_proba",
    "xgboost": "predict_proba",
}


class RecordingTrial:
    """A stand-in for an Optuna trial that records every suggestion and returns the low end."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def suggest_categorical(self, name: str, choices: list) -> object:
        self.calls.append((name, CategoricalDistribution(choices)))
        return choices[0]

    def suggest_int(self, name: str, low: int, high: int) -> int:
        self.calls.append((name, IntDistribution(low, high)))
        return low

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        log: bool = False,
    ) -> float:
        self.calls.append((name, FloatDistribution(low, high, log=log)))
        return low


class PairsData(NamedTuple):
    """Synthetic pair features and their bool labels (True = same author)."""

    features: sp.csr_matrix
    labels: np.ndarray


def _pairs_data(n_pairs: int, seed: int) -> PairsData:
    """Non-negative sparse pair features (MNB needs counts) with a bool label, as in the notebooks.

    Pair i is a same-author pair (True) when i is even; same pairs have larger values in the
    first five columns.
    """
    rng = np.random.default_rng(seed)
    labels = np.arange(n_pairs) % 2 == 0
    counts = rng.poisson(1.0, size=(n_pairs, 20)).astype(np.float64)
    counts[labels, :5] += rng.poisson(3.0, size=(int(labels.sum()), 5))
    return PairsData(features=sp.csr_matrix(counts), labels=labels)


@pytest.fixture(scope="module")
def pairs_data() -> PairsData:
    """60 training pairs."""
    return _pairs_data(n_pairs=60, seed=1)


def _low_params(classifier_name: str) -> dict[str, object]:
    """The low end of every searched hyperparameter (a forest with 10 trees, to be fast)."""
    params = suggest_params(RecordingTrial(), CLASSIFIER_SEARCH_SPACES[classifier_name])
    if classifier_name == "random_forest":
        params["n_estimators"] = 10
    return params


def _fitted(
    classifier_name: str,
    features: sp.csr_matrix,
    labels: np.ndarray,
    n_jobs: int = 1,
) -> object:
    """The classifier with the low end of every searched hyperparameter, fitted on the pairs."""
    params = _low_params(classifier_name)
    classifier = build_classifier(classifier_name, params, n_jobs=n_jobs)
    classifier.fit(features, labels)
    return classifier


def test_classifier_tables_name_the_same_six_classifiers() -> None:
    """Every table holds the six classifiers of CLASSIFIER_NAMES, in the same order."""
    expected_names = list(SCORE_METHODS)

    assert CLASSIFIER_NAMES == expected_names
    assert list(CLASSIFIER_SEARCH_SPACES) == expected_names
    assert list(CLASSIFIER_FIXED_PARAMS) == expected_names
    # the classifiers whose fit or predict uses threads (LogReg does not for two classes)
    assert classifiers.THREADED_CLASSIFIERS == [
        "k-nearest_neighbors",
        "random_forest",
        "xgboost",
    ]
    assert classifiers.DECISION_FUNCTION_CLASSIFIERS == ["linear_svc"]


def test_classifier_classes_are_the_agreed_classes() -> None:
    """Every name builds the scikit-learn or XGBoost class the thesis names."""
    expected_classes = {
        "multinomial_naive_bayes": MultinomialNB,
        "logistic_regression": LogisticRegression,
        "linear_svc": LinearSVC,
        "k-nearest_neighbors": KNeighborsClassifier,
        "random_forest": RandomForestClassifier,
        "xgboost": XGBClassifier,
    }

    assert classifiers.CLASSIFIER_CLASSES == expected_classes


def test_search_spaces_suggest_the_params_in_a_fixed_order() -> None:
    """The params are suggested in a fixed order - the seeded sampler depends on it."""
    expected_order = {
        "multinomial_naive_bayes": ["alpha"],
        "logistic_regression": ["C", "penalty"],
        "linear_svc": ["C", "loss", "class_weight"],
        "k-nearest_neighbors": ["n_neighbors", "weights", "metric"],
        "random_forest": [
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
        ],
        "xgboost": [
            "eta",
            "max_depth",
            "subsample",
            "colsample_bytree",
            "reg_lambda",
            "reg_alpha",
        ],
    }

    for classifier_name, names in expected_order.items():
        assert list(CLASSIFIER_SEARCH_SPACES[classifier_name]) == names


def test_search_spaces_are_the_agreed_spaces() -> None:
    """Every searched hyperparameter has the agreed kind, range or choices."""
    expected = {
        "multinomial_naive_bayes": {
            "alpha": ("float_log", 1e-3, 1.0),
        },
        "logistic_regression": {
            "C": ("float_log", 0.01, 1_000.0),
            "penalty": ("categorical", "l1", "l2"),
        },
        "linear_svc": {
            "C": ("float_log", 0.01, 1_000.0),
            "loss": ("categorical", "hinge", "squared_hinge"),
            "class_weight": ("categorical", None, "balanced"),
        },
        "k-nearest_neighbors": {
            "n_neighbors": ("int", 1, 50),
            "weights": ("categorical", "uniform", "distance"),
            "metric": ("categorical", "euclidean", "manhattan", "cosine"),
        },
        "random_forest": {
            "n_estimators": ("int", 100, 600),
            "max_depth": ("int", 5, 50),
            "min_samples_split": ("int", 2, 20),
            "min_samples_leaf": ("int", 1, 10),
        },
        "xgboost": {
            "eta": ("float", 0.01, 0.3),
            "max_depth": ("int", 3, 12),
            "subsample": ("float", 0.5, 1.0),
            "colsample_bytree": ("float", 0.5, 1.0),
            "reg_lambda": ("float_log", 1e-3, 10.0),
            "reg_alpha": ("float_log", 1e-3, 1.0),
        },
    }

    assert CLASSIFIER_SEARCH_SPACES == expected


def test_fixed_params_are_the_agreed_params() -> None:
    """The fixed params - seeds of the four random classifiers, liblinear, dual="auto", no n_jobs."""
    expected = {
        "multinomial_naive_bayes": {},
        "logistic_regression": {
            "solver": "liblinear",
            "max_iter": 100,
            "random_state": 42,
        },
        "linear_svc": {
            "penalty": "l2",
            "dual": "auto",
            "max_iter": 5000,
            "random_state": 42,
        },
        "k-nearest_neighbors": {"algorithm": "auto"},
        "random_forest": {"random_state": 42},
        "xgboost": {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_estimators": 300,
            "tree_method": "auto",
            "random_state": 42,
        },
    }

    assert CLASSIFIER_FIXED_PARAMS == expected


def test_search_spaces_are_accepted_as_study_settings(tmp_path: Path) -> None:
    """Every search space can be stored as a study setting (no nested list)."""
    study_path = tmp_path / "study.db"
    settings = {}
    for classifier_name, search_space in CLASSIFIER_SEARCH_SPACES.items():
        for name, space in search_space.items():
            settings[f"search.{classifier_name}.{name}"] = space

    runs.check_study_settings(study_path, settings)


def test_suggest_params_suggests_every_kind_in_the_order_of_the_space() -> None:
    """A categorical gets its flat choices as a list, float_log a log-scaled float."""
    trial = RecordingTrial()
    search_space = {
        "C": ("float_log", 0.01, 5.0),
        "class_weight": ("categorical", None, "balanced"),
        "n_neighbors": ("int", 1, 50),
        "eta": ("float", 0.01, 0.3),
    }
    expected_calls = [
        ("C", FloatDistribution(0.01, 5.0, log=True)),
        ("class_weight", CategoricalDistribution([None, "balanced"])),
        ("n_neighbors", IntDistribution(1, 50)),
        ("eta", FloatDistribution(0.01, 0.3)),
    ]

    params = suggest_params(trial, search_space)

    assert trial.calls == expected_calls
    assert params == {"C": 0.01, "class_weight": None, "n_neighbors": 1, "eta": 0.01}


def test_suggest_params_rejects_an_unknown_kind() -> None:
    """A space whose kind is not categorical, int, float or float_log stops the trial."""
    trial = RecordingTrial()
    search_space = {"alpha": ("normal", 0.0, 1.0)}

    with pytest.raises(ValueError, match="Unknown kind 'normal' of alpha"):
        suggest_params(trial, search_space)


def test_suggest_params_stays_inside_the_spaces_of_a_real_study() -> None:
    """Twenty seeded TPE trials of every classifier suggest values inside its space."""
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    n_trials = 20

    for search_space in CLASSIFIER_SEARCH_SPACES.values():
        for _ in range(n_trials):
            trial = study.ask()
            params = suggest_params(trial, search_space)
            for name, (kind, *arguments) in search_space.items():
                if kind == "categorical":
                    assert params[name] in arguments
                else:
                    assert arguments[0] <= params[name] <= arguments[1]
            study.tell(trial, 0.0)


@pytest.mark.parametrize("classifier_name", CLASSIFIER_NAMES)
def test_build_classifier_sets_the_searched_and_the_fixed_params(
    classifier_name: str,
) -> None:
    """The classifier gets the searched params, the fixed params and the threads if it takes them."""
    params = _low_params(classifier_name)
    n_jobs = 3

    classifier = build_classifier(classifier_name, params, n_jobs=n_jobs)

    actual_params = classifier.get_params()
    assert isinstance(classifier, classifiers.CLASSIFIER_CLASSES[classifier_name])

    for name, value in params.items():
        assert actual_params[name] == value
    for name, value in CLASSIFIER_FIXED_PARAMS[classifier_name].items():
        assert actual_params[name] == value
    if classifier_name in classifiers.THREADED_CLASSIFIERS:
        assert actual_params["n_jobs"] == n_jobs


@pytest.mark.parametrize(
    "classifier_name",
    ["logistic_regression", "linear_svc", "random_forest", "xgboost"],
)
def test_build_classifier_seeds_every_random_classifier(classifier_name: str) -> None:
    """LogReg, LinearSVC, RF and XGBoost get random_state=SEED."""
    params = _low_params(classifier_name)
    n_jobs = 1

    classifier = build_classifier(classifier_name, params, n_jobs=n_jobs)

    assert classifier.get_params()["random_state"] == SEED


def test_build_classifier_rejects_an_unknown_classifier() -> None:
    """A name that is not in CLASSIFIER_NAMES is refused."""
    classifier_name = "svm"
    params: dict[str, object] = {}
    n_jobs = 1

    with pytest.raises(ValueError, match="Unknown classifier 'svm'"):
        build_classifier(classifier_name, params, n_jobs=n_jobs)


@pytest.mark.parametrize(
    ("case", "params"),
    [
        (
            "vectorizer param of the best trial",
            {"C": 1.0, "loss": "hinge", "class_weight": None, "max_features": None},
        ),
        ("searched param missing", {"C": 1.0, "loss": "hinge"}),
        (
            "fixed param",
            {"C": 1.0, "loss": "hinge", "class_weight": None, "random_state": 0},
        ),
    ],
)
def test_build_classifier_takes_exactly_the_searched_params(
    case: str,
    params: dict[str, object],
) -> None:
    """Any other set of params is refused - an extra one, a missing one or a fixed one."""
    classifier_name = "linear_svc"
    n_jobs = 1

    with pytest.raises(ValueError, match="linear_svc takes the params"):
        build_classifier(classifier_name, params, n_jobs=n_jobs)


def test_build_classifier_gives_each_pipeline_its_own_scaler_copy(
    pairs_data: PairsData,
) -> None:
    """The pipeline fits its own copy of the scaler, so one scaler can be passed to every trial."""
    features = pairs_data.features
    labels = pairs_data.labels
    classifier_name = "linear_svc"
    scaler = StandardScaler(with_mean=False)
    params = {"C": 1.0, "loss": "squared_hinge", "class_weight": None}
    n_jobs = 1

    pipeline = build_classifier(classifier_name, params, n_jobs=n_jobs, scaler=scaler)
    pipeline.fit(features, labels)

    assert isinstance(pipeline, Pipeline)
    assert pipeline.named_steps["scaler"] is not scaler
    assert not hasattr(scaler, "scale_")
    assert pipeline.named_steps["classifier"].get_params()["C"] == 1.0


@pytest.mark.parametrize("classifier_name", CLASSIFIER_NAMES)
def test_predict_pairs_decides_with_predict_and_scores_with_the_probability(
    classifier_name: str,
    pairs_data: PairsData,
) -> None:
    """pred is predict as 0 / 1 (fitted on bool labels); score is the class-1 probability or the decision function."""
    features = pairs_data.features
    labels = pairs_data.labels
    classifier = _fitted(classifier_name, features, labels)
    score_method = getattr(classifier, SCORE_METHODS[classifier_name])
    expected_score = score_method(features)
    if expected_score.ndim == 2:
        expected_score = expected_score[:, 1]
    expected_pred = classifier.predict(features).astype(int)

    predictions = predict_pairs(classifier, classifier_name, features)

    assert predictions.pred.dtype == np.dtype(int)
    assert np.array_equal(predictions.pred, expected_pred)
    assert predictions.score.dtype == np.float64
    assert np.array_equal(predictions.score, expected_score.astype(np.float64))


def test_predict_pairs_keeps_the_decision_of_predict_on_a_linear_svc_tie(
    pairs_data: PairsData,
) -> None:
    """A LinearSVC score of exactly 0 is "different author", as its predict says (score > 0)."""
    features = pairs_data.features
    labels = pairs_data.labels
    classifier_name = "linear_svc"
    classifier = _fitted(classifier_name, features, labels)
    # every decision value becomes exactly 0
    classifier.coef_[:] = 0.0
    classifier.intercept_[:] = 0.0
    expected_pred = np.zeros(features.shape[0], dtype=int)

    predictions = predict_pairs(classifier, classifier_name, features)

    assert np.all(predictions.score == 0.0)
    assert np.array_equal(predictions.pred, expected_pred)


def test_predict_pairs_keeps_the_decision_of_predict_on_a_knn_tie() -> None:
    """Two neighbours that disagree give the probability 0.5 and "different author"."""
    fit_features = sp.csr_matrix(np.array([[0.0, 1.0], [0.0, 3.0]]))
    fit_labels = np.array([True, False])
    target_features = sp.csr_matrix(np.array([[0.0, 2.0]]))
    classifier_name = "k-nearest_neighbors"
    params = {"n_neighbors": 2, "weights": "uniform", "metric": "euclidean"}
    n_jobs = 1
    classifier = build_classifier(classifier_name, params, n_jobs=n_jobs)
    classifier.fit(fit_features, fit_labels)

    predictions = predict_pairs(classifier, classifier_name, target_features)

    assert predictions.score.tolist() == [0.5]
    assert predictions.pred.tolist() == [0]


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine"])
def test_predict_pairs_gives_knn_results_independent_of_the_thread_count(
    metric: str,
) -> None:
    """Tied neighbour distances are broken the same way with 1 and 2 OpenMP threads."""
    fit_data = _pairs_data(n_pairs=2_500, seed=2)
    target_data = _pairs_data(n_pairs=1_500, seed=3)
    classifier_name = "k-nearest_neighbors"
    params = {"n_neighbors": 4, "weights": "uniform", "metric": metric}
    predictions_by_threads = {}

    for threads in (1, 2):
        with threadpool_limits(limits=threads, user_api="openmp"):
            classifier = build_classifier(classifier_name, params, n_jobs=threads)
            classifier.fit(fit_data.features, fit_data.labels)
            predictions_by_threads[threads] = predict_pairs(
                classifier,
                classifier_name,
                target_data.features,
            )

    one_thread = predictions_by_threads[1]
    two_threads = predictions_by_threads[2]
    assert np.array_equal(one_thread.pred, two_threads.pred)
    assert np.array_equal(one_thread.score, two_threads.score)


@pytest.mark.parametrize("classifier_name", CLASSIFIER_NAMES)
def test_predict_pairs_gives_the_same_scores_in_two_runs_with_two_threads(
    classifier_name: str,
    pairs_data: PairsData,
) -> None:
    """Two fits with the same params and two threads give the same decisions and scores."""
    features = pairs_data.features
    labels = pairs_data.labels
    n_jobs = 2

    first_classifier = _fitted(classifier_name, features, labels, n_jobs)
    second_classifier = _fitted(classifier_name, features, labels, n_jobs)

    first = predict_pairs(first_classifier, classifier_name, features)
    second = predict_pairs(second_classifier, classifier_name, features)

    assert np.array_equal(first.pred, second.pred)
    assert np.array_equal(first.score, second.score)


@pytest.mark.parametrize(
    "scaler",
    [None, StandardScaler(with_mean=False)],
    ids=["alone", "in_a_pipeline"],
)
def test_predict_pairs_predicts_a_forest_with_one_thread(
    scaler: StandardScaler | None,
    pairs_data: PairsData,
) -> None:
    """A forest fitted with threads predicts with one, also inside a pipeline."""
    features = pairs_data.features
    labels = pairs_data.labels
    classifier_name = "random_forest"
    params = _low_params(classifier_name)
    n_jobs = 2
    classifier = build_classifier(classifier_name, params, n_jobs=n_jobs, scaler=scaler)
    classifier.fit(features, labels)

    predict_pairs(classifier, classifier_name, features)

    if scaler is None:
        forest = classifier
    else:
        forest = classifier.named_steps["classifier"]
    assert forest.get_params()["n_jobs"] == 1


@pytest.mark.parametrize(
    "classifier_name",
    ["linear_svc", "logistic_regression", "random_forest"],
)
def test_predict_pairs_predicts_with_the_whole_pipeline(
    classifier_name: str,
    pairs_data: PairsData,
) -> None:
    """predict_pairs predicts with the pipeline (scaler first), not only with the classifier in it."""
    features = pairs_data.features
    labels = pairs_data.labels
    scaler = StandardScaler(with_mean=False)
    params = _low_params(classifier_name)
    n_jobs = 2
    pipeline = build_classifier(classifier_name, params, n_jobs=n_jobs, scaler=scaler)
    pipeline.fit(features, labels)

    predictions = predict_pairs(pipeline, classifier_name, features)

    # the pipeline's own answer; asked after predict_pairs, which sets a forest to one
    # thread, so the forest adds up its votes in the same order both times
    pipeline_pred = pipeline.predict(features).astype(int)
    if classifier_name == "linear_svc":
        pipeline_score = pipeline.decision_function(features)
    else:
        same_author_column = 1
        pipeline_score = pipeline.predict_proba(features)[:, same_author_column]
    assert np.array_equal(predictions.pred, pipeline_pred)
    assert np.array_equal(predictions.score, pipeline_score)


def test_predict_pairs_rejects_an_unknown_classifier(
    pairs_data: PairsData,
) -> None:
    """A name that is not in CLASSIFIER_NAMES is refused."""
    features = pairs_data.features
    labels = pairs_data.labels
    classifier = _fitted("logistic_regression", features, labels)
    classifier_name = "logistic"

    with pytest.raises(ValueError, match="Unknown classifier 'logistic'"):
        predict_pairs(classifier, classifier_name, features)


def test_predict_pairs_refuses_labels_other_than_0_and_1(
    pairs_data: PairsData,
) -> None:
    """A classifier fitted on other labels would make column 1 of the probabilities wrong."""
    features = pairs_data.features
    labels = pairs_data.labels
    classifier_name = "logistic_regression"
    other_labels = labels.astype(int) + 1
    classifier = _fitted(classifier_name, features, other_labels)

    with pytest.raises(ValueError, match="not \\[0, 1\\]"):
        predict_pairs(classifier, classifier_name, features)


def test_build_prediction_table_holds_one_row_per_pair() -> None:
    """id, same (as 0 / 1), score and pred of every pair, in the order of the pairs."""
    pairs_df = pd.DataFrame({"id": ["007", "a"], "same": [True, False]})
    predictions = PairPredictions(score=np.array([0.9, -0.2]), pred=np.array([0, 0]))
    expected_df = pd.DataFrame(
        {
            "id": ["007", "a"],
            "same": [1, 0],
            "score": [0.9, -0.2],
            "pred": [0, 0],
        },
    )

    table_df = build_prediction_table(pairs_df, predictions)

    pd.testing.assert_frame_equal(table_df, expected_df)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("repeated id", "repeated id"),
        ("one score missing", "2 scores and 3 decisions for 3 pairs"),
        ("one decision missing", "3 scores and 2 decisions for 3 pairs"),
        ("NaN score", "NaN"),
    ],
)
def test_build_prediction_table_refuses_wrong_predictions(
    case: str,
    message: str,
) -> None:
    """Repeated ids, a missing prediction or a NaN score never reach the table."""
    pairs_df = pd.DataFrame({"id": ["a", "b", "c"], "same": [True, False, True]})
    score = np.array([0.1, 0.2, 0.3])
    pred = np.array([0, 0, 1])
    if case == "repeated id":
        pairs_df.loc[2, "id"] = "a"
    elif case == "one score missing":
        score = score[:2]
    elif case == "one decision missing":
        pred = pred[:2]
    elif case == "NaN score":
        score[1] = np.nan
    predictions = PairPredictions(score=score, pred=pred)

    with pytest.raises(ValueError, match=message):
        build_prediction_table(pairs_df, predictions)


def test_prediction_table_is_accepted_by_write_run_files(
    pairs_data: PairsData,
    tmp_path: Path,
) -> None:
    """A prediction table of real XGBoost predictions passes the checks of write_run_files."""
    features = pairs_data.features
    labels = pairs_data.labels
    classifier_name = "xgboost"
    pair_ids = []
    for pair_number in range(len(labels)):
        pair_ids.append(f"p{pair_number}")
    pairs_df = pd.DataFrame({"id": pair_ids, "same": labels})
    classifier = _fitted(classifier_name, features, labels)
    predictions = predict_pairs(classifier, classifier_name, features)
    table_df = build_prediction_table(pairs_df, predictions)
    fake_commit = "a" * 40
    context = tracking.RunContext(environment={"commit": fake_commit}, packages={})
    # write_run_files takes only an absolute folder with _v2 in its name that does not exist
    run_dir = tmp_path / "run_v2"

    tracking.write_run_files(
        run_dir,
        context=context,
        params={},
        metrics={},
        tables={"predictions-test": table_df},
        figures={},
    )

    manifest_path = run_dir / tracking.MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    written_files = manifest["files"]
    assert "predictions-test.csv" in written_files
