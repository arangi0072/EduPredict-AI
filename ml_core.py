"""EduPredict AI - core machine-learning code.

Everything that trains or evaluates a model lives here, with NO Streamlit imports.
`app.py` (the dashboard) and `run_experiments.py` (which produces the numbers and
figures for CA_REPORT.md) both call these functions, so the report and the app can
never disagree.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, make_scorer,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold,
                                     cross_val_score, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DATA_PATH = Path(__file__).resolve().parent / "student_performance.csv"
FEATURES = ["study_hours", "attendance_pct", "previous_score",
            "assignments_completed", "sleep_hours", "internet_access"]
TARGET = "final_score"
SEED = 42

REGRESSION = "regression"
CLASSIFICATION = "classification"

# The bundled CSV also has an `academic_risk` column built with a cut-off of 50, but
# only 12 of 600 records fall below 50 (2 % - about 2 positives in a 20 % test set),
# which is too few to evaluate a classifier. The app therefore derives the label from
# `final_score` with an adjustable cut-off, defaulting to 60.
DEFAULT_RISK_THRESHOLD = 60

BASELINE_PREFIX = "Baseline"
# F1 that returns 0 (instead of warning) when a model never predicts "at risk"
F1_SCORER = make_scorer(f1_score, zero_division=0)
CV_MAE = "CV MAE (lower is better)"
CV_F1 = "CV F1"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Read the CSV and check that the expected columns are there."""
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"{Path(path).name} is missing required column(s): {missing}")
    return df


def risk_labels(df: pd.DataFrame, threshold: float = DEFAULT_RISK_THRESHOLD) -> pd.Series:
    """1 = at risk (final score below `threshold`), 0 = not at risk."""
    return (df[TARGET] < threshold).astype(int)


def split_data(df: pd.DataFrame, task: str, test_size: float,
               risk_threshold: float = DEFAULT_RISK_THRESHOLD):
    """Train/test split. Classification is stratified so both sets keep the class ratio."""
    X = df[FEATURES]
    if task == REGRESSION:
        return train_test_split(X, df[TARGET], test_size=test_size, random_state=SEED)
    y = risk_labels(df, risk_threshold)
    if y.sum() < 2 or (1 - y).sum() < 2:
        raise ValueError("Need at least 2 records in each class - change the risk cut-off.")
    return train_test_split(X, y, test_size=test_size, random_state=SEED, stratify=y)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def _pipeline(model, scale: bool = False, poly: bool = False) -> Pipeline:
    """Median imputation -> (polynomial features) -> (scaling) -> model."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if poly:
        steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
    if scale or poly:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def make_regressors() -> dict:
    return {
        "Baseline (predict the mean)": _pipeline(DummyRegressor(strategy="mean")),
        "Linear Regression": _pipeline(LinearRegression()),
        "Ridge (L2)": _pipeline(Ridge(alpha=1.0), scale=True),
        "Lasso (L1)": _pipeline(Lasso(alpha=0.1, max_iter=10000), scale=True),
        "Polynomial Regression (degree 2)": _pipeline(Ridge(alpha=1.0), poly=True),
        "Decision Tree Regressor": _pipeline(DecisionTreeRegressor(max_depth=5, random_state=SEED)),
        "Random Forest Regressor": _pipeline(
            RandomForestRegressor(n_estimators=150, max_depth=8, random_state=SEED)),
        "SVM (RBF)": _pipeline(SVR(C=10, kernel="rbf"), scale=True),
    }


def make_classifiers(balanced: bool = False) -> dict:
    cw = "balanced" if balanced else None
    return {
        "Baseline (never at risk)": _pipeline(DummyClassifier(strategy="most_frequent")),
        "Logistic Regression": _pipeline(LogisticRegression(max_iter=2000, class_weight=cw), scale=True),
        "Linear SVM": _pipeline(SVC(kernel="linear", C=1, class_weight=cw), scale=True),
        "RBF SVM": _pipeline(SVC(kernel="rbf", C=1, class_weight=cw), scale=True),
        "Decision Tree": _pipeline(
            DecisionTreeClassifier(max_depth=5, class_weight=cw, random_state=SEED)),
        "Random Forest": _pipeline(
            RandomForestClassifier(n_estimators=150, max_depth=8, class_weight=cw, random_state=SEED)),
    }


def cv_splitter(task: str, folds: int):
    cls = KFold if task == REGRESSION else StratifiedKFold
    return cls(n_splits=folds, shuffle=True, random_state=SEED)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_predictions(task: str, y_true, y_pred) -> dict:
    """Test-set metrics, using the column names shown in the app and the report."""
    if task == REGRESSION:
        return {"Test MAE": mean_absolute_error(y_true, y_pred),
                "Test RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
                "Test R²": r2_score(y_true, y_pred)}
    return {"Test Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1-score": f1_score(y_true, y_pred, zero_division=0)}


def compare_models(task: str, models: dict, X_train, y_train, X_test, y_test, cv_folds: int):
    """Cross-validate every model on the training data, then score it on the test set.

    Returns (results table, {name: fitted pipeline}, best model name). The best model
    is chosen by cross-validation score only - the test set is never used to choose -
    and the baseline is never eligible.
    """
    scoring = "neg_mean_absolute_error" if task == REGRESSION else F1_SCORER
    cv = cv_splitter(task, cv_folds)
    rows, fitted = [], {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        model.fit(X_train, y_train)
        row = {"Model": name}
        row[CV_MAE if task == REGRESSION else CV_F1] = -scores.mean() if task == REGRESSION else scores.mean()
        row.update(evaluate_predictions(task, y_test, model.predict(X_test)))
        rows.append(row)
        fitted[name] = model
    key = CV_MAE if task == REGRESSION else CV_F1
    results = (pd.DataFrame(rows)
               .sort_values(key, ascending=(task == REGRESSION), kind="stable")
               .reset_index(drop=True))
    contenders = results[~results["Model"].str.startswith(BASELINE_PREFIX)]
    return results, fitted, contenders.iloc[0]["Model"]


def complexity_curve(task: str, X_train, y_train, cv_folds: int,
                     balanced: bool = False, max_depth: int = 15) -> pd.DataFrame:
    """Train vs cross-validated score of a decision tree as its depth grows.

    The gap between the two lines is overfitting: a deep tree memorises the
    training data but stops improving (or gets worse) on unseen folds.
    Regression reports MAE (lower is better); classification reports F1 (higher is better).
    """
    cv = cv_splitter(task, cv_folds)
    rows = []
    for depth in range(1, max_depth + 1):
        if task == REGRESSION:
            model = _pipeline(DecisionTreeRegressor(max_depth=depth, random_state=SEED))
            scoring, sign = "neg_mean_absolute_error", -1
        else:
            model = _pipeline(DecisionTreeClassifier(
                max_depth=depth, random_state=SEED, class_weight="balanced" if balanced else None))
            scoring, sign = F1_SCORER, 1
        res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True)
        rows.append({"max_depth": depth,
                     "Training score": sign * res["train_score"].mean(),
                     "Cross-validation score": sign * res["test_score"].mean()})
    return pd.DataFrame(rows)


def best_depth(curve: pd.DataFrame, task: str) -> int:
    """Smallest depth that reaches the best cross-validation score."""
    col = "Cross-validation score"
    target = curve[col].min() if task == REGRESSION else curve[col].max()
    return int(curve.loc[np.isclose(curve[col], target), "max_depth"].min())


def tune_model(task: str, X_train, y_train, cv_folds: int, balanced: bool = False) -> GridSearchCV:
    """GridSearchCV: Random Forest for regression, SVM for classification."""
    cv = cv_splitter(task, cv_folds)
    if task == REGRESSION:
        pipe = _pipeline(RandomForestRegressor(random_state=SEED))
        grid = {"model__n_estimators": [100, 150], "model__max_depth": [4, 6, 8, None]}
        scoring = "neg_mean_absolute_error"
    else:
        pipe = _pipeline(SVC(class_weight="balanced" if balanced else None), scale=True)
        grid = [{"model__kernel": ["linear"], "model__C": [0.1, 1, 10]},
                {"model__kernel": ["rbf"], "model__C": [0.1, 1, 10], "model__gamma": ["scale", "auto"]}]
        scoring = F1_SCORER
    return GridSearchCV(pipe, grid, cv=cv, scoring=scoring, n_jobs=-1).fit(X_train, y_train)


def tuning_table(search: GridSearchCV, task: str, top: int = 5) -> pd.DataFrame:
    """Top GridSearchCV candidates as a readable table."""
    res = pd.DataFrame(search.cv_results_)
    # Built from the raw dicts so that max_depth=None shows as "None" (not NaN) and a
    # parameter that does not apply (gamma for a linear kernel) shows as a dash.
    rows = [{k.replace("model__", ""): ("None" if v is None else v) for k, v in p.items()}
            for p in res["params"]]
    params = pd.DataFrame(rows).astype(object).fillna("–").astype(str)
    score = -res["mean_test_score"] if task == REGRESSION else res["mean_test_score"]
    out = params.assign(**{"CV MAE" if task == REGRESSION else "CV F1": score.round(3),
                           "rank": res["rank_test_score"]})
    return out.sort_values("rank").head(top).reset_index(drop=True)


def feature_importance(task: str, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """Random Forest impurity importance + permutation importance on the test set.

    Permutation importance = how much the score gets worse when one feature's values
    are shuffled. Regression: increase in MAE. Classification: drop in ROC-AUC.
    """
    if task == REGRESSION:
        model = _pipeline(RandomForestRegressor(n_estimators=120, max_depth=8, random_state=SEED))
        scoring = "neg_mean_absolute_error"
    else:
        model = _pipeline(RandomForestClassifier(n_estimators=120, max_depth=8, random_state=SEED))
        scoring = "roc_auc" if pd.Series(y_test).nunique() > 1 else "accuracy"
    model.fit(X_train, y_train)
    perm = permutation_importance(model, X_test, y_test, scoring=scoring, n_repeats=20,
                                  random_state=SEED)
    return pd.DataFrame({"Impurity importance": model.named_steps["model"].feature_importances_,
                         "Permutation importance": perm.importances_mean,
                         "Permutation std": perm.importances_std},
                        index=FEATURES)


def fit_small_tree(task: str, X_train, y_train, depth: int = 3, balanced: bool = False):
    """A shallow tree that is small enough to read when plotted."""
    if task == REGRESSION:
        tree = DecisionTreeRegressor(max_depth=depth, random_state=SEED)
    else:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=SEED,
                                      class_weight="balanced" if balanced else None)
    return tree.fit(X_train, y_train)


# --------------------------------------------------------------------------- #
# Single prediction (used by the app's form)
# --------------------------------------------------------------------------- #
def predict_one(model, task: str, values: dict) -> dict:
    row = pd.DataFrame([values], columns=FEATURES)
    if task == REGRESSION:
        return {"score": float(np.clip(model.predict(row)[0], 0, 100))}
    out = {"risk": int(model.predict(row)[0]), "probability": None}
    if hasattr(model, "predict_proba"):
        out["probability"] = float(model.predict_proba(row)[0][1])
    return out
