"""Quick checks for EduPredict AI.  Run:  python test_project.py   (or: pytest)

They test the data and the ML code in ml_core.py with loose expectations, so they keep
passing across scikit-learn versions while still catching a broken pipeline.
"""
import warnings

import numpy as np
import pandas as pd

import ml_core as core

warnings.filterwarnings("ignore")
df = core.load_data()


def test_dataset_is_clean():
    assert df.shape == (600, 8)
    assert not df.isna().any().any()
    assert df[core.TARGET].between(0, 100).all()
    assert df.attendance_pct.between(0, 100).all() and df.assignments_completed.between(0, 10).all()
    assert set(df.internet_access) == {0, 1}


def test_risk_label_uses_the_cutoff():
    assert core.risk_labels(df, 60).sum() == (df[core.TARGET] < 60).sum() == 55
    assert core.risk_labels(df, 50).sum() == 12


def test_risk_label_boundary_is_strictly_below():
    edge = pd.DataFrame({core.TARGET: [59.9, 60.0, 60.1]})      # no real record sits exactly on 60
    assert core.risk_labels(edge, 60).tolist() == [1, 0, 0]


def test_splits_do_not_leak_and_are_stratified():
    X_train, X_test, y_train, y_test = core.split_data(df, core.CLASSIFICATION, 0.2, 60)
    assert len(X_test) == 120 and len(X_train) == 480
    assert set(X_train.index).isdisjoint(X_test.index)
    assert abs(y_train.mean() - y_test.mean()) < 0.02          # same at-risk share in both


def test_regression_models_beat_the_baseline():
    X_train, X_test, y_train, y_test = core.split_data(df, core.REGRESSION, 0.2)
    results, fitted, best = core.compare_models(core.REGRESSION, core.make_regressors(),
                                                X_train, y_train, X_test, y_test, 5)
    assert not best.startswith(core.BASELINE_PREFIX)
    row = results.set_index("Model")
    assert row.loc[best, "Test MAE"] < 0.6 * row.loc["Baseline (predict the mean)", "Test MAE"]
    assert row.loc[best, "Test R²"] > 0.7


def test_classifiers_beat_the_baseline_on_f1():
    X_train, X_test, y_train, y_test = core.split_data(df, core.CLASSIFICATION, 0.2, 60)
    results, fitted, best = core.compare_models(core.CLASSIFICATION, core.make_classifiers(),
                                                X_train, y_train, X_test, y_test, 5)
    row = results.set_index("Model")
    assert row.loc["Baseline (never at risk)", "F1-score"] == 0
    assert row.loc[best, core.CV_F1] > 0.5 and not best.startswith(core.BASELINE_PREFIX)


def test_deep_trees_overfit():
    X_train, _, y_train, _ = core.split_data(df, core.REGRESSION, 0.2)
    curve = core.complexity_curve(core.REGRESSION, X_train, y_train, 3, max_depth=12)
    deep = curve.iloc[-1]
    assert deep["Training score"] < 0.1 * deep["Cross-validation score"]   # memorised training data
    assert 2 <= core.best_depth(curve, core.REGRESSION) <= 8


def test_single_prediction_is_valid():
    X_train, X_test, y_train, y_test = core.split_data(df, core.REGRESSION, 0.2)
    model = core.make_regressors()["Linear Regression"].fit(X_train, y_train)
    typical = dict(study_hours=3, attendance_pct=75, previous_score=65,
                   assignments_completed=7, sleep_hours=7, internet_access=1)
    extreme = dict(study_hours=12, attendance_pct=100, previous_score=100,
                   assignments_completed=10, sleep_hours=12, internet_access=1)
    assert 0 <= core.predict_one(model, core.REGRESSION, typical)["score"] <= 100
    assert core.predict_one(model, core.REGRESSION, extreme)["score"] == 100      # clipped

    Xc, _, yc, _ = core.split_data(df, core.CLASSIFICATION, 0.2, 60)
    clf = core.make_classifiers()["Logistic Regression"].fit(Xc, yc)
    out = core.predict_one(clf, core.CLASSIFICATION, typical)
    assert out["risk"] in (0, 1) and 0 <= out["probability"] <= 1
    weak = dict(study_hours=0.5, attendance_pct=45, previous_score=30,
                assignments_completed=0, sleep_hours=4, internet_access=0)
    assert core.predict_one(clf, core.CLASSIFICATION, weak)["risk"] == 1


def test_permutation_importance_finds_the_strong_features():
    X_train, X_test, y_train, y_test = core.split_data(df, core.REGRESSION, 0.2)
    imp = core.feature_importance(core.REGRESSION, X_train, y_train, X_test, y_test)
    top2 = set(imp["Permutation importance"].nlargest(2).index)
    assert top2 == {"previous_score", "study_hours"}
    assert np.isclose(imp["Impurity importance"].sum(), 1.0)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print("ok  ", name)
    print(f"\nAll {len(tests)} checks passed.")
