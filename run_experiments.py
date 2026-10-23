"""Reproduce every number, table and figure used in CA_REPORT.md.

    python run_experiments.py

Writes figures/*.png, results/results.json and results/tables.md. It uses the same
functions as the Streamlit app (ml_core.py, plots.py), with the app's default settings:
20 % test data, 5-fold cross-validation, at-risk cut-off 60, seed 42.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import ml_core as core
import plots

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
FIG, RES = ROOT / "figures", ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

TEST, FOLDS, THRESHOLD = 0.20, 5, core.DEFAULT_RISK_THRESHOLD
THRESHOLDS = [50, 55, 60, 65, 70]
SEEDS = range(10)


def save(fig, name):
    fig.savefig(FIG / name, dpi=150, facecolor=fig.get_facecolor())


def md_table(df: pd.DataFrame, digits: int = 3, index: bool = False) -> str:
    d = df.reset_index() if index else df
    def fmt(v):
        if not isinstance(v, (float, np.floating)):
            return str(v)
        text = f"{v:.{digits}f}"
        return text.lstrip("-") if float(text) == 0 else text      # avoid "-0.000"
    lines = ["| " + " | ".join(map(str, d.columns)) + " |",
             "|" + "|".join("---" for _ in d.columns) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join(lines)


def run(task, threshold=THRESHOLD, balanced=False, test=TEST, folds=FOLDS):
    df = core.load_data()
    X_train, X_test, y_train, y_test = core.split_data(df, task, test, threshold)
    models = core.make_regressors() if task == core.REGRESSION else core.make_classifiers(balanced)
    results, fitted, best = core.compare_models(task, models, X_train, y_train, X_test, y_test, folds)
    return dict(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                results=results, fitted=fitted, best=best)


def repeated_holdout(task, threshold=THRESHOLD):
    """Mean +/- std of test metrics over 10 different random train/test splits."""
    df = core.load_data()
    X = df[core.FEATURES]
    y = df[core.TARGET] if task == core.REGRESSION else core.risk_labels(df, threshold)
    rows = []
    for seed in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST, random_state=seed,
                                              stratify=None if task == core.REGRESSION else y)
        models = core.make_regressors() if task == core.REGRESSION else core.make_classifiers()
        for name, m in models.items():
            m.fit(Xtr, ytr)
            rows.append({"Model": name, "seed": seed,
                         **core.evaluate_predictions(task, yte, m.predict(Xte))})
    d = pd.DataFrame(rows)
    keys = ["Test MAE", "Test R²"] if task == core.REGRESSION else ["F1-score", "Recall", "Precision"]
    g = d.groupby("Model", sort=False)[keys].agg(["mean", "std"])
    out = pd.DataFrame({"Model": g.index})
    for k in keys:
        out[k + " (mean ± std)"] = [f"{g[(k, 'mean')][m]:.3f} ± {g[(k, 'std')][m]:.3f}" for m in g.index]
    out["_sort"] = [g[(keys[0], "mean")][m] for m in g.index]
    return out.sort_values("_sort", ascending=(task == core.REGRESSION)).drop(columns="_sort").reset_index(drop=True)


def main():
    df = core.load_data()
    out, tables = {}, []

    def section(title, body):
        tables.append(f"### {title}\n\n{body}\n")

    # ---- dataset ----------------------------------------------------------
    desc = df[core.FEATURES + [core.TARGET]].describe().T[["mean", "std", "min", "50%", "max"]].rename(
        columns={"50%": "median"})
    section("Descriptive statistics", md_table(desc, 2, index=True).replace("| index |", "| Variable |"))
    corr = df[core.FEATURES + [core.TARGET]].corr()[core.TARGET].drop(core.TARGET).sort_values(ascending=False)
    section("Correlation with final score", md_table(corr.rename("Pearson r").to_frame(), 3, index=True)
            .replace("| index |", "| Feature |"))
    # Descriptive only: a linear model fitted on all 600 rows, to read effects in "points per unit".
    from sklearn.linear_model import LinearRegression
    lin = LinearRegression().fit(df[core.FEATURES], df[core.TARGET])
    coef = pd.Series(lin.coef_, index=core.FEATURES)
    spread = (coef * df[core.FEATURES].std()).rename("Points per 1 std dev")
    coef_df = pd.DataFrame({"Points per unit": coef, "Points per 1 std dev": spread}).sort_values(
        "Points per 1 std dev", ascending=False)
    section("Linear-regression coefficients (fitted on all 600 rows, descriptive only)",
            md_table(coef_df, 2, index=True).replace("| index |", "| Feature |")
            + f"\n\nIntercept: {lin.intercept_:.2f}; in-sample R² = {lin.score(df[core.FEATURES], df[core.TARGET]):.3f}")
    out["dataset"] = {"rows": len(df), "ceiling_100": int((df[core.TARGET] == 100).sum()),
                      "csv_risk_lt50": int(df.academic_risk.sum()),
                      "risk_counts": {str(t): int((df[core.TARGET] < t).sum()) for t in THRESHOLDS},
                      "mean_score": df[core.TARGET].mean(), "corr": corr.round(3).to_dict(),
                      "linear_coef": coef_df.round(3).to_dict("index"), "linear_intercept": float(lin.intercept_)}
    save(plots.study_hours_scatter(df), "study_hours_scatter.png")
    save(plots.correlation_heatmap(df), "correlation_matrix.png")

    # ---- regression -------------------------------------------------------
    reg = run(core.REGRESSION)
    section("Regression: model comparison (20 % test, 5-fold CV)", md_table(reg["results"]))
    out["regression"] = {"best": reg["best"], "results": reg["results"].round(4).to_dict("records"),
                         "n_train": len(reg["y_train"]), "n_test": len(reg["y_test"])}
    best_pred = reg["fitted"][reg["best"]].predict(reg["X_test"])
    save(plots.model_comparison(reg["results"], core.REGRESSION), "regression_model_comparison.png")
    save(plots.actual_vs_predicted(reg["y_test"], best_pred, reg["best"]), "regression_actual_vs_predicted.png")
    resid = reg["y_test"].to_numpy() - best_pred
    out["regression"]["residual_mean"], out["regression"]["residual_std"] = float(resid.mean()), float(resid.std())
    stab = repeated_holdout(core.REGRESSION)
    section("Regression: stability over 10 random splits", md_table(stab))
    out["regression"]["stability"] = stab.to_dict("records")

    # ---- classification (default cut-off) ---------------------------------
    cls = run(core.CLASSIFICATION)
    section(f"Classification: model comparison (cut-off {THRESHOLD}, 20 % test, 5-fold CV)", md_table(cls["results"]))
    cm = confusion_matrix(cls["y_test"], cls["fitted"][cls["best"]].predict(cls["X_test"]), labels=[0, 1])
    out["classification"] = {"best": cls["best"], "results": cls["results"].round(4).to_dict("records"),
                             "n_train": len(cls["y_train"]), "n_test": len(cls["y_test"]),
                             "pos_train": int(cls["y_train"].sum()), "pos_test": int(cls["y_test"].sum()),
                             "confusion": cm.tolist()}
    save(plots.model_comparison(cls["results"], core.CLASSIFICATION), "classification_model_comparison.png")
    save(plots.confusion_matrix_chart(cm, cls["best"]), "classification_confusion_matrix.png")
    bal = run(core.CLASSIFICATION, balanced=True)
    section("Classification with class_weight='balanced'", md_table(bal["results"]))
    out["classification"]["balanced"] = {"best": bal["best"], "results": bal["results"].round(4).to_dict("records")}
    stab_c = repeated_holdout(core.CLASSIFICATION)
    section("Classification: stability over 10 random splits", md_table(stab_c))
    out["classification"]["stability"] = stab_c.to_dict("records")

    # ---- cut-off sensitivity ---------------------------------------------
    rows = []
    for t in THRESHOLDS:
        r = run(core.CLASSIFICATION, threshold=t)
        top = r["results"][r["results"]["Model"] == r["best"]].iloc[0]
        base = r["results"][r["results"]["Model"].str.startswith("Baseline")].iloc[0]
        rows.append({"Cut-off": t, "At-risk records": int((df[core.TARGET] < t).sum()),
                     "Train / test positives": f"{int(r['y_train'].sum())} / {int(r['y_test'].sum())}",
                     "Best model (CV F1)": r["best"], "CV F1": top["CV F1"], "Test precision": top["Precision"],
                     "Test recall": top["Recall"], "Test F1": top["F1-score"],
                     "Accuracy": top["Test Accuracy"], "Baseline accuracy": base["Test Accuracy"]})
    sens = pd.DataFrame(rows)
    section("Sensitivity to the at-risk cut-off", md_table(sens))
    out["sensitivity"] = sens.round(4).to_dict("records")

    # ---- overfitting ------------------------------------------------------
    out["complexity"] = {}
    for task, run_ in ((core.REGRESSION, reg), (core.CLASSIFICATION, cls)):
        curve = core.complexity_curve(task, run_["X_train"], run_["y_train"], FOLDS)
        bd = core.best_depth(curve, task)
        save(plots.complexity_chart(curve, task, bd), f"{task}_overfitting_curve.png")
        shown = curve[curve.max_depth.isin([1, 2, 3, 4, 5, 6, 8, 10, 15])].rename(columns={"max_depth": "max_depth"})
        section(f"Overfitting curve ({task})", md_table(shown) + f"\n\nBest cross-validation depth: **{bd}**")
        out["complexity"][task] = {"best_depth": bd, "curve": curve.round(4).to_dict("records")}
        save(plots.tree_chart(core.fit_small_tree(task, run_["X_train"], run_["y_train"], 3), task),
             f"{task}_decision_tree.png")

    # ---- tuning -----------------------------------------------------------
    out["tuning"] = {}
    for task, run_ in ((core.REGRESSION, reg), (core.CLASSIFICATION, cls)):
        s = core.tune_model(task, run_["X_train"], run_["y_train"], FOLDS)
        test = core.evaluate_predictions(task, run_["y_test"], s.best_estimator_.predict(run_["X_test"]))
        params = {k.replace("model__", ""): v for k, v in s.best_params_.items()}
        score = -s.best_score_ if task == core.REGRESSION else s.best_score_
        section(f"GridSearchCV ({task})", f"Best parameters: `{params}`; best CV "
                f"{'MAE' if task == core.REGRESSION else 'F1'} = {score:.3f}\n\n"
                + md_table(core.tuning_table(s, task)) + "\n\nTest metrics of the tuned model: "
                + ", ".join(f"{k} = {v:.3f}" for k, v in test.items()))
        out["tuning"][task] = {"params": {k: (None if v is None else v) for k, v in params.items()},
                               "cv": score, "test": {k: float(v) for k, v in test.items()},
                               "n_candidates": len(s.cv_results_["params"])}

    # ---- importance -------------------------------------------------------
    out["importance"] = {}
    for task, run_ in ((core.REGRESSION, reg), (core.CLASSIFICATION, cls)):
        imp = core.feature_importance(task, run_["X_train"], run_["y_train"], run_["X_test"], run_["y_test"])
        save(plots.importance_chart(imp, task), f"{task}_feature_importance.png")
        section(f"Feature importance ({task})", md_table(imp.sort_values("Permutation importance", ascending=False), 3, index=True)
                .replace("| index |", "| Feature |"))
        out["importance"][task] = imp.round(4).to_dict("index")

    (RES / "results.json").write_text(json.dumps(out, indent=2, default=float))
    (RES / "tables.md").write_text("# Generated tables\n\n" + "\n".join(tables), encoding="utf-8")
    print(f"Wrote {len(list(FIG.glob('*.png')))} figures to {FIG} and results to {RES}")


if __name__ == "__main__":
    main()
