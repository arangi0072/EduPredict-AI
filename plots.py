"""EduPredict AI - chart helpers.

Each function returns a matplotlib Figure. `app.py` shows them with st.pyplot and
`run_experiments.py` saves the same figures as PNGs for the report.

Charts follow one small design system: a light surface, hairline grid, two accent
colours (blue / orange) for the two series a chart ever needs, a blue->red diverging
scale for correlations, and a one-hue blue ramp for counts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from sklearn.tree import plot_tree

from ml_core import CV_F1, CV_MAE, FEATURES, REGRESSION, TARGET

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#3987e5", "#104281"])
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#2a78d6", "#f0efec", "#e34948"])

LABELS = {"study_hours": "Study hours", "attendance_pct": "Attendance %",
          "previous_score": "Previous score", "assignments_completed": "Assignments",
          "sleep_hours": "Sleep hours", "internet_access": "Internet access",
          "final_score": "Final score"}


def _fig(width=6.4, height=4.0, ncols=1, **kw) -> tuple:
    fig = Figure(figsize=(width, height), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, ncols, **kw)
    for ax in np.atleast_1d(axes):
        _style(ax)
    return fig, axes


def _style(ax, grid="y") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, color=AXIS)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=1, linestyle="-")
    else:
        ax.grid(False)


def _title(ax, text) -> None:
    ax.set_title(text, loc="left", fontsize=11, fontweight="bold", color=INK)


def _label(name: str) -> str:
    return LABELS.get(name, name)


# --------------------------------------------------------------------------- #
def study_hours_scatter(df: pd.DataFrame) -> Figure:
    fig, ax = _fig()
    ax.scatter(df.study_hours, df[TARGET], s=22, color=BLUE, alpha=.55, linewidths=0,
               label="Student record")
    slope, intercept = np.polyfit(df.study_hours, df[TARGET], 1)
    xs = np.array([df.study_hours.min(), df.study_hours.max()])
    ax.plot(xs, slope * xs + intercept, color=ORANGE, linewidth=2,
            label=f"Linear trend (+{slope:.1f} points per hour)")
    ax.set_xlabel("Study hours per day")
    ax.set_ylabel("Final score")
    _title(ax, "Study hours vs final score")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")
    return fig


def correlation_heatmap(df: pd.DataFrame) -> Figure:
    cols = FEATURES + [TARGET]
    corr = df[cols].corr()
    fig, ax = _fig(6.4, 5.0)
    _style(ax, grid=None)
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap=DIV, aspect="equal")
    ax.set_xticks(range(len(cols)), [_label(c) for c in cols], rotation=40, ha="right")
    ax.set_yticks(range(len(cols)), [_label(c) for c in cols])
    for i in range(len(cols)):
        for j in range(len(cols)):
            if i != j:
                v = corr.iloc[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > .65 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelsize=8, length=0)
    _title(ax, "Correlation matrix (Pearson r)")
    return fig


def model_comparison(results: pd.DataFrame, task: str) -> Figure:
    """Horizontal bars: cross-validation score vs held-out test score, best model labelled."""
    if task == REGRESSION:
        cv_col, test_col, xlabel = CV_MAE, "Test MAE", "Mean absolute error (points) - lower is better"
    else:
        cv_col, test_col, xlabel = CV_F1, "F1-score", "F1-score - higher is better"
    df = results.iloc[::-1].reset_index(drop=True)      # best model ends up at the top
    fig, ax = _fig(7.0, 0.55 * len(df) + 1.4)
    _style(ax, grid="x")
    y = np.arange(len(df))
    h = 0.34
    ax.barh(y + h / 2 + 0.01, df[cv_col], height=h, color=BLUE, label="Cross-validation (training data)")
    ax.barh(y - h / 2 - 0.01, df[test_col], height=h, color=ORANGE, label="Held-out test set")
    ax.set_yticks(y, df["Model"])
    ax.set_xlabel(xlabel)
    best = len(df) - 1 if not df["Model"].iloc[-1].startswith("Baseline") else len(df) - 2
    for val, off in ((df[cv_col].iloc[best], h / 2), (df[test_col].iloc[best], -h / 2)):
        ax.text(val, best + off + (0.01 if off > 0 else -0.01), f" {val:.2f}", va="center",
                fontsize=8.5, color=INK2)
    # long bars are at the bottom for error metrics (baseline) and at the top for F1
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2,
              loc="upper right" if task == REGRESSION else "lower right")
    _title(ax, "Model comparison")
    return fig


def actual_vs_predicted(y_true, y_pred, model_name: str) -> Figure:
    fig, (a, b) = _fig(9.0, 4.0, ncols=2)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    a.scatter(y_true, y_pred, s=24, color=BLUE, alpha=.65, linewidths=0)
    lo, hi = min(y_true.min(), y_pred.min()) - 2, max(y_true.max(), y_pred.max()) + 2
    a.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1)
    a.text(hi - 1, lo + 1, "perfect prediction", ha="right", va="bottom", fontsize=8.5, color=MUTED)
    a.set_xlabel("Actual final score")
    a.set_ylabel("Predicted final score")
    _title(a, f"Actual vs predicted - {model_name}")
    resid = y_true - y_pred
    b.hist(resid, bins=18, color=BLUE, edgecolor=SURFACE, linewidth=2)
    b.axvline(0, color=MUTED, linewidth=1)
    b.set_xlabel("Residual (actual - predicted, points)")
    b.set_ylabel("Test records")
    b.yaxis.set_major_locator(MaxNLocator(integer=True))
    _title(b, f"Residuals (mean {resid.mean():+.1f}, std {resid.std():.1f})")
    return fig


def confusion_matrix_chart(cm, model_name: str) -> Figure:
    cm = np.asarray(cm)
    fig, ax = _fig(4.6, 4.0)
    _style(ax, grid=None)
    ax.imshow(cm, cmap=SEQ, aspect="equal")
    names = ["Not at risk", "At risk"]
    ax.set_xticks([0, 1], names)
    ax.set_yticks([0, 1], names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=15,
                    fontweight="bold", color="white" if cm[i, j] > cm.max() / 2 else INK)
    _title(ax, f"Confusion matrix - {model_name}")
    return fig


def complexity_chart(curve: pd.DataFrame, task: str, best: int) -> Figure:
    fig, ax = _fig(7.0, 4.0)
    train, cv = curve["Training score"], curve["Cross-validation score"]
    for series, color, name in ((train, BLUE, "Training score"), (cv, ORANGE, "Cross-validation score")):
        ax.plot(curve.max_depth, series, color=color, linewidth=2, label=name,
                marker="o", markersize=6, markerfacecolor=color, markeredgecolor=SURFACE,
                markeredgewidth=2)
    ax.axvline(best, color=MUTED, linewidth=1)
    lo, hi = ax.get_ylim()
    ax.text(best + .15, lo + .02 * (hi - lo), f"best CV depth = {best}", fontsize=8.5, color=INK2,
            va="bottom")
    ax.set_xlabel("Decision-tree max_depth (model complexity)")
    ax.set_ylabel("MAE (points) - lower is better" if task == REGRESSION else "F1-score - higher is better")
    ax.set_xticks(curve.max_depth)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="center right")
    _title(ax, "Overfitting: training vs cross-validation score")
    return fig


def importance_chart(imp: pd.DataFrame, task: str) -> Figure:
    """Two panels (different units, so never one shared axis)."""
    fig, (a, b) = _fig(9.5, 3.8, ncols=2)
    perm_label = ("Increase in MAE when shuffled (points)" if task == REGRESSION
                  else "Drop in ROC-AUC when shuffled")
    for ax, col, title, xlabel in (
            (a, "Impurity importance", "Random Forest impurity importance", "Share of total split gain"),
            (b, "Permutation importance", "Permutation importance (test set)", perm_label)):
        _style(ax, grid="x")
        d = imp.sort_values(col)
        names = [_label(n) for n in d.index]
        xerr = d["Permutation std"] if col.startswith("Permutation") else None
        ax.barh(names, d[col], height=.55, color=BLUE, xerr=xerr,
                error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 0})
        ax.set_xlabel(xlabel)
        _title(ax, title)
    return fig


def tree_chart(tree, task: str) -> Figure:
    fig = Figure(figsize=(13, 5.5), facecolor=SURFACE, layout="constrained")
    FigureCanvasAgg(fig)   # plot_tree measures text, which needs a canvas with a renderer
    ax = fig.subplots()
    plot_tree(tree, feature_names=FEATURES, filled=True, rounded=True, fontsize=8, precision=1,
              class_names=None if task == REGRESSION else ["Not at risk", "At risk"], ax=ax)
    return fig
