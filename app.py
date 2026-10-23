"""EduPredict AI - Streamlit dashboard (Machine Learning-II CA project).

Run with:  streamlit run app.py
The ML code is in ml_core.py and the charts in plots.py; this file is the interface.
"""
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

import ml_core as core
import plots

st.set_page_config(page_title="EduPredict AI", page_icon="🎓", layout="wide")
st.title("🎓 EduPredict AI")
st.caption("Student performance prediction and academic risk analysis | Machine Learning-II CA Project")
st.info("Educational demonstration only. The bundled dataset is synthetic, not real student records. "
        "Predictions must not be used to label or penalize real students.")


# --------------------------------------------------------------------------- #
# Cached helpers - models are trained once per combination of sidebar settings,
# not on every click.
# --------------------------------------------------------------------------- #
@st.cache_data
def load_data():
    return core.load_data()


@st.cache_resource(show_spinner="Training and cross-validating models...")
def run_comparison(task, test_size, cv_folds, risk_threshold, balanced):
    df = core.load_data()
    X_train, X_test, y_train, y_test = core.split_data(df, task, test_size, risk_threshold)
    models = core.make_regressors() if task == core.REGRESSION else core.make_classifiers(balanced)
    results, fitted, best = core.compare_models(task, models, X_train, y_train, X_test, y_test, cv_folds)
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
            "results": results, "fitted": fitted, "best": best}


@st.cache_data(show_spinner="Sweeping tree depth...")
def complexity(task, test_size, cv_folds, risk_threshold, balanced):
    run = run_comparison(task, test_size, cv_folds, risk_threshold, balanced)
    return core.complexity_curve(task, run["X_train"], run["y_train"], cv_folds, balanced)


@st.cache_data(show_spinner="Computing feature importance...")
def importance(task, test_size, cv_folds, risk_threshold, balanced):
    run = run_comparison(task, test_size, cv_folds, risk_threshold, balanced)
    return core.feature_importance(task, run["X_train"], run["y_train"], run["X_test"], run["y_test"])


try:
    df = load_data()
except (FileNotFoundError, ValueError) as err:
    st.error(f"Could not load the dataset: {err}. Keep student_performance.csv in the same folder as app.py.")
    st.stop()

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Project controls")
    task_label = st.radio("Choose task", ["Score prediction (Regression)", "Academic risk (Classification)"])
    task = core.REGRESSION if task_label.startswith("Score") else core.CLASSIFICATION
    test_size = st.slider("Test data (%)", 10, 40, 20, 5)
    cv_folds = st.select_slider("Cross-validation folds", options=[3, 5, 10], value=5)
    risk_threshold = st.slider("At-risk cut-off (final score below)", 50, 70, core.DEFAULT_RISK_THRESHOLD, 5,
                               help="A student counts as 'at risk' when the final score is below this value. "
                                    "At 50 only 12 of 600 records qualify, which is too few for reliable "
                                    "classification metrics.")
    balanced = False
    if task == core.CLASSIFICATION:
        balanced = st.checkbox("Balance classes (class_weight='balanced')", value=False,
                               help="Gives the rare at-risk class more weight. Usually raises recall "
                                    "and lowers precision.")
    st.caption("Models are trained locally and cached, so changing a setting retrains once.")

params = dict(task=task, test_size=test_size / 100, cv_folds=cv_folds,
              risk_threshold=risk_threshold, balanced=balanced)
n_risk = int((df[core.TARGET] < risk_threshold).sum())

# --------------------------------------------------------------------------- #
# 1. Dataset overview
# --------------------------------------------------------------------------- #
st.subheader("1. Dataset overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Student records", f"{len(df):,}")
c2.metric("Input features", len(core.FEATURES))
c3.metric("Mean final score", f"{df[core.TARGET].mean():.1f}/100")
c4.metric(f"At-risk records (< {risk_threshold})", f"{n_risk} ({n_risk / len(df):.1%})")
with st.expander("Preview dataset"):
    st.dataframe(df.head(20))
with st.expander("Descriptive statistics"):
    st.dataframe(df[core.FEATURES + [core.TARGET]].describe().T.round(2))
with st.expander("Feature descriptions"):
    st.markdown("""
    - **Study hours:** average daily study time
    - **Attendance:** percentage of classes attended
    - **Previous score:** earlier assessment score
    - **Assignments completed:** count out of 10
    - **Sleep hours:** average daily sleep
    - **Internet access:** 1 = yes, 0 = no
    - **Final score:** generated target score from 0 to 100 (18 records sit exactly at the 100 ceiling)
    - **Academic risk:** derived label, 1 when the final score is below the cut-off chosen in the sidebar
      (default 60). The CSV's own `academic_risk` column uses a cut-off of 50 and is not used, because only
      12 records fall below 50.
    """)

# --------------------------------------------------------------------------- #
# 2. Explore relationships
# --------------------------------------------------------------------------- #
st.subheader("2. Explore relationships")
col_a, col_b = st.columns(2)
with col_a:
    st.pyplot(plots.study_hours_scatter(df))
with col_b:
    st.pyplot(plots.correlation_heatmap(df))

# --------------------------------------------------------------------------- #
# 3. Compare machine learning models
# --------------------------------------------------------------------------- #
st.subheader("3. Compare machine learning models")
run = run_comparison(**params)
results, fitted, best_name = run["results"], run["fitted"], run["best"]
X_train, X_test, y_train, y_test = run["X_train"], run["X_test"], run["y_train"], run["y_test"]

if task == core.CLASSIFICATION:
    pos_train, pos_test = int(y_train.sum()), int(y_test.sum())
    st.caption(f"Training set: {len(y_train)} records ({pos_train} at risk) · "
               f"Test set: {len(y_test)} records ({pos_test} at risk)")
    if pos_test < 8 or pos_train < 3 * cv_folds:
        st.warning("Very few at-risk records in the training or test set, so the classification metrics below "
                   "are unreliable. Raise the at-risk cut-off, lower the number of folds, or use less test data.")
else:
    st.caption(f"Training set: {len(y_train)} records · Test set: {len(y_test)} records")

st.dataframe(results.round(3))
if task == core.REGRESSION:
    st.success(f"Best model by cross-validation MAE: **{best_name}**")
    st.markdown("**Interpretation:** MAE/RMSE measure average prediction error in score points (lower is better); "
                "R² measures how much score variation the model explains (closer to 1 is generally better). "
                "A useful model must clearly beat the *Baseline* row, which always predicts the average score.")
else:
    st.success(f"Best model by cross-validation F1: **{best_name}**")
    st.markdown("**Interpretation:** Accuracy is the fraction correct, but it is misleading when few students are "
                "at risk - the *Baseline* row never flags anyone yet still scores high accuracy. Precision is how "
                "often a flagged student is truly at risk, recall is how many at-risk students are caught, and F1 "
                "balances the two. The best model is chosen on cross-validation F1, never on the test set.")
st.pyplot(plots.model_comparison(results, task))

best_pred = fitted[best_name].predict(X_test)
if task == core.REGRESSION:
    st.pyplot(plots.actual_vs_predicted(y_test, best_pred, best_name))
else:
    cm = confusion_matrix(y_test, best_pred, labels=[0, 1])
    left, _ = st.columns([1, 1])
    with left:
        st.pyplot(plots.confusion_matrix_chart(cm, best_name))

# --------------------------------------------------------------------------- #
# 4. Predict
# --------------------------------------------------------------------------- #
if task == core.REGRESSION:
    st.subheader("4. Predict a student's score")
    form_id, button = "score", "Predict final score"
else:
    st.subheader("4. Predict academic risk")
    form_id, button = "risk", "Analyze academic risk"

with st.form(f"{form_id}_form"):
    a, b, c = st.columns(3)
    vals = {
        "study_hours": a.slider("Study hours/day", 0.0, 12.0, 3.0, 0.5, key=f"{form_id}_study"),
        "attendance_pct": b.slider("Attendance (%)", 0.0, 100.0, 75.0, 1.0, key=f"{form_id}_attend"),
        "previous_score": c.slider("Previous score", 0.0, 100.0, 65.0, 1.0, key=f"{form_id}_prev"),
        "assignments_completed": a.slider("Assignments completed (out of 10)", 0, 10, 7, key=f"{form_id}_assign"),
        "sleep_hours": b.slider("Sleep hours/day", 0.0, 12.0, 7.0, 0.5, key=f"{form_id}_sleep"),
        "internet_access": c.selectbox("Internet access", [1, 0], format_func=lambda x: "Yes" if x else "No",
                                       key=f"{form_id}_net"),
    }
    submitted = st.form_submit_button(button)
if submitted:
    out = core.predict_one(fitted[best_name], task, vals)
    if task == core.REGRESSION:
        st.metric("Predicted final score", f"{out['score']:.1f}/100")
        st.caption(f"Model: {best_name}. This is an estimate from synthetic demonstration data, not a guarantee; "
                   f"typical error on held-out records is about "
                   f"{results.loc[results['Model'] == best_name, 'Test MAE'].iloc[0]:.1f} points.")
    else:
        if out["risk"]:
            st.warning("The model flags this input as potentially at risk in the demonstration dataset.")
        else:
            st.success("The model does not flag this input as at risk in the demonstration dataset.")
        if out["probability"] is not None:
            st.metric("Estimated probability of risk", f"{out['probability']:.0%}")
        st.caption("This is not a real-world student assessment or diagnosis.")

# --------------------------------------------------------------------------- #
# 5. Overfitting and model complexity
# --------------------------------------------------------------------------- #
st.subheader("5. Overfitting and model complexity")
curve = complexity(**params)
depth = core.best_depth(curve, task)
st.pyplot(plots.complexity_chart(curve, task, depth))
st.markdown(f"**Interpretation:** a very shallow tree is too simple (underfitting). As `max_depth` grows the "
            f"training score keeps improving because the tree memorises the training data, but the "
            f"cross-validation score stops improving around depth **{depth}** and then stays flat or worsens - "
            f"that widening gap is overfitting. Choosing the complexity that is best on *unseen* folds is "
            f"exactly what cross-validation is for.")
with st.expander("Inspect a decision tree (depth 3)"):
    st.caption("Each box shows the split rule, the impurity of the node (squared error for regression, Gini for "
               "classification), the number of samples and the predicted value. The tree picks the split that "
               "reduces impurity the most.")
    st.pyplot(plots.tree_chart(core.fit_small_tree(task, X_train, y_train, 3, balanced), task))

# --------------------------------------------------------------------------- #
# 6. Hyperparameter tuning
# --------------------------------------------------------------------------- #
st.subheader("6. Hyperparameter tuning (GridSearchCV)")
tune_key = tuple(params.items())
if st.button("Tune Random Forest regressor" if task == core.REGRESSION else "Tune SVM classifier"):
    with st.spinner("Searching hyperparameters..."):
        st.session_state["tuning"] = (tune_key, core.tune_model(task, X_train, y_train, cv_folds, balanced))
saved = st.session_state.get("tuning")
if saved and saved[0] == tune_key:
    search = saved[1]
    best_cv = -search.best_score_ if task == core.REGRESSION else search.best_score_
    st.write("Best parameters:", {k.replace("model__", ""): v for k, v in search.best_params_.items()})
    st.write(f"Best cross-validation {'MAE' if task == core.REGRESSION else 'F1'}: {best_cv:.3f}")
    test_scores = core.evaluate_predictions(task, y_test, search.best_estimator_.predict(X_test))
    st.write("Tuned model on the held-out test set:",
             {k: round(float(v), 3) for k, v in test_scores.items()})
    with st.expander("Top parameter combinations"):
        st.dataframe(core.tuning_table(search, task))

# --------------------------------------------------------------------------- #
# 7. Feature importance
# --------------------------------------------------------------------------- #
st.subheader("7. Feature importance")
imp = importance(**params)
st.pyplot(plots.importance_chart(imp, task))
top = imp["Permutation importance"].idxmax()
st.markdown(f"**Interpretation:** impurity importance measures how much each feature improves the Random "
            f"Forest's splits during training; permutation importance measures how much the held-out "
            f"{'error' if task == core.REGRESSION else 'ROC-AUC'} gets worse when one feature is shuffled. "
            f"The most influential feature for this task is **{top.replace('_', ' ')}**. "
            f"Importance shows what the model relies on, not what causes a good score.")

st.divider()
st.markdown("**Conclusion:** This project demonstrates model evaluation, cross-validation, L1/L2 regularization, "
            "polynomial regression, SVM kernels, decision trees, random forests, overfitting analysis, feature "
            "importance, and GridSearchCV. Results are illustrative because the bundled data is synthetic.")
