# EduPredict AI — Machine Learning-II CA Project

A Streamlit application for student score prediction and academic-risk classification. It demonstrates concepts from a Machine Learning-II syllabus with a complete, honest evaluation workflow: baselines, cross-validation, a held-out test set, overfitting analysis, tuning and feature importance. The written report is in [`CA_REPORT.md`](CA_REPORT.md).

## What is in this folder

| File | Purpose |
|---|---|
| `app.py` | The Streamlit dashboard (the interface) |
| `ml_core.py` | All model-building and evaluation code, with no Streamlit in it |
| `plots.py` | Chart functions used by both the app and the report |
| `run_experiments.py` | Regenerates every number, table and figure in the report |
| `test_project.py` | 9 quick checks on the data and the ML code |
| `student_performance.csv` | Synthetic dataset (600 records) |
| `CA_REPORT.md` | Project report with results and discussion |
| `figures/`, `results/` | Charts and tables produced by `run_experiments.py` |
| `requirements.txt` | Python dependencies |

## Features

- Exploratory analysis: scatter plot, correlation matrix, descriptive statistics
- Regression comparison: Baseline, Linear Regression, Ridge (L2), Lasso (L1), Polynomial Regression, Decision Tree, Random Forest and SVR
- Classification comparison: Baseline, Logistic Regression, linear/RBF SVM, Decision Tree and Random Forest
- Train/test split and k-fold cross-validation (3, 5 or 10 folds); the best model is chosen on cross-validation only
- Regression metrics (MAE, RMSE, R²) with an actual-vs-predicted and residual plot
- Classification metrics (accuracy, precision, recall, F1) with a confusion matrix, an adjustable at-risk cut-off and an optional class-balancing switch
- Overfitting demonstration: training vs cross-validation score as decision-tree depth grows, plus a readable depth-3 tree
- GridSearchCV hyperparameter tuning for Random Forest (regression) or SVM (classification), with test-set results for the tuned model
- Feature importance: Random Forest impurity importance and held-out permutation importance
- Interactive prediction form (with an estimated risk probability where the model provides one)

## Important dataset notes

`student_performance.csv` is a **synthetic demonstration dataset** generated for this project. It is not real student data and model scores are not evidence of real-world predictive performance. If your instructor requires a real dataset, replace it with an approved dataset and adjust the column names in `ml_core.py` (`FEATURES`, `TARGET`).

**About the at-risk label.** The CSV's `academic_risk` column marks scores below 50, but only 12 of the 600 records qualify, which leaves about 2 at-risk students in a 20 % test set — too few to evaluate a classifier. The app therefore derives the label from `final_score` using a sidebar cut-off that **defaults to 60** (55 at-risk records) and can be set between 50 and 70. The CSV column itself is left unchanged and unused. Set `DEFAULT_RISK_THRESHOLD` in `ml_core.py` if you want a different default. The report explains this choice and shows results at every cut-off.

## Run locally

Use Python 3.10 or newer.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
streamlit run app.py
```

The browser opens the local dashboard. Keep `app.py`, `ml_core.py`, `plots.py` and `student_performance.csv` in the same folder.

## Reproduce the report

```bash
python run_experiments.py   # about 30 seconds; rewrites figures/ and results/
python test_project.py      # runs the 9 checks
```

`run_experiments.py` uses the same code and default settings as the app (20 % test data, 5-fold cross-validation, cut-off 60, seed 42), so the report and the dashboard agree. Different library versions can change the last decimal place, especially for random forests.

## Suggested 30-mark CA presentation

1. Problem statement and objectives — 3 marks
2. Dataset and preprocessing — 4 marks
3. Algorithms and methodology — 7 marks
4. Implementation and live demo — 7 marks
5. Results, metrics, and comparison — 5 marks
6. Conclusion and viva — 4 marks

This is a suggested presentation structure, not an official university marking rubric.

## Syllabus mapping

| Syllabus topic | Where demonstrated |
|---|---|
| Model evaluation and cross-validation | Section 3 comparison table (cross-validation vs test) and fold selector; 10-split stability table in the report |
| Overfitting / model complexity | Section 5 depth sweep (training vs cross-validation score) |
| Ridge and Lasso regularization | Regression model comparison; report Section 6.1 explains why they did not help here |
| Polynomial regression / non-linearity | Degree-2 polynomial pipeline |
| SVM and kernel methods | Linear SVM, RBF SVM, SVR; GridSearchCV over kernel, C and gamma |
| Decision trees and impurity-driven splits | Tree models, depth sweep and the depth-3 tree view (Gini / squared-error impurity per node) |
| Ensembles and Random Forest | Random Forest models; feature importance (Section 7) |
| Hyperparameter tuning | Section 6 GridSearchCV buttons |
| Evaluation with imbalanced classes | Baseline row, F1 as the selection metric, class-balancing switch, confusion matrix, cut-off slider |

## Likely viva questions

- **Why is accuracy not enough for the risk task?** Only about 9 % of students are at risk, so a model that flags nobody is already 91 % accurate. Precision, recall and F1 describe how well the at-risk students are actually found.
- **Why choose the model by cross-validation rather than the test set?** Choosing on the test set would let the test data influence the model, making the final score optimistic. The test set is used once, to report.
- **Why did linear regression beat the random forest?** The features are nearly uncorrelated and the relationship with the score looks close to linear, so extra flexibility only adds variance. This is a finding about this synthetic data, not a general rule.
- **Why did Ridge, Lasso and polynomial terms not improve things?** There are only six features and 480 training rows, nothing is strongly correlated, and Lasso removed no feature, so there is little for regularisation to fix.
- **Where do you see overfitting?** In Section 5: training error falls to almost zero as tree depth grows while cross-validation error stops improving and then rises.
- **What did GridSearchCV show?** The default Random Forest settings were already the best of the grid, so tuning did not help. Tuning refines a model within its family; it cannot turn a poorer family into a better one.
- **Impurity vs permutation importance?** Impurity importance is computed on training data and favours features with many split points; permutation importance measures the loss in held-out performance when a feature is shuffled and is more trustworthy.
- **Why was the cut-off changed from 50 to 60?** Only 12 records fall below 50, leaving about 2 positives in the test set. At 60 there are 55 (11 in the test set), enough for the metrics to carry information. The report shows what happens at every cut-off.
- **What are the main limitations?** Synthetic data, a small number of at-risk records, a label derived from the target, and no fairness analysis. The app must not be used on real students.

## Troubleshooting

- `ModuleNotFoundError: ml_core` or `plots` — run `streamlit run app.py` from this folder, or keep all the `.py` files together.
- `Could not load the dataset` — keep `student_performance.csv` in the same folder as `app.py`.
- A very low number of at-risk records triggers a warning in the app — raise the cut-off, use fewer folds or less test data.
