# Generated tables

### Descriptive statistics

| Variable | mean | std | min | median | max |
|---|---|---|---|---|---|
| study_hours | 4.22 | 2.15 | 0.50 | 4.25 | 8.00 |
| attendance_pct | 72.73 | 16.23 | 45.00 | 72.15 | 100.00 |
| previous_score | 64.48 | 19.73 | 30.10 | 63.60 | 97.90 |
| assignments_completed | 4.89 | 3.15 | 0.00 | 5.00 | 10.00 |
| sleep_hours | 6.52 | 1.44 | 4.00 | 6.50 | 9.00 |
| internet_access | 0.51 | 0.50 | 0.00 | 1.00 | 1.00 |
| final_score | 77.63 | 12.68 | 44.20 | 78.20 | 100.00 |

### Correlation with final score

| Feature | Pearson r |
|---|---|
| previous_score | 0.557 |
| study_hours | 0.542 |
| assignments_completed | 0.346 |
| attendance_pct | 0.239 |
| sleep_hours | 0.216 |
| internet_access | 0.061 |

### Linear-regression coefficients (fitted on all 600 rows, descriptive only)

| Feature | Points per unit | Points per 1 std dev |
|---|---|---|
| previous_score | 0.35 | 6.95 |
| study_hours | 3.11 | 6.69 |
| assignments_completed | 1.22 | 3.83 |
| attendance_pct | 0.19 | 3.01 |
| sleep_hours | 1.41 | 2.03 |
| internet_access | 1.89 | 0.95 |

Intercept: 12.21; in-sample R² = 0.792

### Regression: model comparison (20 % test, 5-fold CV)

| Model | CV MAE (lower is better) | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|
| Linear Regression | 4.695 | 4.724 | 6.122 | 0.791 |
| Ridge (L2) | 4.697 | 4.721 | 6.120 | 0.791 |
| Lasso (L1) | 4.712 | 4.705 | 6.115 | 0.791 |
| Polynomial Regression (degree 2) | 4.712 | 4.804 | 6.229 | 0.783 |
| SVM (RBF) | 5.453 | 4.888 | 6.378 | 0.773 |
| Random Forest Regressor | 5.674 | 5.261 | 6.821 | 0.740 |
| Decision Tree Regressor | 7.168 | 6.385 | 8.170 | 0.627 |
| Baseline (predict the mean) | 10.207 | 10.955 | 13.379 | 0.000 |

### Regression: stability over 10 random splits

| Model | Test MAE (mean ± std) | Test R² (mean ± std) |
|---|---|---|
| Linear Regression | 4.661 ± 0.258 | 0.793 ± 0.025 |
| Ridge (L2) | 4.662 ± 0.259 | 0.793 ± 0.025 |
| Lasso (L1) | 4.670 ± 0.264 | 0.793 ± 0.024 |
| Polynomial Regression (degree 2) | 4.737 ± 0.260 | 0.788 ± 0.026 |
| SVM (RBF) | 5.112 ± 0.382 | 0.750 ± 0.042 |
| Random Forest Regressor | 5.475 ± 0.381 | 0.727 ± 0.046 |
| Decision Tree Regressor | 7.038 ± 0.457 | 0.544 ± 0.057 |
| Baseline (predict the mean) | 10.357 ± 0.510 | -0.010 ± 0.012 |

### Classification: model comparison (cut-off 60, 20 % test, 5-fold CV)

| Model | CV F1 | Test Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|---|
| Logistic Regression | 0.738 | 0.942 | 0.750 | 0.545 | 0.632 |
| Linear SVM | 0.732 | 0.942 | 0.750 | 0.545 | 0.632 |
| RBF SVM | 0.698 | 0.958 | 0.875 | 0.636 | 0.737 |
| Decision Tree | 0.607 | 0.950 | 1.000 | 0.455 | 0.625 |
| Random Forest | 0.607 | 0.942 | 0.833 | 0.455 | 0.588 |
| Baseline (never at risk) | 0.000 | 0.908 | 0.000 | 0.000 | 0.000 |

### Classification with class_weight='balanced'

| Model | CV F1 | Test Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|---|
| Linear SVM | 0.664 | 0.933 | 0.588 | 0.909 | 0.714 |
| Random Forest | 0.661 | 0.967 | 0.818 | 0.818 | 0.818 |
| Logistic Regression | 0.661 | 0.950 | 0.692 | 0.818 | 0.750 |
| RBF SVM | 0.606 | 0.917 | 0.529 | 0.818 | 0.643 |
| Decision Tree | 0.597 | 0.917 | 0.529 | 0.818 | 0.643 |
| Baseline (never at risk) | 0.000 | 0.908 | 0.000 | 0.000 | 0.000 |

### Classification: stability over 10 random splits

| Model | F1-score (mean ± std) | Recall (mean ± std) | Precision (mean ± std) |
|---|---|---|---|
| Logistic Regression | 0.771 ± 0.084 | 0.764 ± 0.143 | 0.799 ± 0.098 |
| Linear SVM | 0.748 ± 0.085 | 0.736 ± 0.132 | 0.779 ± 0.102 |
| RBF SVM | 0.663 ± 0.205 | 0.627 ± 0.262 | 0.779 ± 0.142 |
| Random Forest | 0.657 ± 0.134 | 0.536 ± 0.168 | 0.915 ± 0.100 |
| Decision Tree | 0.605 ± 0.085 | 0.627 ± 0.090 | 0.605 ± 0.159 |
| Baseline (never at risk) | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |

### Sensitivity to the at-risk cut-off

| Cut-off | At-risk records | Train / test positives | Best model (CV F1) | CV F1 | Test precision | Test recall | Test F1 | Accuracy | Baseline accuracy |
|---|---|---|---|---|---|---|---|---|---|
| 50 | 12 | 10 / 2 | Decision Tree | 0.347 | 0.000 | 0.000 | 0.000 | 0.942 | 0.983 |
| 55 | 28 | 22 / 6 | Logistic Regression | 0.652 | 1.000 | 0.167 | 0.286 | 0.958 | 0.950 |
| 60 | 55 | 44 / 11 | Logistic Regression | 0.738 | 0.750 | 0.545 | 0.632 | 0.942 | 0.908 |
| 65 | 107 | 86 / 21 | Linear SVM | 0.672 | 0.875 | 0.667 | 0.757 | 0.925 | 0.825 |
| 70 | 159 | 127 / 32 | Linear SVM | 0.801 | 0.806 | 0.781 | 0.794 | 0.892 | 0.733 |

### Overfitting curve (regression)

| max_depth | Training score | Cross-validation score |
|---|---|---|
| 1 | 8.727 | 9.584 |
| 2 | 7.039 | 7.893 |
| 3 | 6.334 | 7.487 |
| 4 | 5.428 | 7.110 |
| 5 | 4.414 | 7.168 |
| 6 | 3.465 | 7.309 |
| 8 | 1.605 | 7.573 |
| 10 | 0.500 | 7.708 |
| 15 | 0.003 | 7.620 |

Best cross-validation depth: **4**

### Overfitting curve (classification)

| max_depth | Training score | Cross-validation score |
|---|---|---|
| 1 | 0.000 | 0.000 |
| 2 | 0.380 | 0.317 |
| 3 | 0.717 | 0.479 |
| 4 | 0.798 | 0.523 |
| 5 | 0.897 | 0.607 |
| 6 | 0.966 | 0.603 |
| 8 | 1.000 | 0.595 |
| 10 | 1.000 | 0.595 |
| 15 | 1.000 | 0.595 |

Best cross-validation depth: **7**

### GridSearchCV (regression)

Best parameters: `{'max_depth': 8, 'n_estimators': 150}`; best CV MAE = 5.674

| max_depth | n_estimators | CV MAE | rank |
|---|---|---|---|
| 8 | 150 | 5.674 | 1 |
| None | 150 | 5.680 | 2 |
| 6 | 150 | 5.686 | 3 |
| None | 100 | 5.703 | 4 |
| 8 | 100 | 5.710 | 5 |

Test metrics of the tuned model: Test MAE = 5.261, Test RMSE = 6.821, Test R² = 0.740

### GridSearchCV (classification)

Best parameters: `{'C': 1, 'kernel': 'linear'}`; best CV F1 = 0.732

| C | kernel | gamma | CV F1 | rank |
|---|---|---|---|---|
| 1.0 | linear | – | 0.732 | 1 |
| 10.0 | linear | – | 0.732 | 1 |
| 1.0 | rbf | auto | 0.698 | 3 |
| 1.0 | rbf | scale | 0.698 | 3 |
| 0.1 | linear | – | 0.690 | 5 |

Test metrics of the tuned model: Test Accuracy = 0.942, Precision = 0.750, Recall = 0.545, F1-score = 0.632

### Feature importance (regression)

| Feature | Impurity importance | Permutation importance | Permutation std |
|---|---|---|---|
| previous_score | 0.378 | 4.244 | 0.460 |
| study_hours | 0.347 | 3.907 | 0.533 |
| assignments_completed | 0.100 | 1.190 | 0.126 |
| attendance_pct | 0.097 | 0.974 | 0.182 |
| sleep_hours | 0.072 | 0.127 | 0.141 |
| internet_access | 0.005 | 0.030 | 0.018 |

### Feature importance (classification)

| Feature | Impurity importance | Permutation importance | Permutation std |
|---|---|---|---|
| study_hours | 0.281 | 0.191 | 0.054 |
| previous_score | 0.267 | 0.111 | 0.044 |
| attendance_pct | 0.175 | 0.039 | 0.020 |
| assignments_completed | 0.129 | 0.004 | 0.018 |
| internet_access | 0.017 | 0.001 | 0.004 |
| sleep_hours | 0.131 | -0.004 | 0.010 |
