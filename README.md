# Credit Card Customer Churn Prediction

ML Assignment 2: five classification models trained on the same dataset, compared on six metrics, and shown in a Streamlit app.

- **GitHub repository:** <https://github.com/csrathour83/credit-card-churn-prediction-ML-A2>
- **Live Streamlit app:** <https://credit-card-churn-prediction-ml-a2-n7hdihztgnre5pwrh9zsmg.streamlit.app/>

---

## a. Problem statement

A bank is losing credit card customers. The goal is to predict which customers are likely to leave (churn), so the bank can contact them before they go.

This is a **binary classification** problem:
- `1` = Attrited Customer (left the bank)
- `0` = Existing Customer

---

## b. Dataset description

- **Source:** Kaggle, "Credit Card customers" (`BankChurners.csv`)
- **Instances:** 10,127 customers
- **Target:** `Attrition_Flag`
- **Class balance:** 1,627 attrited (16.1%) and 8,500 existing (83.9%). The data is imbalanced.

**Features used (18):**

| Type | Features |
|---|---|
| Customer info | `Customer_Age`, `Gender`, `Dependent_count`, `Education_Level`, `Marital_Status`, `Income_Category` |
| Account info | `Card_Category`, `Months_on_book`, `Total_Relationship_Count`, `Months_Inactive_12_mon`, `Contacts_Count_12_mon` |
| Credit and spending | `Credit_Limit`, `Total_Revolving_Bal`, `Total_Amt_Chng_Q4_Q1`, `Total_Trans_Amt`, `Total_Trans_Ct`, `Total_Ct_Chng_Q4_Q1`, `Avg_Utilization_Ratio` |

**Columns removed:**
- `CLIENTNUM`: customer ID, not useful for prediction.
- Two `Naive_Bayes_Classifier_...` columns: these are predictions from another model and contain the answer (data leakage).
- `Avg_Open_To_Buy`: equal to `Credit_Limit - Total_Revolving_Bal`, so it adds no new information.

**Preprocessing** (same for all models, inside a scikit-learn pipeline):
- Categorical features: one-hot encoding.
- Numeric features: `PowerTransformer` (Yeo-Johnson), which reduces skew and scales the values.

**Train/test split:** 80/20, stratified, `random_state=42`. The test set (2,026 rows) is saved as `test_data.csv` and used in the Streamlit app.

---

## c. GitHub repository link

<https://github.com/csrathour83/credit-card-churn-prediction-ML-A2>

The repository contains `app.py`, `requirements.txt`, `README.md`, `test_data.csv`, the saved models for all five classifiers in `model/`, and the training notebooks in `notebooks/`.

---

## d. Models used

All five models use the same split and the same preprocessing.

| Model | Settings | How the settings were chosen |
|---|---|---|
| Logistic Regression | `max_iter=1000` | Default. `class_weight="balanced"` was tested but gave lower F1 and MCC. |
| Decision Tree | `max_depth=6`, `min_samples_leaf=1` | 5-fold CV (F1). The default tree overfit (depth 19, train score 1.0). |
| KNN | `n_neighbors=7` | 5-fold CV (F1). k=3 and k=7 scored almost the same, and k=7 varied less between folds. |
| Naive Bayes | `GaussianNB()`, default | Multinomial cannot be used because scaled features have negative values. |
| SVM | RBF kernel, `C=10`, `gamma="scale"` | 5-fold CV (F1). |

Hyperparameters were chosen with cross-validation on the training data only. The test data was used only for the final scores below.

### Comparison of evaluation metrics (test set)

| ML Model Name          | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression    | 0.9161   | 0.9316 | 0.8189    | 0.6123 | 0.7007 | 0.6624 |
| Decision Tree          | 0.9368   | 0.9476 | 0.8456    | 0.7415 | 0.7902 | 0.7554 |
| KNN                    | 0.9156   | 0.9131 | 0.8632    | 0.5631 | 0.6816 | 0.6547 |
| Naive Bayes            | 0.8697   | 0.8532 | 0.6000    | 0.5631 | 0.5810 | 0.5043 |
| Support Vector Machine | **0.9442** | **0.9657** | 0.8581 | **0.7815** | **0.8180** | **0.7864** |

Precision, Recall and F1 are for the churn class (Attrited). Because only 16% of customers churn, F1 and MCC are the main metrics for comparing models, not accuracy.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Train and test scores are almost the same, so there is no overfitting. AUC is good (0.932), but recall is 0.612, so it misses about 39% of churned customers. As a linear model, it cannot capture combinations of features (for example, low transaction count together with low amount). |
| Decision Tree | Second best model (F1 0.790, MCC 0.755). The default tree overfit, so depth was limited to 6 using CV. This improved test AUC from 0.88 to 0.95. Recall (0.742) is much better than Logistic Regression. |
| KNN | Accuracy is high (0.916), but recall is only 0.563, so it misses about 44% of churned customers. Precision is the highest of all models (0.863): when it predicts churn, it is usually correct. Its F1 and MCC are lower than Logistic Regression. |
| Naive Bayes | Weakest model (accuracy 0.870, F1 0.581, MCC 0.504). Recall (0.563) is the same as KNN, but precision is only 0.600, so 4 out of 10 churn predictions are wrong. It assumes features are independent, but many features here are related (e.g. transaction amount and count), which is likely why it does worse. |
| Support Vector Machine | Best scores of all models (accuracy 0.944, AUC 0.966, F1 0.818, MCC 0.786). Highest recall (0.782): it finds about 78% of churned customers. Precision (0.858) is close to the best (KNN, 0.863). |
| **Overall, Winner for your dataset?** | **SVM.** It has the highest accuracy, AUC, recall, F1 and MCC. Decision Tree is second. Naive Bayes is the weakest. |

---

## Repository structure

```
credit-card-churn-prediction-ML-A2/
|-- app.py                 Streamlit app
|-- requirements.txt
|-- README.md
|-- BankChurners.csv       full dataset
|-- test_data.csv          test split (20%) used in the app
|-- model/                 saved pipelines for all five models (.joblib)
|-- results/               metrics for each model (.json) and metrics.csv
|-- src/common.py          data loading, split, preprocessing, metrics
|-- notebooks/
    |-- 00_prepare_data.ipynb        data checks and preprocessing decisions
    |-- 01_logistic_regression.ipynb
    |-- 02_decision_tree.ipynb
    |-- 03_knn.ipynb
    |-- 04_naive_bayes.ipynb
    |-- 05_svm.ipynb
```

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload `test_data.csv` in the app, pick a model, and see its metrics, confusion matrix and classification report. The app uses the saved models from `model/` and does not retrain them.
