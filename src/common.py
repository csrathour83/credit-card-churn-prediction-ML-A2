import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "BankChurners.csv"
TEST_CSV = ROOT / "test_data.csv"
MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "results"

TARGET = "Attrition_Flag"
POSITIVE_LABEL = "Attrited Customer"
CLASS_NAMES = ["Existing", "Attrited"]
# r = 0.996 with Credit_Limit -> redundant
DROP_FEATURES = ["Avg_Open_To_Buy"]
SEED = 42


def load_data():
    df = pd.read_csv(RAW_CSV)
    # drop ID and the two leaked Naive_Bayes_* prediction columns
    leak_cols = [c for c in df.columns if c.startswith("Naive_Bayes_Classifier")]
    df = df.drop(columns=["CLIENTNUM", *leak_cols])
    df[TARGET] = (df[TARGET] == POSITIVE_LABEL).astype(int)
    return df


def get_train_test():
    df = load_data()
    X, y = df.drop(columns=[TARGET, *DROP_FEATURES]), df[TARGET]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def save_test_data():
    _, X_test, _, y_test = get_train_test()
    test_df = X_test.assign(**{TARGET: y_test})
    test_df.to_csv(TEST_CSV, index=False)
    return test_df


def build_preprocessor(X):
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        # Yeo-Johnson fixes skew and standardizes
        ("num", PowerTransformer(), num_cols),
    ])


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def train_and_evaluate(name, estimator):
    X_train, X_test, y_train, y_test = get_train_test()
    pipe = Pipeline([("prep", build_preprocessor(X_train)), ("clf", estimator)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics = {k: round(float(v), 4) for k, v in metrics.items()}

    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_DIR / f"{name}.joblib")
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(metrics, indent=2))
    _update_summary(name, metrics)

    return {"metrics": metrics, "pipeline": pipe, "y_test": y_test,
            "y_pred": y_pred, "y_prob": y_prob}


def plot_confusion_matrix(y_true, y_pred, title=""):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    names = np.array([["True Neg", "False Pos"], ["False Neg", "True Pos"]])
    labels = [[f"{names[i, j]}\n{cm[i, j]}\n{cm[i, j] / cm.sum():.1%}" for j in range(2)] for i in range(2)]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False, ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.show()


def plot_roc_curve(y_true, y_prob, title=""):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_score(y_true, y_prob):.3f}")
    ax.plot([0, 1], [0, 1], ls="--", c="grey", label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (Recall)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.show()


def _update_summary(name, metrics):
    # upsert this model's row in results/metrics.csv
    path = RESULTS_DIR / "metrics.csv"
    summary = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not summary.empty:
        summary = summary[summary["Model"] != name]
    row = pd.DataFrame([{"Model": name, **metrics}])
    pd.concat([summary, row], ignore_index=True).to_csv(path, index=False)
