import io
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from common import CLASS_NAMES, MODEL_DIR, POSITIVE_LABEL, TARGET, TEST_CSV, compute_metrics

MODELS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "svm",
}
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

st.set_page_config(page_title="Credit Card Churn Predictor", layout="wide")


@st.cache_resource
def load_model(file_name):
    return joblib.load(MODEL_DIR / f"{file_name}.joblib")


def prepare_data(df, pipe):
    """Split uploaded CSV into features and 0/1 target; return error text if invalid."""
    if TARGET not in df.columns:
        return None, None, f"Column `{TARGET}` (target) is missing in the uploaded file."

    y = df[TARGET]
    if y.dtype == object:
        y = (y == POSITIVE_LABEL).astype(int)
    if y.nunique() < 2:
        return None, None, "The file must have both churned and existing customers to compute AUC."

    needed = list(pipe.feature_names_in_)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return None, None, f"These feature columns are missing: {', '.join(missing)}"

    # extra columns (ID, leaked NB columns etc.) are ignored
    return df[needed], y, None


def evaluate(pipe, X, y):
    y_pred = pipe.predict(X)
    y_prob = pipe.predict_proba(X)[:, 1]
    metrics = {k: round(float(v), 4) for k, v in compute_metrics(y, y_pred, y_prob).items()}
    return metrics, y_pred


def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def plot_cm(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(3.6, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 13},
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, linewidths=2, linecolor="white", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig_to_png(fig)


def plot_f1_mcc(table):
    """Grouped horizontal bars, best model on top."""
    data = table.sort_values("F1")
    pos = np.arange(len(data))
    h = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for offset, metric, color in [(h / 2 + 0.02, "F1", "#2a78d6"), (-h / 2 - 0.02, "MCC", "#eb6834")]:
        bars = ax.barh(pos + offset, data[metric], height=h, color=color, label=metric)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8, color="#52514e")
    ax.set_yticks(pos, data.index)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score (0 to 1, higher is better)")
    ax.grid(axis="x", color="#e5e5e2", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", frameon=False, ncols=2)
    return fig_to_png(fig)


def show_model(name, metrics, y, y_pred, churned):
    st.subheader(f"Evaluation metrics: {name}")
    cols = st.columns(6)
    for col, metric in zip(cols, METRIC_NAMES):
        col.metric(metric, f"{metrics[metric]:.4f}")

    caught = int(((y == 1) & (y_pred == 1)).sum())
    false_alarms = int(((y == 0) & (y_pred == 1)).sum())
    st.markdown(f"**Churned customers caught:** {caught} / {churned} "
                f"({caught / churned:.1%}) &nbsp;|&nbsp; **False alarms:** {false_alarms}")

    left, right = st.columns([1, 1.4])
    with left:
        st.markdown("**Confusion matrix**")
        st.image(plot_cm(y, y_pred), width=360)
    with right:
        st.markdown("**Classification report**")
        report = classification_report(y, y_pred, target_names=CLASS_NAMES, output_dict=True)
        st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")


# ---------- Sidebar ----------
st.sidebar.header("1. Upload test data")
uploaded = st.sidebar.file_uploader("CSV with all features + Attrition_Flag", type="csv")
st.sidebar.download_button(
    "Download sample test_data.csv",
    data=TEST_CSV.read_bytes(),
    file_name="test_data.csv",
    mime="text/csv",
)

st.sidebar.header("2. Choose a model")
model_name = st.sidebar.selectbox("Model", list(MODELS))

st.sidebar.markdown("---")
st.sidebar.caption("Models are trained once in the notebooks and loaded from `model/`. "
                   "Nothing is retrained on the uploaded data.")

# ---------- Header ----------
st.title("Credit Card Customer Churn Predictor")
st.write("Predicts which bank customers are likely to close their credit card account. "
         "Upload the test data, pick a model, and see how well it finds churned customers.")

if uploaded is None:
    st.info("Upload a CSV from the sidebar to start. "
            "You can download the sample `test_data.csv` there (2,026 customers, 20% hold-out set).")
    st.stop()

df = pd.read_csv(uploaded)
X, y, error = prepare_data(df, load_model(MODELS[model_name]))
if error:
    st.error(error)
    st.stop()

# ---------- Data summary ----------
churned = int(y.sum())
c1, c2, c3 = st.columns(3)
c1.metric("Customers", f"{len(y):,}")
c2.metric("Churned (Attrited)", f"{churned:,}", f"{churned / len(y):.1%} of total", delta_color="off")
c3.metric("Existing", f"{len(y) - churned:,}")
with st.expander("Preview uploaded data"):
    st.dataframe(df.head(10), width="stretch")

results = {}
for name, file_name in MODELS.items():
    results[name] = evaluate(load_model(file_name), X, y)

# selected model's tab comes first
order = [model_name] + [m for m in MODELS if m != model_name]
tabs = st.tabs(order + ["Compare all models"])

for tab, name in zip(tabs, order):
    with tab:
        metrics, y_pred = results[name]
        show_model(name, metrics, y, y_pred, churned)

# ---------- Last tab: all models ----------
with tabs[-1]:
    st.subheader("All five models on the uploaded data")
    table = pd.DataFrame({name: m for name, (m, _) in results.items()}).T[METRIC_NAMES]
    st.dataframe(table.sort_values("F1", ascending=False), width="stretch")

    best = table["F1"].idxmax()
    st.success(f"Best model by F1: **{best}** (F1 {table.loc[best, 'F1']:.4f}, "
               f"MCC {table.loc[best, 'MCC']:.4f})")
    st.caption("Only ~16% of customers churn, so F1 and MCC are better for comparing models than accuracy.")

    st.markdown("**F1 and MCC by model**")
    st.image(plot_f1_mcc(table), width="stretch")
