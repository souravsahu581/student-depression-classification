"""Interactive Streamlit evaluator for five saved student-depression classifiers.

The uploaded CSV MUST include the Depression target. No training happens in this app.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIRECTORY = ROOT / "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Student Depression | Classifier Lab", page_icon="📊", layout="wide")
st.title("Student Depression Classification")
st.caption("BITS Pilani M.Tech AIML · Machine Learning Assignment 2")
st.info("Educational demonstration only. Predictions are not medical diagnoses or clinical assessments.")


@st.cache_resource
def load_model(filename: str):
    """Load a pre-trained pipeline (including its fitted preprocessing)."""
    return joblib.load(MODEL_DIRECTORY / filename)


@st.cache_resource
def load_feature_info():
    return joblib.load(MODEL_DIRECTORY / "feature_info.joblib")


try:
    feature_info = load_feature_info()
except (FileNotFoundError, OSError, ValueError, ImportError) as exc:
    st.error(f"Cannot load model/feature_info.joblib: {exc}")
    st.stop()

expected_columns = feature_info["feature_columns"]
target_column = feature_info["target_column"]
positive_label = feature_info["positive_label"]
labels = feature_info["class_labels"]

with st.sidebar:
    st.header("Evaluation controls")
    model_name = st.selectbox("Choose classifier", list(MODEL_FILES))
    uploaded_file = st.file_uploader("Upload labeled test CSV", type=["csv"])
    st.caption("Include the original input columns plus the Depression target column.")
    st.caption("The uploaded data is used for evaluation only, never for fitting.")

if uploaded_file is None:
    st.markdown("### Getting started")
    st.write("Upload your `test_data.csv` using the sidebar to evaluate a saved classifier.")
    st.write("The repository includes five saved pipelines, each with its own trained classifier and preprocessing.")
    st.stop()

try:
    data = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Unable to read uploaded CSV: {exc}")
    st.stop()

required_columns = expected_columns + [target_column]
missing_columns = [name for name in required_columns if name not in data.columns]
if missing_columns:
    st.error("Missing required CSV columns: " + ", ".join(missing_columns))
    st.stop()
if data.empty:
    st.error("The uploaded CSV is empty.")
    st.stop()
if data[required_columns].isna().any().any():
    st.error("The uploaded CSV has missing values in required columns. Please fix these first.")
    st.stop()
if not set(data[target_column].astype(str).unique()).issubset(set(labels)):
    st.error(f"Unexpected target values. Expected only: {labels}")
    st.stop()

X_test = data[expected_columns].copy()
y_test = data[target_column].astype(str)

# The original fitted pipeline expects numerical features as numbers.
for col in feature_info["numeric_features"]:
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
if X_test[feature_info["numeric_features"]].isna().any().any():
    st.error("One or more numerical columns contain non-numeric values.")
    st.stop()

try:
    pipeline = load_model(MODEL_FILES[model_name])
    y_pred = pipeline.predict(X_test)
    class_names = list(pipeline.classes_)
    yes_index = class_names.index(positive_label)
    probabilities = pipeline.predict_proba(X_test)[:, yes_index]
except Exception as exc:
    st.error(f"Could not load or run the selected model: {exc}")
    st.stop()

is_positive = (y_test == positive_label).astype(int)
auc = roc_auc_score(is_positive, probabilities) if is_positive.nunique() == 2 else float("nan")
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC": auc,
    "Precision": precision_score(y_test, y_pred, pos_label=positive_label, zero_division=0),
    "Recall": recall_score(y_test, y_pred, pos_label=positive_label, zero_division=0),
    "F1 Score": f1_score(y_test, y_pred, pos_label=positive_label, zero_division=0),
    "MCC": matthews_corrcoef(y_test, y_pred),
}

st.subheader(f"Results · {model_name}")
st.caption(f"Evaluating {len(data):,} labeled rows | {len(expected_columns)} features")
columns = st.columns(3)
for index, (metric_name, value) in enumerate(metrics.items()):
    columns[index % 3].metric(metric_name, "N/A" if pd.isna(value) else f"{value:.4f}")
if pd.isna(auc):
    st.warning("AUC needs both actual target classes in the uploaded test CSV.")

left, right = st.columns(2)
with left:
    st.markdown("#### Confusion matrix")
    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5.2, 4.1))
    ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title("Actual vs predicted")
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

with right:
    st.markdown("#### Classification report")
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

with st.expander("Preview uploaded test data"):
    st.dataframe(data.head(20), use_container_width=True)

with st.expander("View predictions and download results"):
    predicted = data.copy()
    predicted["Predicted_Depression"] = y_pred
    predicted["Probability_Yes"] = probabilities
    st.dataframe(predicted.head(30), use_container_width=True)
    st.download_button(
        "Download predictions CSV",
        data=predicted.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )

st.caption("All evaluations use the saved joblib pipeline; no training or fitting is performed here.")
