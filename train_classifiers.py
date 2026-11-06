"""
BITS M.Tech AIML - Machine Learning Assignment 2
Five classifiers for Student_Depression_Dataset.csv

Run:
    pip install -r requirements.txt
    python train_classifiers.py

The script:
1. Loads the supplied dataset.
2. Uses Depression as the target.
3. Splits the data into training (80%) and test (20%) sets.
4. Preprocesses numeric and categorical columns inside each model pipeline.
5. Trains five classifiers and evaluates them on the same held-out test set.
6. Saves the fitted pipelines, test_data.csv, and evaluation_metrics.csv.

IMPORTANT:
- Keep Student_Depression_Dataset.csv in the same folder as this script.
- The saved .joblib files contain preprocessing AND the classifier.
- Do not fit/retrain models on test_data.csv when evaluating in the Streamlit app.
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# 1. PATHS AND SETTINGS
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "Student_Depression_Dataset.csv"
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "outputs"

TARGET_COLUMN = "Depression"
TEST_SIZE = 0.20
RANDOM_STATE = 42
POSITIVE_LABEL = "Yes"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# 2. LOAD AND VALIDATE DATASET
# ---------------------------------------------------------------------

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}\n"
        "Place Student_Depression_Dataset.csv in the same folder as this script."
    )

df = pd.read_csv(DATASET_PATH)

print("\n========== DATASET INFORMATION ==========")
print(f"Dataset shape: {df.shape}")
print("\nColumns:")
print(df.columns.tolist())

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' was not found. "
        f"Available columns: {df.columns.tolist()}"
    )

if df[TARGET_COLUMN].isna().any():
    raise ValueError("The target column contains missing values. Please inspect the dataset.")

# Remove rows with missing values, if any.
# The supplied dataset currently has no missing values.
df = df.dropna().copy()

# Remove accidental duplicate rows, if any.
df = df.drop_duplicates().copy()

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN].astype(str)

if y.nunique() != 2:
    raise ValueError(
        f"This script is configured for binary classification, but found "
        f"{y.nunique()} target classes: {sorted(y.unique())}"
    )

if POSITIVE_LABEL not in set(y.unique()):
    raise ValueError(
        f"Expected positive class '{POSITIVE_LABEL}' but found {sorted(y.unique())}. "
        "Update POSITIVE_LABEL in the script if required."
    )

print(f"\nNumber of instances after cleaning: {len(df)}")
print(f"Number of input features: {X.shape[1]}")
print("\nTarget distribution:")
print(y.value_counts())


# ---------------------------------------------------------------------
# 3. IDENTIFY NUMERIC AND CATEGORICAL FEATURES
# ---------------------------------------------------------------------

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)


# ---------------------------------------------------------------------
# 4. TRAIN-TEST SPLIT
# ---------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\n========== TRAIN-TEST SPLIT ==========")
print(f"Training instances: {len(X_train)}")
print(f"Testing instances:  {len(X_test)}")


# ---------------------------------------------------------------------
# 5. PREPROCESSING
# ---------------------------------------------------------------------

def make_preprocessor():
    """
    Numeric columns:
        StandardScaler

    Categorical columns:
        OneHotEncoder

    Dense output is used because GaussianNB requires dense input.
    This preprocessing is fitted only on the training data because it
    is part of each sklearn Pipeline.
    """
    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


# ---------------------------------------------------------------------
# 6. DEFINE THE FIVE CLASSIFIERS
# ---------------------------------------------------------------------

classifiers = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE,
    ),

    "K-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=5,
    ),

    "Gaussian Naive Bayes": GaussianNB(),

    "Support Vector Machine": SVC(
        kernel="rbf",
        probability=True,
        random_state=RANDOM_STATE,
    ),
}


# ---------------------------------------------------------------------
# 7. TRAIN, PREDICT, AND EVALUATE EACH MODEL
# ---------------------------------------------------------------------

results = []
trained_models = {}

for model_name, classifier in classifiers.items():

    print("\n" + "=" * 72)
    print(f"MODEL: {model_name}")
    print("=" * 72)

    # A separate preprocessing pipeline is fitted for each classifier.
    pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("classifier", classifier),
        ]
    )

    # Train using training data only.
    pipeline.fit(X_train, y_train)

    # Predict class labels for the unseen test set.
    y_pred = pipeline.predict(X_test)

    # Get probability estimates for the positive class for ROC-AUC.
    positive_class_index = list(pipeline.classes_).index(POSITIVE_LABEL)
    y_probability = pipeline.predict_proba(X_test)[:, positive_class_index]

    # Evaluation metrics.
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(
        (y_test == POSITIVE_LABEL).astype(int),
        y_probability,
    )
    precision = precision_score(
        y_test,
        y_pred,
        pos_label=POSITIVE_LABEL,
        zero_division=0,
    )
    recall = recall_score(
        y_test,
        y_pred,
        pos_label=POSITIVE_LABEL,
        zero_division=0,
    )
    f1 = f1_score(
        y_test,
        y_pred,
        pos_label=POSITIVE_LABEL,
        zero_division=0,
    )
    mcc = matthews_corrcoef(y_test, y_pred)

    # Save the metrics for the comparison table.
    results.append(
        {
            "ML Model Name": model_name,
            "Accuracy": accuracy,
            "AUC": auc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "MCC": mcc,
        }
    )

    # Print metrics.
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"AUC       : {auc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"MCC       : {mcc:.4f}")

    # Confusion matrix. Label order is fixed for consistent interpretation.
    labels = ["No", "Yes"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    print("\nConfusion Matrix (rows = actual, columns = predicted)")
    print(pd.DataFrame(
        cm,
        index=[f"Actual {label}" for label in labels],
        columns=[f"Predicted {label}" for label in labels],
    ))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save the complete pipeline (preprocessing + trained classifier).
    safe_name = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )
    model_path = MODEL_DIR / f"{safe_name}.joblib"
    joblib.dump(pipeline, model_path)

    trained_models[model_name] = pipeline
    print(f"Saved model: {model_path}")


# ---------------------------------------------------------------------
# 8. SAVE MODEL COMPARISON TABLE
# ---------------------------------------------------------------------

metrics_df = pd.DataFrame(results)

metrics_path = OUTPUT_DIR / "evaluation_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)

print("\n\n========== MODEL COMPARISON ==========")
print(metrics_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
print(f"\nMetrics saved to: {metrics_path}")


# ---------------------------------------------------------------------
# 9. SAVE HELD-OUT TEST DATA FOR STREAMLIT
# ---------------------------------------------------------------------

# Keep the original, untransformed columns and include the actual target.
# The Streamlit app can load this CSV and use the saved pipelines to predict.
test_data = X_test.copy()
test_data[TARGET_COLUMN] = y_test.values

test_data_path = BASE_DIR / "test_data.csv"
test_data.to_csv(test_data_path, index=False)

print(f"Test data saved to: {test_data_path}")


# ---------------------------------------------------------------------
# 10. SAVE FEATURE INFORMATION
# ---------------------------------------------------------------------

feature_info = {
    "target_column": TARGET_COLUMN,
    "positive_label": POSITIVE_LABEL,
    "feature_columns": X.columns.tolist(),
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "class_labels": sorted(y.unique().tolist()),
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
}

joblib.dump(feature_info, MODEL_DIR / "feature_info.joblib")

print("\n========== COMPLETED ==========")
print("All five classifiers have been trained and evaluated.")
print(f"Trained pipelines: {MODEL_DIR}")
print(f"Test CSV:          {test_data_path}")
print(f"Metrics CSV:       {metrics_path}")
