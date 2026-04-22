import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# File paths for dataset and saved outputs
DATA_PATH = Path("data/company_bankruptcy_prediction.csv")
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
FEATURES_PATH = ARTIFACTS_DIR / "selected_features.json"

# Define target column and selected input features
TARGET_COLUMN = "Bankrupt?"
SELECTED_FEATURES = [
    "ROA(C) before interest and depreciation before interest",
    "Operating Gross Margin",
    "Current Ratio",
    "Debt ratio %",
    "Net worth/Assets",
]

# load_data()
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the Kaggle CSV there first."
        )
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


# validate_columns()
def validate_columns(df: pd.DataFrame) -> None:
    required = [TARGET_COLUMN] + SELECTED_FEATURES
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# build_pipeline()
def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )

# evaluate()
def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "log_loss": float(log_loss(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, zero_division=0, output_dict=True
        ),
    }

# main() steps
def main() -> None:
    # Step 1: Load the dataset
    print("Loading data...")
    df = load_data(DATA_PATH)
    # Step 2: Check that the target and selected features exist
    validate_columns(df)

    # Step 3: Choose the input features (X) and target (y)
    # X = selected financial features
    # y = bankruptcy label
    X = df[SELECTED_FEATURES].copy()
    y = df[TARGET_COLUMN].astype(int)

    # Step 4: Split the data into training and test sets
    # Use stratify=y so class balance is preserved
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # Step 5: Build the preprocessing + model pipeline
    print("Training model...")
    pipeline = build_pipeline()
    # Step 6: Train the model on the training data
    pipeline.fit(X_train, y_train)

    # Step 7: Evaluate the model on the test data
    print("Evaluating model...")
    metrics = evaluate(pipeline, X_test, y_test)

    # Step 8: Create the artifacts folder if it does not exist
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 9: Save the trained model and outputs
    # - model.joblib = trained model
    # - metrics.json = evaluation results
    # - selected_features.json = input features used by the model
    print("Saving model and artifacts...")
    joblib.dump(pipeline, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(SELECTED_FEATURES, f, indent=2)

    # Step 10: Print confirmation
    print("\nTraining complete.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Features saved to: {FEATURES_PATH}")


if __name__ == "__main__":
    main()
