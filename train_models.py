"""Train five classifiers on the UCI EEG Eye State dataset (ID 264)."""
from pathlib import Path
import warnings
import joblib
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "results"
TARGET = "eyeDetection"
FEATURES = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2",
            "P8", "T8", "FC6", "F4", "F8", "AF4"]
RANDOM_STATE = 42


def load_dataset():
    """Download the official UCI dataset and return one labelled DataFrame."""
    print("Downloading UCI EEG Eye State dataset (ID 264)...")
    dataset = fetch_ucirepo(id=264)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()
    if TARGET not in y.columns:
        y.columns = [TARGET]
    data = pd.concat([X, y[TARGET]], axis=1)
    data = data[FEATURES + [TARGET]].dropna().reset_index(drop=True)
    data[TARGET] = data[TARGET].astype(int)
    if set(data[TARGET].unique()) != {0, 1}:
        raise ValueError("Expected binary target labels 0 and 1.")
    return data


def build_models():
    """Build all five classifiers required by the assignment."""
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DecisionTreeClassifier(max_depth=12, min_samples_leaf=4,
                                                    random_state=RANDOM_STATE))
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=7, weights="distance"))
        ]),
        "Gaussian Naive Bayes": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", GaussianNB())
        ]),
        "Support Vector Machine": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale",
                                probability=True, random_state=RANDOM_STATE))
        ])
    }


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    data = load_dataset()
    X, y = data[FEATURES], data[TARGET]

    # The UCI observations are chronological. Keep the last 20% as a later-in-time holdout.
    split_at = int(len(data) * 0.80)
    X_train, X_test = X.iloc[:split_at], X.iloc[split_at:]
    y_train, y_test = y.iloc[:split_at], y.iloc[split_at:]

    print(f"Total rows: {len(data):,}")
    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows: {len(X_test):,}")

    # This labelled file is the test data consumed by app.py.
    test_data = X_test.copy()
    test_data[TARGET] = y_test.values
    test_data.to_csv(ROOT / "test_data.csv", index=False)

    rows = []
    for name, model in build_models().items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        row = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, predicted),
            "AUC": roc_auc_score(y_test, probabilities),
            "Precision": precision_score(y_test, predicted, zero_division=0),
            "Recall": recall_score(y_test, predicted, zero_division=0),
            "F1": f1_score(y_test, predicted, zero_division=0),
            "MCC": matthews_corrcoef(y_test, predicted)
        }
        rows.append(row)
        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, MODEL_DIR / filename)
        print(f"Saved {filename} | Accuracy={row['Accuracy']:.4f} | AUC={row['AUC']:.4f}")

    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    print("\nMODEL COMPARISON")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nTraining complete. Models, test_data.csv and comparison results are saved.")


if __name__ == "__main__":
    main()
