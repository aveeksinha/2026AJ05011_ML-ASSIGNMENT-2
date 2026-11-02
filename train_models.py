
# ============================================================
# ML ASSIGNMENT 2
# EEG EYE STATE CLASSIFICATION
# Train and compare five improved ML models
# ============================================================

from pathlib import Path
import warnings
import joblib
import pandas as pd

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    balanced_accuracy_score
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "results"

TARGET = "eyeDetection"

FEATURES = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
]

RANDOM_STATE = 42


# ============================================================
# QUESTION 1: LOAD DATASET
# ============================================================

def load_dataset():

    print("Downloading EEG Eye State dataset...")

    dataset = fetch_ucirepo(id=264)

    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    if TARGET not in y.columns:
        y.columns = [TARGET]

    data = pd.concat([X, y[TARGET]], axis=1)

    data = data[FEATURES + [TARGET]].dropna()
    data[TARGET] = data[TARGET].astype(int)

    return data.reset_index(drop=True)


# ============================================================
# QUESTION 2: BUILD MODELS AND PARAMETER GRIDS
# ============================================================

def get_model_grids():

    models = {

        "Logistic Regression": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    max_iter=3000,
                    random_state=RANDOM_STATE
                ))
            ]),
            {
                "classifier__C": [0.1, 1, 10],
                "classifier__class_weight": [None, "balanced"]
            }
        ),

        "Decision Tree": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", DecisionTreeClassifier(
                    random_state=RANDOM_STATE
                ))
            ]),
            {
                "classifier__max_depth": [5, 10, 20, None],
                "classifier__min_samples_leaf": [2, 5, 10],
                "classifier__class_weight": [None, "balanced"]
            }
        ),

        "K-Nearest Neighbors": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier())
            ]),
            {
                "classifier__n_neighbors": [5, 7, 11, 15],
                "classifier__weights": ["uniform", "distance"],
                "classifier__p": [1, 2]
            }
        ),

        "Gaussian Naive Bayes": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", GaussianNB())
            ]),
            {
                "classifier__var_smoothing": [
                    1e-11, 1e-10, 1e-9, 1e-8, 1e-7
                ]
            }
        ),

        "Support Vector Machine": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", SVC(
                    probability=False,
                    random_state=RANDOM_STATE
                ))
            ]),
            {
                "classifier__kernel": ["rbf", "linear"],
                "classifier__C": [0.1, 1, 10],
                "classifier__gamma": ["scale", "auto"]
            }
        )
    }

    return models


# ============================================================
# QUESTION 3: TRAIN AND TUNE ALL MODELS
# ============================================================

def main():

    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    data = load_dataset()

    X = data[FEATURES]
    y = data[TARGET]

    # Stratified split keeps class proportions similar.
    # Note: random splitting can be optimistic for continuous EEG data.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nTraining rows:", len(X_train))
    print("Testing rows:", len(X_test))

    print("\nTraining class distribution:")
    print(y_train.value_counts(normalize=True))

    print("\nTesting class distribution:")
    print(y_test.value_counts(normalize=True))

    # Save the labelled test data for Streamlit.
    test_data = X_test.copy()
    test_data[TARGET] = y_test.values

    test_data.to_csv(ROOT / "test_data.csv", index=False)

    model_grids = get_model_grids()

    results = []

    for name, (pipeline, parameter_grid) in model_grids.items():

        print("\n" + "=" * 55)
        print("Training:", name)
        print("=" * 55)

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring="f1_macro",
            cv=3,
            n_jobs=-1,
            refit=True
        )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_

        # Enable probability predictions for the final SVM model.
        if name == "Support Vector Machine":
            best_model.set_params(classifier__probability=True)
            best_model.fit(X_train, y_train)

        predicted = best_model.predict(X_test)
        probabilities = best_model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predicted)
        auc = roc_auc_score(y_test, probabilities)
        precision = precision_score(
            y_test, predicted, zero_division=0
        )
        recall = recall_score(
            y_test, predicted, zero_division=0
        )
        f1 = f1_score(
            y_test, predicted, zero_division=0
        )
        mcc = matthews_corrcoef(y_test, predicted)
        balanced_acc = balanced_accuracy_score(y_test, predicted)

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "AUC": auc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "MCC": mcc,
            "Balanced Accuracy": balanced_acc
        })

        # Save model using the same filename convention as app.py.
        filename = name.lower().replace(" ", "_") + ".joblib"

        joblib.dump(best_model, MODEL_DIR / filename)

        print("\nBest parameters:")
        print(search.best_params_)

        print("\nBest validation F1-macro:")
        print(round(search.best_score_, 4))

        print("\nTest results:")
        print("Accuracy:", round(accuracy, 4))
        print("AUC:", round(auc, 4))
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1:", round(f1, 4))
        print("MCC:", round(mcc, 4))
        print("Balanced Accuracy:", round(balanced_acc, 4))

        print("\nSaved:", filename)

    # ========================================================
    # QUESTION 4: COMPARE ALL MODELS
    # ========================================================

    comparison = pd.DataFrame(results)

    comparison = comparison.sort_values(
        by="F1",
        ascending=False
    )

    comparison.to_csv(
        RESULTS_DIR / "model_comparison.csv",
        index=False
    )

    print("\n\nFINAL MODEL COMPARISON")
    print(comparison.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    ))

    print("\nAll models trained and saved successfully.")


if __name__ == "__main__":
    main()
