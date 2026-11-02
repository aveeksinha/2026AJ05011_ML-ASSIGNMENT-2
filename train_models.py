# ============================================================
# ML ASSIGNMENT 2
# EEG Eye State Classification
#
# Models:
# 1. Logistic Regression
# 2. Decision Tree
# 3. K-Nearest Neighbors
# 4. Gaussian Naive Bayes
# 5. Support Vector Machine
#
# This program:
# - Downloads the EEG Eye State dataset
# - Splits the data into training and testing sets
# - Tunes the models using GridSearchCV
# - Evaluates all five models
# - Saves the trained models
# - Saves the model comparison results
# - Saves the test dataset
# ============================================================


# ============================================================
# 1. IMPORTING REQUIRED LIBRARIES
# ============================================================

import os
import joblib
import pandas as pd

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    PowerTransformer
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    balanced_accuracy_score
)


# ============================================================
# 2. BASIC SETTINGS
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

MODEL_FOLDER = "model"
RESULT_FOLDER = "results"

# Creating folders if they do not already exist
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


# ============================================================
# 3. DOWNLOADING THE EEG EYE STATE DATASET
# ============================================================

print("\nDownloading EEG Eye State dataset...")

# UCI dataset ID = 264
eeg_data = fetch_ucirepo(id=264)

# Getting input features and target variable
X = eeg_data.data.features.copy()
y = eeg_data.data.targets.copy()

# Converting target into a Series
if isinstance(y, pd.DataFrame):
    target_name = y.columns[0]
    y = y.iloc[:, 0]
else:
    target_name = "eyeDetection"

# Removing rows where target is missing
valid_rows = y.notna()

X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

# Converting feature columns into numeric values
X = X.apply(pd.to_numeric, errors="coerce")

# Converting target into integer values
y = pd.to_numeric(y, errors="coerce")

# Removing rows where target conversion failed
valid_rows = y.notna()

X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].astype(int)

print("\nDataset downloaded successfully!")

print("Number of observations:", X.shape[0])
print("Number of features:", X.shape[1])

print("\nTarget class distribution:")
print(y.value_counts())


# ============================================================
# 4. SPLITTING DATA INTO TRAINING AND TESTING SETS
# ============================================================

# Stratified splitting keeps the class proportions similar
# in both training and testing datasets.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining observations:", X_train.shape[0])
print("Testing observations:", X_test.shape[0])


# ============================================================
# 5. DEFINING THE FIVE MACHINE LEARNING MODELS
# ============================================================

def get_model_grids():

    model_grids = {

        # ----------------------------------------------------
        # MODEL 1: LOGISTIC REGRESSION
        # ----------------------------------------------------

        "Logistic Regression": (

            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    max_iter=5000,
                    random_state=RANDOM_STATE
                ))
            ]),

            {
                "scaler": [
                    StandardScaler(),
                    RobustScaler()
                ],

                "classifier__C": [
                    0.001, 0.01, 0.1, 1, 10, 100
                ],

                "classifier__class_weight": [
                    None, "balanced"
                ]
            }
        ),


        # ----------------------------------------------------
        # MODEL 2: DECISION TREE
        # ----------------------------------------------------

        "Decision Tree": (

            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),

                ("classifier", DecisionTreeClassifier(
                    random_state=RANDOM_STATE
                ))
            ]),

            {
                "classifier__max_depth": [
                    5, 10, 20, None
                ],

                "classifier__min_samples_leaf": [
                    2, 5, 10
                ],

                "classifier__class_weight": [
                    None, "balanced"
                ]
            }
        ),


        # ----------------------------------------------------
        # MODEL 3: K-NEAREST NEIGHBORS
        # ----------------------------------------------------

        "K-Nearest Neighbors": (

            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),

                ("classifier", KNeighborsClassifier())
            ]),

            {
                "classifier__n_neighbors": [
                    5, 7, 11, 15
                ],

                "classifier__weights": [
                    "uniform", "distance"
                ],

                "classifier__p": [
                    1, 2
                ]
            }
        ),


        # ----------------------------------------------------
        # MODEL 4: GAUSSIAN NAIVE BAYES
        # ----------------------------------------------------

        "Gaussian Naive Bayes": (

            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),

                ("transformer", PowerTransformer(
                    method="yeo-johnson"
                )),

                ("classifier", GaussianNB())
            ]),

            {
                "transformer": [
                    PowerTransformer(method="yeo-johnson"),

                    PowerTransformer(
                        method="yeo-johnson",
                        standardize=False
                    ),

                    StandardScaler(),

                    RobustScaler()
                ],

                "classifier__var_smoothing": [
                    1e-12,
                    1e-11,
                    1e-10,
                    1e-9,
                    1e-8,
                    1e-7,
                    1e-6,
                    1e-5
                ]
            }
        ),


        # ----------------------------------------------------
        # MODEL 5: SUPPORT VECTOR MACHINE
        # ----------------------------------------------------

        "Support Vector Machine": (

            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),

                ("classifier", SVC(
                    probability=True,
                    random_state=RANDOM_STATE
                ))
            ]),

            {
                "classifier__kernel": [
                    "rbf", "linear"
                ],

                "classifier__C": [
                    0.1, 1, 10
                ],

                "classifier__gamma": [
                    "scale", "auto"
                ]
            }
        )
    }

    return model_grids


# ============================================================
# 6. TRAINING AND EVALUATING THE MODELS
# ============================================================

def main():

    model_grids = get_model_grids()

    results = []

    print("\nStarting model training and hyperparameter tuning...")

    # Going through each model one by one
    for name, (pipeline, parameter_grid) in model_grids.items():

        print("\n" + "=" * 60)
        print("Training:", name)
        print("=" * 60)

        # Choosing the scoring method
        # Logistic Regression and GaussianNB:
        # We are prioritizing accuracy during tuning.
        #
        # Other models:
        # We are prioritizing macro F1-score.

        if name in [
            "Logistic Regression",
            "Gaussian Naive Bayes"
        ]:

            scoring_method = "accuracy"

        else:

            scoring_method = "f1_macro"

        print("Tuning metric:", scoring_method)

        # GridSearchCV checks different parameter combinations
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring=scoring_method,
            cv=3,
            n_jobs=-1,
            refit=True
        )

        # Training and tuning
        search.fit(X_train, y_train)

        # Getting the best model
        best_model = search.best_estimator_

        print("\nBest parameters:")
        print(search.best_params_)

        print("\nBest cross-validation score:")
        print(search.best_score_)

        # Predicting test data
        y_pred = best_model.predict(X_test)

        # Getting prediction probabilities
        y_probability = best_model.predict_proba(X_test)

        # Identifying the column corresponding to class 1
        class_labels = list(best_model.classes_)
        positive_class_index = class_labels.index(1)

        positive_probability = y_probability[
            :, positive_class_index
        ]

        # ----------------------------------------------------
        # CALCULATING EVALUATION METRICS
        # ----------------------------------------------------

        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0
        )

        auc = roc_auc_score(
            y_test,
            positive_probability
        )

        mcc = matthews_corrcoef(
            y_test,
            y_pred
        )

        balanced_accuracy = balanced_accuracy_score(
            y_test,
            y_pred
        )

        # ----------------------------------------------------
        # DISPLAYING RESULTS
        # ----------------------------------------------------

        print("\nTest Results:")

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("AUC:", auc)
        print("MCC:", mcc)
        print("Balanced Accuracy:", balanced_accuracy)

        # ----------------------------------------------------
        # SAVING TRAINED MODEL
        # ----------------------------------------------------

        # Replacing spaces with underscores in filenames
        model_filename = name.lower().replace(" ", "_")

        model_path = os.path.join(
            MODEL_FOLDER,
            model_filename + ".joblib"
        )

        joblib.dump(best_model, model_path)

        print("\nModel saved at:", model_path)

        # ----------------------------------------------------
        # STORING RESULTS
        # ----------------------------------------------------

        results.append({

            "Model": name,

            "Accuracy": accuracy,

            "AUC": auc,

            "Precision": precision,

            "Recall": recall,

            "F1": f1,

            "MCC": mcc,

            "Balanced Accuracy": balanced_accuracy,

            "Best CV Score": search.best_score_,

            "Best Parameters": str(search.best_params_)

        })


    # ========================================================
    # 7. CREATING MODEL COMPARISON TABLE
    # ========================================================

    results_df = pd.DataFrame(results)

    # Sorting models according to test accuracy
    results_df = results_df.sort_values(
        by="Accuracy",
        ascending=False
    )

    # Saving comparison results
    results_path = os.path.join(
        RESULT_FOLDER,
        "model_comparison.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)

    print(results_df.to_string(index=False))

    print("\nComparison saved at:", results_path)


    # ========================================================
    # 8. SAVING TEST DATA
    # ========================================================

    # Saving test features along with actual target values.
    # This file can be uploaded to the Streamlit application
    # for evaluating the saved models.

    test_data = X_test.copy()

    test_data[target_name] = y_test

    test_data_path = "test_data.csv"

    test_data.to_csv(
        test_data_path,
        index=False
    )

    print("\nTest dataset saved at:", test_data_path)


    # ========================================================
    # 9. DISPLAYING BEST MODEL
    # ========================================================

    best_model_row = results_df.iloc[0]

    print("\n" + "=" * 60)
    print("BEST MODEL BASED ON TEST ACCURACY")
    print("=" * 60)

    print("Model:", best_model_row["Model"])

    print("Accuracy:", best_model_row["Accuracy"])

    print("F1 Score:", best_model_row["F1"])

    print("AUC:", best_model_row["AUC"])

    print("\nAll models have been trained successfully!")


# ============================================================
# 10. RUNNING THE PROGRAM
# ============================================================

if __name__ == "__main__":

    main()