"""Streamlit interface for evaluating the saved EEG Eye State classifiers."""
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET = "eyeDetection"
FEATURES = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2",
            "P8", "T8", "FC6", "F4", "F8", "AF4"]
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k-nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

st.set_page_config(page_title="EEG Eye State Classifier", page_icon="🧠", layout="wide")
st.title("EEG Eye State Classification")
st.write("Evaluate one of five saved classifiers on a labelled CSV file. Models are loaded from disk and are not retrained on uploaded data.")

with st.sidebar:
    st.header("Evaluation settings")
    selected = st.selectbox("Select classifier", list(MODEL_FILES.keys()))
    st.caption("0 = eyes open | 1 = eyes closed")

uploaded = st.file_uploader("Upload labelled test data (CSV)", type=["csv"])
default_path = ROOT / "test_data.csv"
use_default = False
if uploaded is None and default_path.exists():
    use_default = st.checkbox("Use repository test_data.csv")

if uploaded is not None or use_default:
    try:
        data = pd.read_csv(uploaded) if uploaded is not None else pd.read_csv(default_path)
        missing = [column for column in FEATURES + [TARGET] if column not in data.columns]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
            st.stop()
        if data.empty:
            st.error("The CSV has no records.")
            st.stop()

        X_test = data[FEATURES].apply(pd.to_numeric, errors="coerce")
        y_test = pd.to_numeric(data[TARGET], errors="coerce")
        if y_test.isna().any() or not set(y_test.unique()).issubset({0, 1}):
            st.error("eyeDetection must contain only numeric 0 and 1 labels.")
            st.stop()

        model_path = MODEL_DIR / MODEL_FILES[selected]
        if not model_path.exists():
            st.error("Saved model not found. Run train_models.py first and add the generated model files to the repository.")
            st.stop()

        model = joblib.load(model_path)
        predicted = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        metrics = {
            "Accuracy": accuracy_score(y_test, predicted),
            "AUC": roc_auc_score(y_test, probabilities) if y_test.nunique() == 2 else float("nan"),
            "Precision": precision_score(y_test, predicted, zero_division=0),
            "Recall": recall_score(y_test, predicted, zero_division=0),
            "F1 Score": f1_score(y_test, predicted, zero_division=0),
            "MCC": matthews_corrcoef(y_test, predicted)
        }
        st.subheader("Evaluation metrics")
        columns = st.columns(6)
        for column, (label, value) in zip(columns, metrics.items()):
            column.metric(label, f"{value:.4f}")

        left, right = st.columns(2)
        with left:
            st.subheader("Confusion Matrix")
            matrix = confusion_matrix(y_test, predicted, labels=[0, 1])
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Open (0)", "Closed (1)"],
                        yticklabels=["Open (0)", "Closed (1)"], ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            plt.close(fig)
        with right:
            st.subheader("Classification Report")
            report = classification_report(y_test, predicted,
                target_names=["Eyes open", "Eyes closed"],
                output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

        st.subheader("Prediction preview")
        output = data[FEATURES].copy()
        output["Actual eyeDetection"] = y_test.values
        output["Predicted eyeDetection"] = predicted
        st.dataframe(output.head(20), use_container_width=True)
        st.download_button("Download predictions CSV",
            data=output.to_csv(index=False).encode("utf-8"),
            file_name="eeg_eye_state_predictions.csv", mime="text/csv")
    except Exception as error:
        st.error(f"Could not evaluate this file: {error}")
else:
    st.info("Upload a labelled CSV with all 14 EEG features and eyeDetection. Run train_models.py to download the dataset, train models and create test_data.csv.")

st.divider()
st.caption("M.Tech (AIML) | Machine Learning Assignment 2 | UCI EEG Eye State")
