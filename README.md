# EEG Eye State Classification | ML Assignment 2

M.Tech (AIML) — Machine Learning Assignment 2.

## a. Problem Statement

The objective is to build a binary classification system that predicts whether a person's eyes are open or closed from EEG sensor measurements. Five machine-learning classifiers are trained and compared. A Streamlit application allows a user to upload labelled test data and evaluate a selected saved model.

## b. Dataset Description

- **Dataset:** EEG Eye State
- **Source:** [UCI Machine Learning Repository, dataset 264](https://archive.ics.uci.edu/dataset/264/eeg+eye+state)
- **Creator:** Oliver Roesler (2013)
- **Observations:** 14,980
- **Input features:** 14 EEG channels
- **Target:** `eyeDetection`
- **Task:** Binary classification
- **Labels:** 0 = eyes open; 1 = eyes closed
- **Missing values:** None reported by UCI

The 14 input columns are AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8 and AF4. UCI describes the measurements as a single continuous recording, with rows stored in chronological order.

### Data preparation and split

The training script downloads the dataset using `ucimlrepo`. It uses a stratified random 80:20 train-test split (`test_size=0.20`, `random_state=42`) so that the class proportions are maintained in both subsets. Median imputation and model-specific feature scaling or transformation are included in the pipelines. GridSearchCV with 3-fold cross-validation is used to tune the model hyperparameters. The test file is labelled and includes all 14 features plus `eyeDetection`.

**Evaluation note:** Because the EEG rows come from a continuous recording, randomly splitting rows can place temporally close observations in both training and test sets. This may make the reported scores more optimistic than a strictly chronological or subject-independent evaluation. The scores below should therefore be interpreted as results for this stratified random split.

## c. GitHub Repository Link

[https://github.com/aveeksinha/2026AJ05011_ML-ASSIGNMENT-2](https://github.com/aveeksinha/2026AJ05011_ML-ASSIGNMENT-2)

## d. Models Used and Evaluation Metrics

The five classifiers are:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

Each model is evaluated using Accuracy, AUC, Precision, Recall, F1-score and Matthews Correlation Coefficient (MCC). Precision, Recall and F1 are calculated for the positive class (eyeDetection = 1).

### Model comparison

The following values are from the experiment recorded in `results/model_comparison.csv`. Values are shown to four decimal places.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6272 | 0.6700 | 0.6178 | 0.4446 | 0.5171 | 0.2345 |
| Decision Tree | 0.8274 | 0.8457 | 0.8270 | 0.7784 | 0.8020 | 0.6503 |
| KNN | 0.9119 | 0.9697 | 0.9148 | 0.8862 | 0.9003 | 0.8217 |
| Naive Bayes | 0.5664 | 0.5776 | 0.5635 | 0.1517 | 0.2390 | 0.0854 |
| Support Vector Machine | 0.7463 | 0.8455 | 0.8115 | 0.5665 | 0.6673 | 0.4925 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy is 0.6272 and AUC is 0.6700. Its recall (0.4446) is lower than its precision (0.6178), meaning it misses a considerable number of actual eye-closed observations. The MCC of 0.2345 indicates relatively weak overall agreement. |
| Decision Tree | The Decision Tree performs well, with accuracy of 0.8274 and F1-score of 0.8020. Precision (0.8270) and recall (0.7784) are reasonably balanced. Its AUC of 0.8457 and MCC of 0.6503 also indicate good class separation and prediction agreement. |
| KNN | KNN gives the strongest results across all six reported metrics: accuracy 0.9119, AUC 0.9697, precision 0.9148, recall 0.8862, F1-score 0.9003 and MCC 0.8217. Its precision and recall are both high and relatively balanced, making it the best-performing model in this experiment. |
| Naive Bayes | Naive Bayes has the lowest accuracy (0.5664), AUC (0.5776), F1-score (0.2390) and MCC (0.0854). Its recall is particularly low at 0.1517, showing that it fails to identify many actual eye-closed observations in this test set. |
| Support Vector Machine | SVM achieves moderate accuracy (0.7463) and AUC (0.8455). Its precision is 0.8115, but recall is lower at 0.5665, so it is more conservative in predicting the positive class. Its F1-score is 0.6673 and MCC is 0.4925. |
| Overall conclusion | KNN is the best-performing model on this test split because it has the highest value for each of the six metrics. Decision Tree is the second-strongest model. Logistic Regression and Naive Bayes perform less effectively, especially in recall and MCC. These conclusions apply to the stratified random split used in this experiment; a chronological holdout would be a useful additional robustness check. |

## Project files

```text
2026AJ05011_ML-ASSIGNMENT-2/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv                 # generated by train_models.py
├── model/                        # five saved .joblib models
└── results/
    └── model_comparison.csv      # generated by train_models.py
```

## Run locally or in BITS Virtual Lab

1. Use Python 3.10 or later.
2. Install dependencies: `pip install -r requirements.txt`.
3. Download and train all models: `python train_models.py`.
4. Confirm that `test_data.csv`, five model files in `model/`, and `results/model_comparison.csv` have been created.
5. Start the app: `streamlit run app.py`.
6. Upload `test_data.csv` in the app, or select the repository test file if available.

The app requires a labelled CSV with the 14 EEG feature columns and `eyeDetection`. It loads saved models and does not retrain them on uploaded test data.

## Streamlit Community Cloud deployment

Push the completed project, including the generated model files and test CSV, to GitHub. In Streamlit Community Cloud, create a new app, select this repository, choose the `main` branch and set the main file to `app.py`. Deploy and add the resulting live URL to the final assignment PDF.

## Assignment submission reminders

- Add the generated `test_data.csv` and all five saved models to the repository.
- Keep the model comparison table and observations synchronized with the latest `results/model_comparison.csv` after any retraining.
- Deploy the Streamlit app and include its live URL in the PDF.
- Perform the assignment on BITS Virtual Lab and capture the required screenshot yourself.
- Include the README content in the submitted PDF, in the order required by the assignment.

## Dataset citation

Roesler, O. (2013). *EEG Eye State* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C57G7J. Licensed under CC BY 4.0.
