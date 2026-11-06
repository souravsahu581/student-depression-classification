# Student Depression Classification

## A. Problem statement

The aim is to compare five supervised classification algorithms on the same student-depression dataset. The project predicts the binary `Depression` target (`Yes` or `No`) from twelve input features and provides a Streamlit application for evaluating saved models on labeled test data. The app is an educational demonstration, not a clinical diagnostic tool.

## B. Dataset description

- **Dataset:** `Student_Depression_Dataset.csv` (source citation/URL: [**Original public dataset link**](https://www.kaggle.com/datasets/mkhoirulmustaqim/student-depression-detection-dataset)).
- **Observations:** 4,464.
- **Input features:** 12.
- **Target:** `Depression` (classes `No` and `Yes`).
- **Input columns:** `Gender`, `Age`, `Family_background`, `Relationship_status`, `Marital_status`, `Living_style`, `Current_semester`, `CGPA`, `friend_circle`, `Health_condition`, `Cost_without_study`, `Home_division`.
- **Class distribution:** No = 3,263; Yes = 1,201.
- **Train/test division:** 80% / 20% (stratified, random state = 42); 3,571 training rows and 893 testing rows.

Numerical inputs are standardized; categorical inputs are one-hot encoded. The preprocessing is contained inside each saved scikit-learn pipeline to avoid leakage from the test set.

## C. GitHub repository link

**GitHub Repository:** [student-depression-classification](https://github.com/souravsahu581/student-depression-classification)

**Live Streamlit Application:** [Launch the interactive app](https://souravsahu581-student-depression-classification-app-hqm52n.streamlit.app/)

## D. Models used and evaluation metrics

The same held-out test set is used for all five classifiers. Scores use `Yes` as the positive class.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9709 | 0.9931 | 0.9612 | 0.9292 | 0.9449 | 0.9254 |
| Decision Tree | 0.9966 | 0.9938 | 1.0000 | 0.9875 | 0.9937 | 0.9915 |
| K-Nearest Neighbors | 0.9474 | 0.9786 | 0.9289 | 0.8708 | 0.8989 | 0.8642 |
| Gaussian Naive Bayes | 0.9496 | 0.9875 | 0.8884 | 0.9292 | 0.9084 | 0.8740 |
| Support Vector Machine | 0.9877 | 0.9965 | 0.9872 | 0.9667 | 0.9768 | 0.9686 |

### Performance observations

| ML Model | Observation |
|---|---|
| Logistic Regression | Strong baseline with 0.9709 accuracy and 0.9449 F1. |
| Decision Tree | Best accuracy (0.9966), F1 (0.9937), and MCC (0.9915) in this held-out test. |
| KNN | Lower recall (0.8708) than the other models; misses more `Yes` cases. |
| Gaussian Naive Bayes | Recall is 0.9292, but precision is lower at 0.8884. |
| Support Vector Machine | Best AUC (0.9965) and high accuracy (0.9877). |
| **Overall winner** | **Decision Tree** based on accuracy, F1, and MCC on this particular test split. The SVM has the highest AUC. |

These scores describe this experiment only and are not guaranteed on other samples. Investigate possible shortcut or target-leakage features before interpreting high accuracy as reliable real-world performance.

## E. Project layout

```text
Student-Depression-Classification/
├── app.py
├── train_classifiers.py
├── requirements.txt
├── README.md
├── Student_Depression_Dataset.csv
├── test_data.csv
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── k_nearest_neighbors.joblib
│   ├── gaussian_naive_bayes.joblib
│   ├── support_vector_machine.joblib
│   └── feature_info.joblib
└── Outputs/
    └── evaluation_metrics.csv
```

## F. How to run

1. Put the repository files in the arrangement shown above.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
streamlit run app.py
```

4. Upload the included **`test_data.csv`** using the sidebar, choose any classifier, and view Accuracy, AUC, Precision, Recall, F1, MCC, the confusion matrix, and the classification report.

The app only calls `predict` / `predict_proba` on saved models. **It never retrains on uploaded test data.** If retraining is needed, run `python train_classifiers.py` separately with the original dataset in place.

## G. Streamlit Community Cloud deployment

1. Push all project files to GitHub. The `model/` directory must include all six `.joblib` files.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/), sign in with GitHub, and create a new app.
3. Select the repository, branch `main`, and entry point `app.py`.
4. Deploy and test the app with `test_data.csv`.
5. Include this README's content in the assignment submission PDF.
