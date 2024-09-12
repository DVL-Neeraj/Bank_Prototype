import pickle

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_model_housing_loan():
    display_name = "housing_loan"
    feature_columns = [
        "gender",
        "age",
        "education_level",
        "region",
        "employer",
        "rm_products_sold",
        "rm_meetings",
        "rm_meetings_rescheduled",
        "rm_complaints",
        "loan_history",
        "event",
        "salary",
        "net_worth",
        "rm_skillset",
        "cibil",
        "terminates_products_early",
        "unfavorable_profession",
        "unfavorable_employer",
        "kyc_status",
        "marital_status",
        "conversion_duration",
        "interest_rate",
        "date",
        "loan_size",
        "loan_duration",
        ]

    df = pd.read_csv(
        f"./data/output/customers_{display_name}.csv",
        keep_default_na=False,
        index_col="id",
        )

    pipe = Pipeline(
        [
            ("Vectorizer", DictVectorizer()),
            ("Standardization", StandardScaler(with_mean=False)),
            ("Classifier", MLPClassifier())  # Model Placeholder
            ]
        )

    param_grid = [
        {
            "Classifier": [LogisticRegression(max_iter=250)],
            "Classifier__penalty": ["l2"]
            },
        {
            "Classifier": [MLPClassifier()],
            "Classifier__hidden_layer_sizes": [(20, 20, 20, 20, 20)],
            "Classifier__alpha": [0.15]
            }
        ]

    grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring="roc_auc", refit=True)  # No data leakage if standardization is happening inside each fold

    grid_search.fit(
        df[feature_columns].to_dict(orient="records"),
        df["label"],
        )

    artifact_filename = f"model_{display_name}.pkl"
    local_path = f"data/models/{artifact_filename}"
    with open(local_path, "wb") as model_file:
        pickle.dump(grid_search.best_estimator_, model_file)
