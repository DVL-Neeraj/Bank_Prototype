import pickle

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_model_mutual_fund():
    display_name = "mutual_fund"
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
            ("Classifier", MLPClassifier(hidden_layer_sizes=[50, 50, 50, 50, 50, 50])),
        ]
    )

    pipe.fit(
        df[feature_columns].to_dict(orient="records"),
        df["label"],
    )

    artifact_filename = f"model_{display_name}.pkl"
    local_path = f"data/models/{artifact_filename}"
    with open(local_path, "wb") as model_file:
        pickle.dump(pipe, model_file)
