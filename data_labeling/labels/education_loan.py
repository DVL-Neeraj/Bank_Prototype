import numpy as np
import pandas as pd

from ..helpers import (calculate_age_group, categorical_probability,
                       cibil_score_probability,
                       conversion_duration_probability,
                       education_loan_opportunity, reluctance_to_purchase,
                       rm_relation)

categories = {
    "age_group": {
        "18-24": 0.15,
        "24-30": 0.2,
        "30-35": 0.1,
        "35-40": 0.1,
        "40-45": 0.1,
        "45-50": 0.0,
        "50-55": -2,
        "55-60": -2,
        "60-70": -2,
    },
    "gender": {"M": 0, "F": 0},
    "education_level": {
        "None": 0.05,
        "High School Diploma": 0.07,
        "Bachelors": 0.10,
        "Masters": 0.13,
        "PhD": 0.0,
    },
    "region": {"Urban": 0, "Rural": 0, "Negative Zone": -0.3},
    "employer": {"Unemployed": -0.1, "Employed": 0.25, "Self-Employed": 0.1},
    "loan_history": {
        "None": 0,
        "Loans repaid on time": 0.15,
        "Loan repayment on schedule": 0.1,
        "Loans Not Repaid On Time": -2,
    },
    "event": {"None": 0, "Navratri": 0.5, "Shradhsa": -0.5},
    "rm_skillset": {
        "Knowledgeable Regarding Housing Loans": 0.1,
        "Low Knowledge": -0.1,
        "Low Effort": -0.1,
        "Knowledgeable But Needs Assistance": 0.05,
        "Follows Through": 0.05,
    },
    "terminates_products_early": {True: -0.6, False: 0},
    "unfavorable_profession": {True: -0.75, False: 0},
    "unfavorable_employer": {True: -2, False: 0},
    "kyc_status": {"Non-Compliant": -2, "Compliant": 0},
    "marital_status": {"Married": -0.05, "Unmarried": 0.05},
}


def assign_label_education_loan(base_df):

    df = base_df

    df["probability"] = 0.01
    df["conversion_duration"] = np.random.randint(0, 50, df.shape[0])
    df["interest_rate"] = np.round(np.random.uniform(0, 0.15, df.shape[0]), 2)
    df["age_group"] = calculate_age_group(df["age"])
    df["gender"] = df["gender"].map({"Female": "F", "Male": "M"})
    df["unfavorable_profession"] = pd.cut(
        np.random.random(df.shape[0]), bins=(0, 0.96, 1), labels=(False, True)
    )
    df["unfavorable_employer"] = pd.cut(
        np.random.random(df.shape[0]), bins=(0, 0.96, 1), labels=(False, True)
    )

    df["probability"] += df["interest_rate"] * -0.02
    df["probability"] = cibil_score_probability(df[["probability", "cibil"]])
    df["probability"] = conversion_duration_probability(
        df[["probability", "conversion_duration"]], 40
    )
    df["probability"] = reluctance_to_purchase(
        df[
            [
                "probability",
                "rm_products_sold",
                "rm_meetings",
                "rm_meetings_rescheduled",
            ]
        ]
    )
    df["probability"] = rm_relation(
        df[
            [
                "probability",
                "rm_products_sold",
                "rm_complaints",
                "rm_meetings",
                "rm_meetings_rescheduled",
            ]
        ],
    )

    df.loc[(df["net_worth"] > 3000000), "probability"] += 0.3
    df.loc[(df["net_worth"] > 30000000), "probability"] += 0.3
    df.loc[(df["net_worth"] > 300000000), "probability"] += 0.3

    df["loan_duration"] = np.round(np.random.randint(3, 30, df.shape[0]))
    df["probability"] += (15 - df["loan_duration"]) * 0.01
    df["loan_size"] = np.round(np.random.randint(5, 40, df.shape[0])) * 100000
    df["probability"] += (2.5 - (df["loan_size"] / 1000000)) * 0.15

    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month

    df["probability"] = education_loan_opportunity(df)

    df["probability"] = categorical_probability(df, categories)
    df["label"] = np.random.rand(df.shape[0]) < df["probability"]
    df["label"] = df["label"].map({True: "Education Loan", False: "None"})
    df.drop(["probability", "age_group"], axis=1, inplace=True)

    return df
