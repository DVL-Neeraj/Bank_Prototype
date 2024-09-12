import numpy as np
import pandas as pd

from ..helpers import (calculate_age_group, categorical_probability,
                       cibil_score_probability,
                       conversion_duration_probability, reluctance_to_purchase,
                       rm_relation)

categories = {
    "age_group": {
        "18-24": 0.05,
        "24-30": 0.1,
        "30-35": 0.1,
        "35-40": 0.1,
        "40-45": 0.05,
        "45-50": 0.025,
        "50-55": 0.0,
        "55-60": -0.2,
        "60-70": -0.3,
    },
    "gender": {"M": 0, "F": 0},
    "education_level": {
        "None": 0.03,
        "High School Diploma": 0.04,
        "Bachelors": 0.05,
        "Masters": 0.06,
        "PhD": 0.07,
    },
    "region": {"Urban": 0, "Rural": 0, "Negative Zone": -0.3},
    "employer": {"Unemployed": -0.25, "Employed": 0.05, "Self-Employed": 0.03},
    "loan_history": {
        "None": 0,
        "Loans repaid on time": 0.1,
        "Loan repayment on schedule": 0.05,
        "Loans Not Repaid On Time": -0.5,
    },
    "event": {"None": 0, "Navratri": 0.1, "Shradhsa": -0.1},
    "rm_skillset": {
        "Knowledgeable Regarding Housing Loans": 0.05,
        "Low Knowledge": -0.05,
        "Low Effort": -0.05,
        "Knowledgeable But Needs Assistance": 0.025,
        "Follows Through": 0.025,
    },
    "terminates_products_early": {True: -0.6, False: 0},
    "unfavorable_profession": {True: -0.6, False: 0},
    "unfavorable_employer": {True: -0.6, False: 0},
    "kyc_status": {"Non-Compliant": -0.5, "Compliant": 0},
    "marital_status": {"Married": -0.05, "Unmarried": 0.05},
}


def assign_label_housing_loan(base_df):

    df = base_df

    df["conversion_duration"] = np.random.randint(0, 50, df.shape[0])
    df["interest_rate"] = np.round(np.random.uniform(0, 0.15, df.shape[0]), 2)
    df["age_group"] = calculate_age_group(df["age"])
    df["gender"] = df["gender"].map({"Female": "F", "Male": "M"})
    df["probability"] = 0.01
    df["unfavorable_profession"] = pd.cut(
        np.random.random(df.shape[0]), bins=(0, 0.96, 1), labels=(False, True)
    )
    df["unfavorable_employer"] = pd.cut(
        np.random.random(df.shape[0]), bins=(0, 0.96, 1), labels=(False, True)
    )

    df["probability"] += df["interest_rate"] / (-3)
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

    df.loc[:,"probability"] += ((df["net_worth"] / 500000) * 0.01).clip(0,0.1)

    df["loan_duration"] = np.round(np.random.randint(5, 30, df.shape[0]))
    df["probability"] += ((df["loan_duration"]) * -0.01).clip(-0.25,0)
    
    df["loan_size"] = np.round(np.random.randint(1, 10, df.shape[0])) * 1000000
    df["probability"] += (5 - (df["loan_size"] / 1000000)) * 0.1

    df["probability"] = categorical_probability(df, categories)
    df["label"] = np.random.rand(df.shape[0]) < df["probability"]
    df["label"] = df["label"].map({True: "Housing Loan", False: "None"})
    df.drop(["probability", "job", "age_group"], axis=1, inplace=True)
    return df


#        self.principal_amount = self.features["salary"] * self.loan_multiplier
#        self.interest = self.principal_amount * self.interest_rate
#        self.loan_multiplier = 5
#        self.interest_rate = 0.11
