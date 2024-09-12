import numpy as np
import pandas as pd

from ..helpers import (calculate_age_group, categorical_probability,
                       cibil_score_probability,
                       conversion_duration_probability, reluctance_to_purchase,
                       rm_relation)

categories = {
    "age_group": {
        "18-24": 0.0,
        "24-30": 0.2,
        "30-35": 0.2,
        "35-40": 0.0,
        "40-45": 0.0,
        "45-50": 0.15,
        "50-55": 0.15,
        "55-60": 0.1,
        "60-70": 0.1,
    },
    "gender": {"M": 0, "F": 0},
    "education_level": {
        "None": 0.05,
        "High School Diploma": 0.07,
        "Bachelors": 0.1,
        "Masters": 0.13,
        "PhD": 0.15,
    },
    "region": {"Urban": 0, "Rural": 0, "Negative Zone": -0.3},
    "employer": {"Unemployed": -0.5, "Employed": 0.25, "Self-Employed": 0.1},
    "event": {"None": 0},
    "rm_skillset": {
        "Knowledgeable Regarding Mutual Funds": 0.1,
        "Low Knowledge": -0.1,
        "Low Effort": -0.1,
        "Knowledgeable But Needs Assistance": 0.05,
        "Follows Through": 0.05,
    },
    "terminates_products_early": {True: -0.5, False: 0},
    "unfavorable_profession": {True: -0.5, False: 0},
    "unfavorable_employer": {True: -2, False: 0},
    "kyc_status": {"Non-Compliant": -0.75, "Compliant": 0},
    "marital_status": {"Married": 0.05, "Unmarried": 0.0},
}


def assign_label_mutual_fund(base_df):

    df = base_df

    df["conversion_duration"] = np.random.randint(0, 40, df.shape[0])
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

    df["probability"] += df["interest_rate"] * -0.02
    df["probability"] = cibil_score_probability(df[["probability", "cibil"]])
    df["probability"] = conversion_duration_probability(
        df[["probability", "conversion_duration"]], 30
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

    df.loc[(df["net_worth"] > 30000000), "probability"] += 0.2
    df.loc[(df["net_worth"] > 300000000), "probability"] += 0.2
    df.loc[(df["net_worth"] > 3000000000), "probability"] += 0.2

    df["probability"] = categorical_probability(df, categories)
    df["label"] = np.random.rand(df.shape[0]) < df["probability"]
    df["label"] = df["label"].map({True: "Mutual Fund", False: "None"})
    df.drop(["probability", "age_group"], axis=1, inplace=True)

    return df


#        self.principal_amount = self.features["salary"] * self.loan_multiplier
#        self.interest = self.principal_amount * self.interest_rate
#        self.loan_multiplier = 5
#        self.interest_rate = 0.11
