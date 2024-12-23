import numpy as np
import pandas as pd

from ..helpers import (calculate_age_group, categorical_probability,
                       cibil_score_probability,
                       conversion_duration_probability, reluctance_to_purchase,
                       rm_relation)

categories = {
    "age_group": {
        "18-24": 0.05,
        "24-30": 0.2,
        "30-35": 0.2,
        "35-40": 0.2,
        "40-45": 0.1,
        "45-50": 0.05,
        "50-55": 0.0,
        "55-60": -0.5,
        "60-70": -2.0,
    },
    "gender": {"M": 0, "F": 0},
    "education_level": {
        "None": 0.03,
        "High School Diploma": 0.04,
        "Bachelors": 0.05,
        "Masters": 0.06,
        "PhD": 0.07,
    },
    "region": {"Urban": 0, "Rural": 0, "Negative Zone": -5.0},
    "employer": {"Unemployed": -0.3, "Employed": 0.25, "Self-Employed": 0.15},
    "loan_history": {
        "None": 0,
        "Loans repaid on time": 0.15,
        "Loans Not Repaid On Time": -0.1,
        "Loan Repayment On Schedule": 0.15
    },
    "event": {"None": 0, "Navratri": 0.5, "Shradhsa": -0.5},
    "rm_skillset": {
        "Knowledgeable Regarding Education Loans" : 0.01,
        "Knowledgeable Regarding Housing Loans": 0.2,
        "Knowledgeable Regarding Personal Loans": 0.01,
        "Knowledgeable Regarding Auto Loans":0.05,
        "Low Knowledge": -0.01,
        "Low Effort": -0.01,
        "Knowledgeable But Needs Assistance": 0.05,
        "Follows Through": 0.05,
    },
    "terminates_products_early": {True: -0.6, False: 0},
    "unfavorable_profession": {True: -0.5, False: 0},
    "unfavorable_employer": {True: -0.5, False: 0},
    "kyc_status": {"Non-Compliant": 0, "Compliant": 0},
    "marital_status": {"Married": 0.0, "Unmarried": 0.0},
}


def assign_label_housing_loan(base_df):

    df = base_df

    df["conversion_duration"] = np.random.randint(0, 50, df.shape[0])
    df["interest_rate"] = np.round(np.random.uniform(0, 0.15, df.shape[0]), 2)
    df["age_group"] = calculate_age_group(df["age"])

    df["region"] = np.random.choice(["Urban","Rural","Negative Zone"],df.shape[0],p=[0.47,0.45,0.08])

    df["gender"] = df["gender"].apply(lambda x:'M' if x== "Male" else 'F')
    ###
    df.loc[df["age"] > 65,"age"] = 28
    df.loc[(df["age"] == 53) | (df["age"] == 58) | (df["age"] == 62),"age"] = 30

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

    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month

    df["loan_size"] = np.round(np.random.randint(1, 10, df.shape[0])) * 1000000
    df["probability"] += (5 - (df["loan_size"] / 1000000)) * 0.1

    df["loan_duration"] = np.round(np.random.randint(5, 30, df.shape[0]))
    df["probability"] += ((57 - (df['age']+df['loan_duration']))*1000000/df["loan_size"])*0.1
    
    df["emi"] = (df["loan_size"]*(df["interest_rate"]/12)*((1+(df["interest_rate"]/12))**(df["loan_duration"]*12)))/((1+(df["interest_rate"]/12))**(df["loan_duration"]*12)-1)
    df["probability"] += ((df["salary"]/12)*0.9 - df["emi"])*0.0001

    df["probability"] = categorical_probability(df, categories)
    print(f'''max {df["probability"].max()}, min {df["probability"].min()}, median {df["probability"].median()}, mean {df["probability"].mean()}''')


    df["label"] = "Fair"  # Default label
    df.loc[(df["probability"] >= 0.001) & (df["cibil"] >= 700) & ((df["age"]+df["loan_duration"])<=60), "label"] = "Good"
    df.loc[(df["region"]=="Negative Zone") | (df["kyc_status"]=="Non-Compliant") | (df["cibil"] < 550) | (df["age"]>57),"label"] = "Not Good"

    df.drop(["probability","job","age_group","emi"], axis=1, inplace=True)

    return df


#        self.principal_amount = self.features["salary"] * self.loan_multiplier
#        self.interest = self.principal_amount * self.interest_rate
#        self.loan_multiplier = 5
#        self.interest_rate = 0.11
