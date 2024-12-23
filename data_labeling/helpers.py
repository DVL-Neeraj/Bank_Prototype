import pandas as pd
import numpy as np

def calculate_age_group(age):
    age_group = pd.cut(
        age,
        bins=(17, 24, 30, 35, 40, 45, 50, 55, 60, 70),
        labels=[
            "18-24",
            "24-30",
            "30-35",
            "35-40",
            "40-45",
            "45-50",
            "50-55",
            "55-60",
            "60-70",
        ],
    )
    return age_group


def cibil_score_probability(df: pd.DataFrame):
    df.loc[(df["co_applicant"] == "Not-Available") & (df["cibil"] != 0) & (df["cibil"] < 550), "cibil_score_probability"] += -0.5
    df.loc[(df["co_applicant"] == "Not-Available") & (550 <= df["cibil"]) & (df["cibil"] < 600), "cibil_score_probability"] += -0.4
    df.loc[(df["co_applicant"] == "Not-Available") & (600 <= df["cibil"]) & (df["cibil"] < 650), "cibil_score_probability"] += 0.0
    df.loc[(df["co_applicant"] == "Not-Available") & (650 <= df["cibil"]) & (df["cibil"] < 700), "cibil_score_probability"] += 0.05
    df.loc[(df["co_applicant"] == "Not-Available") & (700 <= df["cibil"]) & (df["cibil"] < 750), "cibil_score_probability"] += 0.1
    df.loc[(df["co_applicant"] == "Not-Available") & (750 <= df["cibil"]), "cibil_score_probability"] += 0.15

    # df.loc[(df["co_applicant"] == "Available") & (df["cibil"] != 0) & (df["cibil"] < 550), "cibil_score_probability"] += -0.5
    # df.loc[(df["co_applicant"] == "Available") & (550 <= df["cibil"]) & (df["cibil"] < 600), "cibil_score_probability"] += -0.4
    # df.loc[(df["co_applicant"] == "Available") & (600 <= df["cibil"]) & (df["cibil"] < 650), "cibil_score_probability"] += 0.0
    # df.loc[(df["co_applicant"] == "Available") & (650 <= df["cibil"]) & (df["cibil"] < 700), "cibil_score_probability"] += 0.01
    # df.loc[(df["co_applicant"] == "Available") & (700 <= df["cibil"]) & (df["cibil"] < 750), "cibil_score_probability"] += 0.02
    # df.loc[(df["co_applicant"] == "Available") & (750 <= df["cibil"]), "cibil_score_probability"] += 0.03
    
    return df["cibil_score_probability"]

def co_applicant_cibil_score_probability(df: pd.DataFrame):
    df.loc[(df["co_applicant"] == "Available") & (df["co_applicant_cibil"] != 0) & (df["co_applicant_cibil"] < 550), "co_applicant_cibil_score_probability"] += -0.5
    df.loc[(df["co_applicant"] == "Available") & (550 <= df["co_applicant_cibil"]) & (df["co_applicant_cibil"] < 600), "co_applicant_cibil_score_probability"] += -0.4
    df.loc[(df["co_applicant"] == "Available") & (600 <= df["co_applicant_cibil"]) & (df["co_applicant_cibil"] < 650), "co_applicant_cibil_score_probability"] += 0.0
    df.loc[(df["co_applicant"] == "Available") & (650 <= df["co_applicant_cibil"]) & (df["co_applicant_cibil"] < 700), "co_applicant_cibil_score_probability"] += 0.02
    df.loc[(df["co_applicant"] == "Available") & (700 <= df["co_applicant_cibil"]) & (df["co_applicant_cibil"] < 750), "co_applicant_cibil_score_probability"] += 0.04
    df.loc[(df["co_applicant"] == "Available") & (750 <= df["co_applicant_cibil"]), "co_applicant_cibil_score_probability"] += 0.075
    #df.loc[:,"probability"] += ((df["cibil"] - 650) * 0.002).clip(-0.35,0.35)
    return df["co_applicant_cibil_score_probability"]


def conversion_duration_probability(
    df: pd.DataFrame,
    threshold,
):
    df.loc[(df["conversion_duration"] >= threshold), "probability"] -= 0.1
    df.loc[(df["conversion_duration"] < threshold), "probability"] += 0.05
    return df["probability"]


def reluctance_to_purchase(df: pd.DataFrame):
    df.loc[
        (df["rm_products_sold"] == 1)
        & ((df["rm_meetings"] > 3) | (df["rm_meetings_rescheduled"] > 4)),
        "probability",
    ] += -0.25
    df.loc[
        (df["rm_products_sold"] == 0)
        & ((df["rm_meetings"] > 3) | (df["rm_meetings_rescheduled"] > 2)),
        "probability",
    ] += -0.1

    return df["probability"]


def rm_relation(df: pd.DataFrame):
    df.loc[(df["rm_products_sold"] >= 2), "probability"] += 0.15
    df.loc[(df["rm_products_sold"] == 1), "probability"] += 0.1
    df.loc[(df["rm_products_sold"] == 0), "probability"] += 0
    df.loc[(df["rm_complaints"] >= 3), "probability"] += -0.03
    df.loc[(df["rm_complaints"] == 2), "probability"] += -0.02
    df.loc[(df["rm_complaints"] == 1), "probability"] += -0.01
    df.loc[(df["rm_complaints"] == 0), "probability"] += 0
    df.loc[:, "probability"] += (df["rm_meetings"] * 0.02) - (
        df["rm_meetings_rescheduled"] * 0.04
    )

    return df["probability"]


def categorical_probability(df: pd.DataFrame, categories: dict):

    for feature, values in categories.items():
        df[f"{feature}_probability"] = 0
        for value, weight in values.items():
            df.loc[df[f"{feature}"] == value, f"{feature}_probability"] += weight
        df[ "probability"] += df[f"{feature}_probability"]
        df.drop(f"{feature}_probability",axis=1,inplace=True)
    return df 


def education_loan_opportunity(df: pd.DataFrame):
    df["probability"] += 0.0
    df.loc[(df["month"] == 1), "probability"] += 0.1
    df.loc[(df["month"] == 4), "probability"] += 0.1
    df.loc[(df["month"] == 5) & (df["day"] <= 15), "probability"] += 0.1
    df.loc[(df["month"] == 7) & (df["day"] >= 15), "probability"] += 0.1
    df.loc[(df["month"] == 8), "probability"] += 0.1
    df.loc[(df["month"] == 9) & (df["day"] <= 15), "probability"] += 0.1
    df.loc[(df["month"] == 12), "probability"] += 0.1

    return df["probability"]


#def add_co_applicant_age(row):
#    if (row["co_applicant"] == "Available"):
#        row["co_applicant_age"] = row["age"] + np.random.randint(3,31)
#    else:
#        row["co_applicant_age"] = 0
#    return row

def calculate_loan_size_probability(df):
    # Create a condition mask for rows where co_applicant is "Available"
    mask = df["co_applicant"] == "Available"
    df["loan_size_probability"] = 0.0

    #for loan amount calculation
    df["eligible_emi"] = ((df["co_applicant_salary"]/12)*0.5) - df["co_applicant_existing_emi"] 
    df["actual_loan_size"] = ((df["eligible_emi"]) * (1-(1/((1+(0.12/12))**(df["loan_duration"]*12))))) / (0.12/12)   

    # Vectorized calculation for rows with co_applicant
    df.loc[~mask,"actual_loan_size"] = df.loc[~mask,"net_worth"]*0.7  
    df.loc[mask, "loan_size_probability"] += (((df.loc[mask, "actual_loan_size"]-df.loc[mask, "loan_size"])/100000) * 0.01)

    # Vectorized calculation for rows without co_applicant
    df.loc[~mask, "loan_size_probability"] += (((df.loc[~mask, "net_worth"]*0.7)-df.loc[~mask, "loan_size"])/100000) * 0.01
    
    df.loc[df["loan_size_probability"]>0.1,"loan_size_probability"] = 0.1
    df.loc[df["loan_size_probability"]<-0.1,"loan_size_probability"] = -0.1
    return df

def calculate_loan_duration_probability(df):
    # Create a condition mask for rows where co_applicant is "Available"
    mask = df["loan_duration"] <= 6
    df["loan_duration_probability"] = 0.05

    # Vectorized calculation for rows with df["loan_duration"] > 6
    df.loc[~mask, "loan_duration_probability"] -= ((df.loc[~mask, "loan_duration"]-6)) * 0.01

    return df
