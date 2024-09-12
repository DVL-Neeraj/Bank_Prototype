import pandas as pd


def calculate_age_group(age):
    age_group = pd.cut(
        age,
        bins=(18, 24, 30, 35, 40, 45, 50, 55, 60, 70),
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
    df.loc[(df["cibil"] < 550), "probability"] += -0.1
    df.loc[(550 <= df["cibil"]) & (df["cibil"] < 600), "probability"] += -0.1
    df.loc[(600 <= df["cibil"]) & (df["cibil"] < 650), "probability"] += -0.1
    #df.loc[(650 <= df["cibil"]) & (df["cibil"] < 750), "probability"] += 0.05
    #df.loc[(750 <= df["cibil"]), "probability"] += 0.1
    df.loc[:,"probability"] += ((df["cibil"] - 650) * 0.002).clip(-0.35,0.35)
    return df["probability"]


def conversion_duration_probability(
    df: pd.DataFrame,
    threshold,
):
    df.loc[(df["conversion_duration"] >= threshold), "probability"] -= 0.3
    df.loc[(df["conversion_duration"] < threshold), "probability"] += 0.05
    return df["probability"]


def reluctance_to_purchase(df: pd.DataFrame):
    df.loc[
        (df["rm_products_sold"] == 1)
        & ((df["rm_meetings"] > 3) | (df["rm_meetings_rescheduled"] > 4)),
        "probability",
    ] += -0.5
    df.loc[
        (df["rm_products_sold"] == 0)
        & ((df["rm_meetings"] > 3) | (df["rm_meetings_rescheduled"] > 2)),
        "probability",
    ] += -0.25

    return df["probability"]


def rm_relation(df: pd.DataFrame):
    df.loc[(df["rm_products_sold"] >= 2), "probability"] += 0.15
    df.loc[(df["rm_products_sold"] == 1), "probability"] += 0.1
    df.loc[(df["rm_products_sold"] == 0), "probability"] += 0
    df.loc[(df["rm_complaints"] >= 3), "probability"] += -0.75
    df.loc[(df["rm_complaints"] == 2), "probability"] += -0.5
    df.loc[(df["rm_complaints"] == 1), "probability"] += -0.25
    df.loc[(df["rm_complaints"] == 0), "probability"] += 0
    df.loc[:, "probability"] += (df["rm_meetings"] * 0.02) - (
        df["rm_meetings_rescheduled"] * 0.04
    )

    return df["probability"]


def categorical_probability(df: pd.DataFrame, categories: dict):

    for feature, values in categories.items():
        for value, weight in values.items():
            df.loc[df[f"{feature}"] == value, "probability"] += weight

    return df["probability"]


def education_loan_opportunity(df: pd.DataFrame):
    df["probability"] -= 0.35
    df.loc[(df["month"] == 1), "probability"] += 0.15
    df.loc[(df["month"] == 4), "probability"] += 0.15
    df.loc[(df["month"] == 5) & (df["day"] <= 15), "probability"] += 0.15
    df.loc[(df["month"] == 7) & (df["day"] >= 15), "probability"] += 0.15
    df.loc[(df["month"] == 8), "probability"] += 0.15
    df.loc[(df["month"] == 9) & (df["day"] <= 15), "probability"] += 0.15
    df.loc[(df["month"] == 12), "probability"] += 0.15

    return df["probability"]
