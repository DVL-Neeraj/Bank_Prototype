import pandas as pd
import os

from .labels.fixed_deposit import assign_label_fixed_deposit
from .labels.housing_loan import assign_label_housing_loan
from .labels.education_loan import assign_label_education_loan
from .labels.mutual_fund import assign_label_mutual_fund
from .labels.personal_loan import assign_label_personal_loan
from .labels.recurring_deposit import assign_label_recurring_deposit
from .labels.saving_account import assign_label_saving_account

# Load the base dataset into a dataframe


def label_products(labels=[]):
    file_path = "data/output/customers.csv"

    if not (os.path.isfile(file_path)):
        print("Base dataset does not exist.")
        print("Labels not assigned.")
        return
    df = pd.read_csv(file_path, keep_default_na=False, index_col="id")

    for label in labels:
        match label:
            case "housing_loan":
                df_housing_loan = assign_label_housing_loan(df)
                df_housing_loan.to_csv(f"{file_path[0:-4]}_housing_loan.csv")
            case "personal_loan":
                df_personal_loan = assign_label_personal_loan(df)
                df_personal_loan.to_csv(f"{file_path[0:-4]}_personal_loan.csv")
            case "education_loan":
                df_education_loan = assign_label_education_loan(df)
                df_education_loan.to_csv(f"{file_path[0:-4]}_education_loan.csv")
            case "mutual_fund":
                df_mutual_fund = assign_label_mutual_fund(df)
                df_mutual_fund.to_csv(f"{file_path[0:-4]}_mutual_fund.csv")
            case "saving_account":
                df_saving_account = assign_label_saving_account(df)
                df_saving_account.to_csv(f"{file_path[0:-4]}_saving_account.csv")
            case "fixed_deposit":
                df_fixed_deposit = assign_label_fixed_deposit(df)
                df_fixed_deposit.to_csv(f"{file_path[0:-4]}_fixed_deposit.csv")
            case "recurring_deposit":
                df_recurring_deposit = assign_label_recurring_deposit(df)
                df_recurring_deposit.to_csv(f"{file_path[0:-4]}_recurring_deposit.csv")
