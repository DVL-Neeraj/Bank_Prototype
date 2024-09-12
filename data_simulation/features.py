import os

from .helpers import create_dict_from_csv, create_dict_from_keys_list

dir = "data/input/weights/"

# dynamically create variables that import the feature outcomes and map them to probabilities using csv files as input
age_groups = create_dict_from_csv(dir + "age_groups.csv")
dependents = create_dict_from_csv(dir + "dependents.csv")
education_levels = create_dict_from_csv(dir + "education_levels.csv")
employers = create_dict_from_csv(dir + "employers.csv")
events = create_dict_from_csv(dir + "events.csv")
genders = create_dict_from_csv(dir + "genders.csv")
kyc_status = create_dict_from_csv(dir + "kyc_status.csv")
loan_histories = create_dict_from_csv(dir + "loan_histories.csv")
marital_status = create_dict_from_csv(dir + "marital_status.csv")
net_worth_groups = create_dict_from_csv(dir + "net_worth_groups.csv")
regions = create_dict_from_csv(dir + "regions.csv")
skillset = create_dict_from_csv(dir + "skillset.csv")

# create multipliers for the features / probabilities (e.g. higher age gives a higher salary on average)
age_salary_multiplier = create_dict_from_keys_list(
    age_groups, [0.7, 0.8, 1.0, 1.0, 1.1, 1.1, 1.2, 1.3, 1, 1]
)
gender_salary_multiplier = create_dict_from_keys_list(genders, [1.05, 0.95])
education_salary_multiplier = create_dict_from_keys_list(
    education_levels, [0.8, 0.85, 1, 1.15, 1.3]
)
region_salary_multiplier = create_dict_from_keys_list(regions, [0.8, 1.1, 0.8])
net_worth_multiplier = create_dict_from_keys_list(
    net_worth_groups, [0.5, 1, 3, 10, 15, 25]
)

# import the jobs and salaries as a dict
salaries = create_dict_from_csv("data/input/salaries/jobs.csv")
