import csv
import os
from random import random

from scipy.stats import powerlaw, uniform


def write_output_to_csv(file_path, array):
    with open(
        file_path,
        "a",
    ) as f:
        writer = csv.writer(f)
        if os.path.getsize(file_path) == 0:
            writer.writerow(list(array[0].__dict__.keys()))
        for object in array:
            writer.writerow(list(object.__dict__.values()))


def create_dict_from_csv(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        rows = f.readlines()
        dictionary = {}
        for row in rows:
            key, value = row.split(",")[0], float(row.split(",")[1].strip())
            dictionary[key] = value
        return dictionary


def create_dict_from_keys_list(dictionary, array):
    new_dictionary = dict(zip(list(dictionary.keys()), array))
    return new_dictionary


def calculate_age_from_age_group(age_group):
    min_age = int(age_group.split("-")[0])
    max_age = int(age_group.split("-")[1])
    age = round(uniform.rvs(loc=min_age, scale=max_age - min_age))
    return age


def calculate_cibil_score(loan_history):
    if loan_history == "Loans Not Repaid On Time":
        score = uniform.rvs(350, 250)
    else:
        roll = random()
        if roll <= 0.2:
            score = uniform.rvs(350, 200)
        elif roll <= 0.5:
            score = uniform.rvs(550, 100)
        elif roll <= 0.8:
            score = uniform.rvs(650, 100)
        else:
            score = uniform.rvs(750, 150)
    return int(score)


def calculate_number_rescheduled_meetings(meetings):
    if meetings == 0:
        roll = round(powerlaw.rvs(a=0.2, loc=0, scale=3))
    else:
        roll = round(powerlaw.rvs(a=0.4, loc=0, scale=5))
    return roll


def calculate_number_complaints(skillset):
    if skillset == "Follows Through":
        roll = round(powerlaw.rvs(a=0.05, loc=0, scale=3))
    elif skillset == "Low Knowledge" or skillset == "Low Effort":
        roll = round(powerlaw.rvs(a=0.3, loc=0, scale=5))
    else:
        roll = round(powerlaw.rvs(a=0.1, loc=0, scale=3))
    return roll


def calculate_number_products_sold(loan_history):
    roll = random()
    if loan_history == "No Loan History":
        return 0
    else:
        if roll <= 0.6:
            return 1
        elif roll <= 0.8:
            return 2
        elif roll <= 0.95:
            return 3
        else:
            return 4


def calculate_number_meetings():
    roll = round(powerlaw.rvs(a=0.1, loc=0, scale=5))
    return roll


def calculate_terminates_products_early(number_of_products_sold):
    roll = random()
    if number_of_products_sold >= 2:
        if roll <= 0.8:
            return False
        else:
            return True
    else:
        return False
