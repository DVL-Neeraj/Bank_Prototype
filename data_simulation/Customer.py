import itertools

from scipy.stats import multivariate_normal

from .features import (
    age_salary_multiplier,
    education_salary_multiplier,
    gender_salary_multiplier,
    net_worth_multiplier,
    region_salary_multiplier,
    salaries,
)
from .helpers import (
    calculate_age_from_age_group,
    calculate_cibil_score,
    calculate_number_complaints,
    calculate_number_meetings,
    calculate_number_products_sold,
    calculate_number_rescheduled_meetings,
    calculate_terminates_products_early,
)
from .providers import fake


class Customer:
    customer_id = itertools.count()

    def __init__(self):

        # generate the attributes

        self.id = next(Customer.customer_id)
        self.gender = fake.gender()
        self.age_group = fake.age_group()
        self.age = calculate_age_from_age_group(self.age_group)
        self.education_level = fake.education_level()
        self.job = fake.job()
        self.employer = fake.employer()
        self.region = fake.region()
        self.loan_history = fake.loan_history()
        self.cibil = calculate_cibil_score(self.loan_history)
        self.event = fake.event()
        self.set_salary()
        self.set_net_worth()
        self.check_unemployed()
        self.rm_skillset = fake.skill()
        self.rm_products_sold = calculate_number_products_sold(self.loan_history)
        self.rm_complaints = calculate_number_complaints(self.rm_skillset)
        self.rm_meetings = calculate_number_meetings()
        self.rm_meetings_rescheduled = calculate_number_rescheduled_meetings(
            self.rm_meetings
        )
        self.terminates_products_early = calculate_terminates_products_early(
            self.rm_products_sold
        )
        self.kyc_status = fake.kyc_status()
        self.marital_status = fake.marital_status()
        self.date = str(fake.date_this_year(before_today=True, after_today=True))

        del self.age_group

    def check_unemployed(self):
        # unemployed should not have a job or salary
        if self.employer == "Unemployed":
            self.job = "None"
            self.salary = 0

    def set_salary(self):
        # salary should be based on job but vary along certain stratums (on average)
        salary_base = (
            salaries[self.job]
            * age_salary_multiplier[self.age_group]
            * gender_salary_multiplier[self.gender]
            * education_salary_multiplier[self.education_level]
            * region_salary_multiplier[self.region]
        )
        # introduce some randomness into the salary
        self.salary = max(
            0.25 * salary_base,
            round(
                multivariate_normal.rvs(
                    mean=salary_base, cov=(0.25 * salary_base) ** 2, size=1
                )
            ),
        )

    def set_net_worth(self):
        # a rule of thumb for net worth is age / 10 * salary; add a net_worth_multiplier to simulate outliers
        net_worth_rule_of_thumb = (
            self.age / 10 * self.salary * net_worth_multiplier[fake.net_worth_group()]
        )

        # introduce some randomness into the net worth
        self.net_worth = max(
            round(0.25 * net_worth_rule_of_thumb),
            round(
                multivariate_normal.rvs(
                    mean=net_worth_rule_of_thumb,
                    cov=(0.15 * net_worth_rule_of_thumb) ** 2,
                    size=1,
                )
            ),
        )
