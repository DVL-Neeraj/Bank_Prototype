from random import choices

from faker import Faker
from faker.providers import DynamicProvider

from .features import (
    age_groups,
    education_levels,
    employers,
    events,
    genders,
    loan_histories,
    net_worth_groups,
    regions,
    salaries,
    skillset,
    kyc_status,
    marital_status,
)

Faker.seed(0)
fake = Faker("en_in")  # Initialize Faker to an indian english locale

age_group_provider = DynamicProvider(
    provider_name="age_group",
    elements=choices(
        population=list(age_groups.keys()), weights=list(age_groups.values()), k=100000
    ),
)

education_level_provider = DynamicProvider(
    provider_name="education_level",
    elements=choices(
        population=list(education_levels.keys()),
        weights=list(education_levels.values()),
        k=100000,
    ),
)

gender_provider = DynamicProvider(
    provider_name="gender",
    elements=choices(
        population=list(genders.keys()), weights=list(genders.values()), k=100000
    ),
)

region_provider = DynamicProvider(
    provider_name="region",
    elements=choices(
        population=list(regions.keys()), weights=list(regions.values()), k=100000
    ),
)

employer_provider = DynamicProvider(
    provider_name="employer",
    elements=choices(
        population=list(employers.keys()), weights=list(employers.values()), k=100000
    ),
)

net_worth_group_provider = DynamicProvider(
    provider_name="net_worth_group",
    elements=choices(
        population=list(net_worth_groups.keys()),
        weights=list(net_worth_groups.values()),
        k=100000,
    ),
)

loan_history_provider = DynamicProvider(
    provider_name="loan_history",
    elements=choices(
        population=list(loan_histories.keys()),
        weights=list(loan_histories.values()),
        k=100000,
    ),
)

event_provider = DynamicProvider(
    provider_name="event",
    elements=choices(
        population=list(events.keys()), weights=list(events.values()), k=100000
    ),
)

skill_provider = DynamicProvider(
    provider_name="skill",
    elements=choices(
        population=list(skillset.keys()), weights=list(skillset.values()), k=100000
    ),
)

kyc_status_provider = DynamicProvider(
    provider_name="kyc_status",
    elements=choices(
        population=list(kyc_status.keys()), weights=list(kyc_status.values()), k=100000
    ),
)

marital_status_provider = DynamicProvider(
    provider_name="marital_status",
    elements=choices(
        population=list(marital_status.keys()),
        weights=list(marital_status.values()),
        k=100000,
    ),
)

jobs_provider = DynamicProvider(provider_name="job", elements=salaries.keys())

fake.add_provider(age_group_provider)
fake.add_provider(education_level_provider)
fake.add_provider(gender_provider)
fake.add_provider(region_provider)
fake.add_provider(employer_provider)
fake.add_provider(net_worth_group_provider)
fake.add_provider(jobs_provider)
fake.add_provider(loan_history_provider)
fake.add_provider(event_provider)
fake.add_provider(skill_provider)
fake.add_provider(kyc_status_provider)
fake.add_provider(marital_status_provider)
