from .deployments.education_loan import deploy_model_education_loan
from .deployments.fixed_deposit import deploy_model_fixed_deposit
from .deployments.housing_loan import deploy_model_housing_loan
from .deployments.mutual_fund import deploy_model_mutual_fund
from .deployments.personal_loan import deploy_model_personal_loan
from .deployments.recurring_deposit import deploy_model_recurring_deposit
from .deployments.saving_account import deploy_model_saving_account
from .models.education_loan import train_model_education_loan
from .models.fixed_deposit import train_model_fixed_deposit
from .models.housing_loan import train_model_housing_loan
from .models.mutual_fund import train_model_mutual_fund
from .models.personal_loan import train_model_personal_loan
from .models.recurring_deposit import train_model_recurring_deposit
from .models.saving_account import train_model_saving_account


def train_models(models):
    for model in models:
        match model:
            case "housing_loan":
                train_model_housing_loan()
            case "education_loan":
                train_model_education_loan()
            case "personal_loan":
                train_model_personal_loan()
            case "mutual_fund":
                train_model_mutual_fund()
            case "saving_account":
                train_model_saving_account()
            case "fixed_deposit":
                train_model_fixed_deposit()
            case "recurring_deposit":
                train_model_recurring_deposit()


def deploy_models(deployments):
    for deployment in deployments:
        match deployment:
            case "housing_loan":
                deploy_model_housing_loan()
            case "education_loan":
                deploy_model_education_loan()
            case "personal_loan":
                deploy_model_personal_loan()
            case "mutual_fund":
                deploy_model_mutual_fund()
            case "saving_account":
                deploy_model_saving_account()
            case "fixed_deposit":
                deploy_model_fixed_deposit()
            case "recurring_deposit":
                deploy_model_recurring_deposit()
