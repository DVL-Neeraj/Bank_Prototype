from data_labeling.label import label_products
from data_modeling.model import deploy_models, train_models
from data_simulation.simulate import simulate

# products = ["housing_loan","education_loan","personal_loan","mutual_fund","saving_account","fixed_deposit","recurring_deposit"]
labels = [
    "housing_loan"
]  # ,"education_loan","personal_loan","mutual_fund","saving_account","fixed_deposit","recurring_deposit"]
models = labels
deployments = labels


def main():
    # Simulate the base dataset that will be used to train the models
    simulate(number_of_customers=5_000_000, new_base_data=True)
    print("data simulated")

    # Assign a label of "{Product}" or "None" to each record in the dataset, and save it as a new labeled dataset.
    label_products(labels)
    print("products labeled")

    # Train a model on the features and label in the labeled dataset
    train_models(models)
    print("models trained")

    # Upload models upload to a GCP endpoint.
    deploy_models(deployments)
    print("models deployed")


if __name__ == "__main__":
    main()
