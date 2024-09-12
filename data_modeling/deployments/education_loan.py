# %%
from google.api_core.exceptions import NotFound
from google.cloud import aiplatform, storage

serving_container_image_uri = (
    "asia-northeast1-docker.pkg.dev/genai-414108/bank-prototype/cpr"
)
location = "asia-northeast1"
project = "genai-414108"
service_account = "vertex-ai-endpoint-predict@genai-414108.iam.gserviceaccount.com"
bucket_name = "bank_prototype"

aiplatform.init(
    project=project,
    location=location,
    staging_bucket="gs://{bucket_name}",
)


# %%
def deploy_model_education_loan(
    bucket_name=bucket_name,
    serving_container_image_uri=serving_container_image_uri,
    service_account=service_account,
):
    display_name = "education_loan"
    parameters = {"sampled_shapley_attribution": {"path_count": 25}}
    inputs = {
        "gender": {},
        "age": {},
        "education_level": {},
        "region": {},
        "employer": {},
        "rm_products_sold": {},
        "rm_meetings": {},
        "rm_meetings_rescheduled": {},
        "rm_complaints": {},
        "loan_history": {},
        "event": {},
        "salary": {},
        "net_worth": {},
        "rm_skillset": {},
        "cibil": {},
        "terminates_products_early": {},
        "unfavorable_profession": {},
        "unfavorable_employer": {},
        "kyc_status": {},
        "marital_status": {},
        "conversion_duration": {},
        "interest_rate": {},
    }
    outputs = {f"{display_name.replace('_',' ').title()}": {}}

    parameters = aiplatform.explain.ExplanationParameters(parameters)
    metadata = aiplatform.explain.ExplanationMetadata(inputs=inputs, outputs=outputs)

    local_model_path = f"./data/models/model_{display_name}.pkl"
    blob_name = f"{display_name}/model.pkl"
    artifact_uri = f"gs://{bucket_name}/{display_name}/"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_model_path)
    print("\nFile uploaded\n")

    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_container_image_uri,
        parent_model=display_name,
        explanation_parameters=parameters,
        explanation_metadata=metadata,
    )
    print("\nModel Uploaded\n")

    try:
        endpoint = aiplatform.Endpoint(endpoint_name=display_name.replace("_", "-"))
        endpoint.undeploy_all()
    except NotFound as e:
        endpoint = aiplatform.Endpoint.create(
            display_name=display_name.replace("_", "-")
        )

    model.deploy(
        endpoint=endpoint,
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=1,
        service_account=service_account,
        deploy_request_timeout=1200,
    )
    print("\nModel Deployed\n")
