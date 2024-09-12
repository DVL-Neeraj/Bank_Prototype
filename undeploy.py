# %%
from google.api_core.exceptions import NotFound
from google.cloud import aiplatform
from sympy import N

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
display_names = [
    "housing-loan",
    "education-loan",
    "personal-loan",
    "mutual-fund",
    "saving-account",
    "fixed-deposit",
    "recurring-deposit",
]

for display_name in display_names:
    try:
        endpoint = aiplatform.Endpoint(endpoint_name=display_name)
        endpoint.undeploy_all()
        # endpoint.delete()
    except NotFound:
        pass
# %%
