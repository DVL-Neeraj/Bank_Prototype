import time

from google.cloud.aiplatform.prediction import LocalModel
from handler import PredictionHandler
from predictor import SklearnPredictor

local_model = LocalModel.build_cpr_model(
    src_dir="./",
    output_image_uri=f"asia-northeast1-docker.pkg.dev/genai-414108/bank-prototype/cpr",
    predictor=SklearnPredictor,
    handler=PredictionHandler,
    requirements_path="./requirements.txt",
)

time.sleep(20)

with local_model.deploy_to_local_endpoint(
    artifact_uri="../artifact/"
) as local_endpoint:
    health_check_response = local_endpoint.run_health_check()
    predict_response = local_endpoint.predict(
        request_file="../test_request/housing_loan.json",
        headers={"Content-Type": "application/json"},
    )

print(predict_response.content)

# local_model.push_image()
