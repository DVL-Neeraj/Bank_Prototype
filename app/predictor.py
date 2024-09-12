import os
import pickle

import numpy as np
from google.cloud.aiplatform.constants import prediction
from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud.aiplatform.utils import prediction_utils


class SklearnPredictor(Predictor):
    """Default Predictor implementation for Sklearn models."""

    def __init__(self):
        return

    def load(self, artifacts_uri: str) -> None:
        """Loads the model artifact.

        Args:
            artifacts_uri (str):
                Required. The value of the environment variable AIP_STORAGE_URI.

        Raises:
            ValueError: If there's no required model files provided in the artifacts
                uri.
        """
        prediction_utils.download_model_artifacts(artifacts_uri)
        if os.path.exists(prediction.MODEL_FILENAME_PKL):
            self._model = pickle.load(open(prediction.MODEL_FILENAME_PKL, "rb"))
        else:
            valid_filenames = [
                prediction.MODEL_FILENAME_PKL,
            ]
            raise ValueError(
                f"One of the following model files must be provided: {valid_filenames}."
            )

    def preprocess(self, prediction_input: dict) -> np.ndarray:
        """Converts the request body to a numpy array before prediction.
        Args:
            prediction_input (dict):
                Required. The prediction input that needs to be preprocessed.
        Returns:
            The preprocessed prediction input.
        """
        instances = prediction_input["instances"]
        return np.asarray(instances)

    def predict(self, instances: np.ndarray) -> np.ndarray:
        """Performs prediction.

        Args:
            instances (np.ndarray):
                Required. The instance(s) used for performing prediction.

        Returns:
            Prediction results.
        """
        return self._model.predict_proba(instances)

    def postprocess(self, prediction_results: np.ndarray) -> dict:
        """Converts numpy array to a dict.
        Args:
            prediction_results (np.ndarray):
                Required. The prediction results.
        Returns:
            The postprocessed prediction results.
        """
        for i in range(len(self._model.classes_)):
            if self._model.classes_[i] != "None":
                index = i
        return {
            "predictions": [
                dict([[self._model.classes_[index], np.round(result[index], 3)]])
                for result in prediction_results
            ]
        }
