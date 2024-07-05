import glob
import os
import pickle
import time

from colorama import Fore, Style

from fraud_detection.config import LOCAL_REGISTRY_PATH
from fraud_detection.ml.model import FraudDetectionClassifier


def save_results(params: dict, metrics: dict) -> None:
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save params locally
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", timestamp + ".pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", timestamp + ".pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Results saved locally")


def save_model(model: FraudDetectionClassifier = None) -> None:
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.sav")
    with open(model_path, 'wb') as file:  
        pickle.dump(model, file)

    print("✅ Model saved locally")

    return None


def load_model() -> FraudDetectionClassifier | None:

    print(Fore.BLUE + "\n⏳ Load latest model from local registry..." + Style.RESET_ALL)

    # Get the latest model version name by the timestamp on disk
    local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
    local_model_paths = glob.glob(f"{local_model_directory}/*")

    if not local_model_paths:
        return None

    most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

    print(Fore.BLUE + "\n⏳ Load latest model from disk..." + Style.RESET_ALL)

    with open(most_recent_model_path_on_disk, 'rb') as model_filename:
        latest_model = pickle.load(model_filename)

    if not latest_model:
        return None
    
    print("✅ Model loaded from local disk")
    return latest_model
