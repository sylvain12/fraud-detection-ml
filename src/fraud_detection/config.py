import os
from pathlib import Path

ROOT_DIR = Path(__file__).absolute().parent.parent.parent

SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "raw_data"

DATA_FILENAME_CSV = "dataset.csv"
DATA_FILENAME_PICKLE = "dataset.pickle"

DATA_PICKLE_PATH = DATA_DIR / DATA_FILENAME_PICKLE
DATA_CSV_PATH = DATA_DIR / DATA_FILENAME_CSV


##################  CONSTANTS  #####################
# LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".ai", "fraud_detection", "data")
# LOCAL_REGISTRY_PATH = os.path.join(
#     os.path.expanduser("~"), ".ai", "fraud_detection", "results"
# )

# COLUMN_NAMES_RAW = [
#     "step",
#     "type",
#     "amount",
#     "nameOrig",
#     "oldbalanceOrg",
#     "newbalanceOrig",
#     "nameDest",
#     "oldbalanceDest",
#     "newbalanceDest",
#     "isFraud",
#     "isFlaggedFraud",
# ]


# ML
MAX_TRIAL = 100
SEED = 42
TEST_SIZE = 0.3
COLUMNS_TO_CLEAN = ["step", "nameOrig", "nameDest", "isFlaggedFraud"]
LOCAL_REGISTRY_PATH = SRC_DIR / "results"
