from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from pandas import DataFrame
from sklearn import metrics
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from fraud_detection.config import DATA_PICKLE_PATH, SEED


def save_pickle_dataset(data: DataFrame, path: Path | str = DATA_PICKLE_PATH):
    with open(path, "wb") as _:
        data.to_pickle(path, compression="infer")


def load_pickle_dataset(path: Path | str = DATA_PICKLE_PATH) -> DataFrame:
    with open(path, "rb") as output:
        return pd.read_pickle(output)


def plot_confusion_matrix(y_true: np.ndarray, y_predicted: np.ndarray) -> None:
    sns.heatmap(
        (metrics.confusion_matrix(y_true, y_predicted)),
        annot=True,
        fmt=".5g",
        cmap="Blues",
    )
    plt.xlabel("Predicted")
    plt.ylabel("Acutals", rotation=0)
    plt.title("CONFUSION MATRIX - CUT OFF (0.5)")


def display_model_result(
    y_true: np.ndarray, y_predicted: np.ndarray, model_score: float = None
) -> None:
    if model_score is not None:
        print(f"Model Score: {model_score}")
        print()
    # print(metrics.confusion_matrix(y_true, y_predicted))
    # print()
    print(metrics.classification_report(y_true, y_predicted))


def display_model_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    x_test_true: np.ndarray,
    x_test_pred: np.ndarray,
    model_name: str,
) -> None:
    precision = precision_score(y_true, y_pred) * 100
    recall = recall_score(y_true, y_pred) * 100
    accuracy = accuracy_score(y_true, y_pred) * 100

    precision_test = precision_score(x_test_true, x_test_pred) * 100
    recall_test = recall_score(x_test_true, x_test_pred) * 100
    accuracy_test = accuracy_score(x_test_true, x_test_pred) * 100

    return pd.DataFrame(
        {
            "train": {
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
            },
            "test": {
                "Model": model_name,
                "Accuracy": accuracy_test,
                "Precision": precision_test,
                "Recall": recall_test,
            },
        }
    )


def display_performance(
    model, X: np.ndarray, y: np.ndarray, *args, **kwargs
) -> pd.DataFrame:
    accuracy = []
    recall = []
    roc_auc = []
    precision = []
    f1score = []

    y_pred = model.predict(X)

    accuracy.append(round(accuracy_score(y, y_pred), 4))
    recall.append(round(recall_score(y, y_pred), 4))
    roc_auc.append(round(roc_auc_score(y, y_pred), 4))
    precision.append(round(precision_score(y, y_pred), 4))
    f1score.append(round(f1_score(y, y_pred), 4))

    model_names = []
    model_names.append(kwargs["model_name"])

    return pd.DataFrame(
        {
            "Accuracy": accuracy,
            "Recall": recall,
            "Roc_Auc": roc_auc,
            "Precision": precision,
            "F1 Score": f1score,
        },
        index=model_names,
    )


# ------------------------------------- ML -----------------------------


# Preprocess
def preprocess() -> Pipeline:
    category_transformer = OneHotEncoder(drop="if_binary", sparse_output=False)
    numeric_transformer = make_pipeline(StandardScaler())

    column_transformer = make_column_transformer(
        (category_transformer, make_column_selector(dtype_include=[object, bool])),
        (numeric_transformer, make_column_selector(dtype_exclude=[object, bool])),
        remainder="passthrough",
    )

    return make_pipeline(column_transformer)


# Model creation
def build_model(model, X: np.ndarray, y: np.ndarray, *args, **kwargs) -> Pipeline:
    if kwargs["resampled"]:
        X_train_transformed = preprocess().fit_transform(X)
        smote = SMOTE(random_state=SEED)
        X_resampled, y_resampled = smote.fit_resample(X_train_transformed, y)
        print("#️⃣ Before sampling SMOTE:", Counter(y))
        print("#️⃣ After sampling SMOTE:", Counter(y_resampled))

        pipeline = get_pipeline(model, None)
        pipeline.fit(X_resampled, y_resampled)
    else:
        pipeline = get_pipeline(model, preprocess())
        pipeline.fit(X, y)

    return pipeline


def get_pipeline(model: Any, preproc: Pipeline | None) -> Pipeline:
    return (
        make_pipeline(preproc, model, verbose=True)
        if preproc is not None
        else make_pipeline(model)
    )


def save_model(path: Path, model: XGBClassifier) -> None:
    model.save_model(path)
    print("✅ Model has been saved")


def load_model(path: Path) -> XGBClassifier:
    try:
        model = XGBClassifier()
        model.load_model(path)
    except FileNotFoundError:
        print("😬 Model file not exists!")
    except Exception:
        print("😬 Sorry unable to load the model!")

    print("✅ Model has been loaded")
    return model


def style_fraud(row: pd.Series):
    if row.lower() == "fraud":
        return "background-color: red; color: white;"
    elif row.lower() == "non-fraud":
        return "background-color: green; color: white;"
    else:
        return None


def get_predictions_results(X: pd.DataFrame, predictions: list[int]) -> pd.DataFrame:
    result_df = X.copy()
    result_df["Status"] = predictions

    s = result_df.style.applymap(
        style_fraud,
        subset=["Status"],
    )
    result_df["Status"] = result_df["Status"].apply(
        lambda x: "Fraud" if x == 1 else "Non-Fraud"
    )

    return s, result_df
