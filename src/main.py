from typing import NoReturn

import numpy as np
import pandas as pd
from fraud_detection.config import SEED, TEST_SIZE
from fraud_detection.ml.data import clean_data, load_data
from fraud_detection.ml.model import FraudDetectionClassifier
from fraud_detection.ml.preprocess import preprocess_pipeline
from fraud_detection.ml.registry import load_model, save_model, save_results
from sklearn.model_selection import train_test_split


def preprocess() -> tuple:
    print("⭐️ preprocessing...\n")
    data = load_data()
    data = clean_data(data)

    X = data.drop("isFraud", axis=1)
    y = data["isFraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )

    print("🙌 preprocess() done 🎉\n")
    print(f"{'-'*40}\n")

    return X_train, X_test, y_train, y_test


def train(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> NoReturn:
    print("⭐️ training...\n")

    model = load_model()

    if model is None:
        print("⛔️ No model found in local registry\n")
        model = FraudDetectionClassifier()

    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    params = dict(context="train", row_count=len(X_train))

    # Save results on the hard drive using taxifare.ml_logic.registry
    save_results(params=params, metrics=dict(score=score))

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model)

    print("✅ train() done 🎉\n")


def evaluate(X: np.ndarray, y: np.ndarray) -> NoReturn:
    print("⭐️ evaluating...\n")

    model = load_model()
    assert model is not None

    metrics_dict = model.evaluate(X, y)
    params = dict(context="evaluate", row_count=len(X))

    save_results(params=params, metrics=metrics_dict)
    print("🙌 evaluate() done 🎉\n")


def pred(X_pred: pd.DataFrame = None) -> int:
    """
    Make a prediction using the latest trained model
    """
    print("⭐️ Predicting...\n")

    # print("\n⭐️ Use case: predict")

    print("📊 Dataset to predict\n")
    print(X_pred)

    if X_pred is None:
        X_pred = pd.DataFrame(
            dict(
                # step=[1],
                type=["TRANSFER"],
                amount=[100.00],
                oldbalanceOrg=[200.00],
                newbalanceOrig=[100.00],
                oldbalanceDest=[200],
                newbalanceDest=[300],
            )
        )

    model = load_model()
    assert model is not None

    y_pred = model.predict(X_pred)
    print("-" * 50)
    model.predict_proba(X_pred)

    print("\n🙌 predict() done 🎉\n")

    return y_pred


def main() -> None:
    X_train, X_test, y_train, y_test = preprocess()
    # train(X_train, y_train, X_test, y_test)
    # evaluate(X_test, y_test)
    # res = pred()
    # print(res)

    # idxs_fraud = [100, 475, 839, 1478, 2371, 2421, 2771, 4794, 5003, 5129]
    # idxs_non_fraud = [
    #     0,
    #     1,
    #     2,
    #     3,
    #     4,
    #     5,
    #     6,
    #     7,
    #     8,
    #     9,
    #     10,
    #     11,
    #     12,
    #     13,
    #     14,
    #     15,
    #     16,
    #     17,
    #     18,
    #     19,
    # ]

    # X_test_fraud = X_test.iloc[idxs_fraud]
    # X_test_non_fraud = X_test.iloc[idxs_non_fraud]

    # X_val = (
    #     pd.concat([X_test_fraud, X_test_non_fraud])
    #     .sample(frac=1)
    #     .reset_index(drop=True)
    # )

    # X_val.to_csv("test.csv")
    # print(X_val)
    # X_test_fraud.to_csv('fraud')

    # train(X_train, y_train, X_test, y_test)
    # evaluate(X_test, y_test)
    # X_new = X_test[45:50]
    # results = pred(X_val)
    # print(results)

    # print([idx for idx, pred in enumerate(results) if pred == 1][:10])
    # print([idx for idx, pred in enumerate(results) if pred == 0][:20])
    # X_val = X_test.loc[X_test["type"] == "TRANSFER"]
    # pred(X_val)
    pred()


if __name__ == "__main__":
    main()
