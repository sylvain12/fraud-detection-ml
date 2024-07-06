from collections import Counter
from enum import Enum

import numpy as np
from colorama import Fore, Style
from sklearn import set_config
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from fraud_detection.config import SEED
from fraud_detection.ml.preprocess import preprocess_pipeline
from fraud_detection.utils import get_pipeline

set_config(enable_metadata_routing=True)


def build_model(model, *args, **kwargs) -> Pipeline:
    pipeline = get_pipeline(model, preprocess_pipeline())
    return pipeline


class TransactionType(Enum):
    CASHIN = "CASH_IN"
    CASHOUT = "CASH_OUT"
    PAYMENT = "PAYMENT"
    TRANSFERT = "TRANSFERT"
    DEBIT = "DEBIT"


def transaction_error(type: TransactionType, transaction):
    match type:
        case TransactionType.CASHIN:
            pass

        case TransactionType.CASHOUT:
            pass

        case TransactionType.PAYMENT:
            pass

        case TransactionType.TRANSFERT:
            pass

        case TransactionType.DEBIT:
            pass


class DetectionMode(Enum):
    SOFT = "soft"
    HARD = "hard"


class FraudDetectionClassifier(BaseEstimator, ClassifierMixin):
    threshold = 0.4983
    weights = [0.33, 0.33, 0.33]

    def __init__(self, mode: DetectionMode.SOFT) -> None:
        super().__init__()
        print(
            Fore.BLUE
            + "\n⏳ Initialize new fraud detection classifer model..."
            + Style.RESET_ALL
        )
        _xgb_params = {
            "n_estimators": 250,
            "learning_rate": 0.2657312661875277,
            "max_depth": 4,
            "subsample": 0.9781872389666053,
            "colsample_bytree": 0.9359205162742357,
            "scale_pos_weight": 240,
            "reg_alpha": 0.2,
            "reg_lambda": 1.5,
        }

        _lr_params = {
            "solver": "liblinear",
            "penalty": "l2",
            "C": 100,
            "class_weight": {0: 0.1670854271356784, 1: 0.8329145728643216},
        }

        _rf_params = {
            "n_estimators": 300,
            "max_depth": 70,
            "min_samples_split": 4,
            "min_samples_leaf": 5,
            "max_features": "log2",
            "class_weight": {0: 0.1542211055276382, 1: 0.8457788944723618},
        }

        self.detection_mode = mode

        self.lr_model = build_model(
            LogisticRegression(random_state=SEED, max_iter=1000, **_lr_params)
        )
        self.rf_model = build_model(
            RandomForestClassifier(random_state=SEED, **_rf_params)
        )
        self.xgb_model = build_model(
            XGBClassifier(random_state=SEED, enable_categorical=True, **_xgb_params)
        )
        # self.detection_mode = detection_mode
        self.model = build_model(
            VotingClassifier(
                estimators=[
                    (
                        "lr",
                        LogisticRegression(
                            random_state=SEED, max_iter=1000, **_lr_params
                        ).set_fit_request(sample_weight=True),
                    ),
                    # ("rf", RandomForestClassifier(random_state=SEED, **_rf_params)),
                    (
                        "xgb",
                        XGBClassifier(
                            random_state=SEED, enable_categorical=True, **_xgb_params
                        ),
                    ),
                ],
                voting=self.detection_mode,
            )
        )

    def fit(self, X, y):
        print(Fore.BLUE + "\n⏳ Start training..." + Style.RESET_ALL)
        self.lr_model.fit(X, y)
        self.rf_model.fit(X, y)
        self.xgb_model.fit(X, y)
        return self

    def predict(self, X) -> int:
        # lr_pred = self.lr_model.predict(X)
        # rf_pred = self.rf_model.predict(X)
        # xgb_pred = self.xgb_model.predict(X)

        lr_prob = self.lr_model.predict_proba(X)[:, 1]
        rf_prob = self.rf_model.predict_proba(X)[:, 1]
        xgb_prob = self.xgb_model.predict_proba(X)[:, 1]

        combined_prob = (
            lr_prob * self.weights[0]
            + rf_prob * self.weights[1]
            + xgb_prob * self.weights[2]
        ) / sum(self.weights)

        # combined_pred = np.maximum(xgb_pred, np.maximum(lr_pred, rf_pred))
        combined_pred = (combined_prob >= self.threshold).astype(int)

        return combined_pred

    def _fraud_details(self, X):
        fraud_transactions = []
        y_pred = self.predict(X)
        for pred, transaction in zip(y_pred, X):
            if pred == 1:
                fraud_transactions.append({"type": transaction["type"], "error": None})

    def predict_with_threshold(self, X): ...

    def score(self, X, y) -> float:
        lr_score = self.lr_model.score(X, y)
        rf_score = self.rf_model.score(X, y)
        xgb_score = self.xgb_model.score(X, y)
        combined_score = (
            lr_score * self.weights[0]
            + rf_score * self.weights[1]
            + xgb_score * self.weights[2]
        ) / sum(self.weights)
        return combined_score

    def predict_proba(self, X):
        lr_prob = self.lr_model.predict_proba(X)
        rf_prob = self.rf_model.predict_proba(X)
        xgb_prob = self.xgb_model.predict_proba(X)

        combined_prob = (
            lr_prob * self.weights[0]
            + rf_prob * self.weights[1]
            + xgb_prob * self.weights[2]
        ) / sum(self.weights)
        # combined_prob = (lr_prob + xgb_prob + rf_prob) / 3
        return combined_prob

    def evaluate(self, X, y) -> dict:
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)[:, 1]

        precision, recall, thresholds = precision_recall_curve(y, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall)

        roc_auc = roc_auc_score(y, y_pred_proba)
        pr_auc = auc(recall, precision)
        f1 = f1_score(y, y_pred)

        self.threshold = thresholds[np.argmax(f1_scores)]

        print(f"Best Decision Threshold: {self.threshold}")
        print(f"ROC AUC: {roc_auc}")
        print(f"Precision-Recall AUC: {pr_auc}")
        print(f"F1 Score: {f1}\n")

        return dict(f1=f1, auc=pr_auc, roc_auc=roc_auc, best_threshold=self.threshold)

    @property
    def params() -> dict: ...
