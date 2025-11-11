import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src import config

def train_models():
    print("=== BLOQUE 4: ENTRENAMIENTO DE MODELOS ===")
    df = pd.read_csv(os.path.join(config.OUTPUT_DIR, "featured_dataset.csv"))
    X = df.drop(columns=["Anemia"])
    y = df["Anemia"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    xgb = XGBClassifier(
        objective="multi:softprob", eval_metric="mlogloss", num_class=len(y.unique()), random_state=42
    )

    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    xgb_pred = xgb.predict(X_test)

    metrics = {
        "rf": {
            "accuracy": accuracy_score(y_test, rf_pred),
            "f1_macro": f1_score(y_test, rf_pred, average="macro"),
            "cohen_kappa": cohen_kappa_score(y_test, rf_pred)
        },
        "xgb": {
            "accuracy": accuracy_score(y_test, xgb_pred),
            "f1_macro": f1_score(y_test, xgb_pred, average="macro"),
            "cohen_kappa": cohen_kappa_score(y_test, xgb_pred)
        }
    }

    joblib.dump(rf, os.path.join(config.ARTIFACTS_DIR, "model_rf.joblib"))
    joblib.dump(xgb, os.path.join(config.ARTIFACTS_DIR, "model_xgb.joblib"))
    print("✔ Modelos entrenados y guardados en artifacts/")
    print(metrics)
    return metrics
