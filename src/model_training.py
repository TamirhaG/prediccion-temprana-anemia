import os
import joblib
import pandas as pd
import numpy as np
import json  # 🔹 Agregar esta línea
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from src import config


def train_models():
    df = pd.read_csv(os.path.join(config.OUTPUT_DIR, "featured_dataset.csv"))
    X = df.drop(columns=["Anemia"])
    y = df["Anemia"]

    # Codificación de etiquetas para modelos (0,1,2,3)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Guardar el mapeo de etiquetas (convertido a tipos nativos)
    label_map = {cls: int(val) for cls, val in zip(le.classes_, le.transform(le.classes_))}
    map_path = os.path.join(config.ARTIFACTS_DIR, "label_mapping.json")

    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=4, ensure_ascii=False)

    print(f"Mapeo de clases guardado en {map_path}")



    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


    # === BLOQUE 4: ENTRENAMIENTO DE MODELOS ===
    print("=== BLOQUE 4: ENTRENAMIENTO DE MODELOS ===")

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=len(np.unique(y_train)),
        random_state=42
    )

    models = {"RandomForest": rf, "XGBoost": xgb}
    metrics = {}

    for name, model in models.items():
        print(f"Entrenando {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Invertimos las etiquetas para obtener los nombres originales
        y_pred_labels = le.inverse_transform(y_pred)
        y_test_labels = le.inverse_transform(y_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        kappa = cohen_kappa_score(y_test, y_pred)

        metrics[name] = {"accuracy": acc, "f1_macro": f1, "kappa": kappa}

        # Guardar modelo
        model_path = os.path.join(config.ARTIFACTS_DIR, f"model_{name}.joblib")
        joblib.dump(model, model_path)
        print(f"Modelo {name} guardado en {model_path}")

    # Exportar métricas iniciales
    metrics_path = os.path.join(config.ARTIFACTS_DIR, "training_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f"Métricas de entrenamiento guardadas en {metrics_path}")

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
    print("Modelos entrenados y guardados en artifacts/")
    print(metrics)
    return metrics
