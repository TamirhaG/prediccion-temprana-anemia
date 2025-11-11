# src/evaluation.py
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    accuracy_score,
    cohen_kappa_score
)
from src import config


def evaluate_models():
    print("=== BLOQUE 5: EVALUACIÓN DE MODELOS ===")

    # Rutas
    rf_path = os.path.join(config.ARTIFACTS_DIR, "model_RandomForest.joblib")
    xgb_path = os.path.join(config.ARTIFACTS_DIR, "model_XGBoost.joblib")
    data_path = os.path.join(config.OUTPUT_DIR, "featured_dataset.csv")
    map_path = os.path.join(config.ARTIFACTS_DIR, "label_mapping.json")

    # Verificar existencia
    for path in [rf_path, xgb_path, data_path, map_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No se encontró: {path}")

    # Cargar dataset y mapeo de etiquetas
    df = pd.read_csv(data_path)
    X = df.drop(columns=["Anemia"])
    y_true = df["Anemia"]

    with open(map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    # Convertir etiquetas de texto a números según el mapeo
    y_true = y_true.map(label_map)
    class_labels = list(label_map.keys())

    # Cargar modelos entrenados
    rf_model = joblib.load(rf_path)
    xgb_model = joblib.load(xgb_path)

    results = {}

    for name, model in [("RandomForest", rf_model), ("XGBoost", xgb_model)]:
        print(f"\n📊 Evaluando modelo: {name}")

        # Predicciones
        y_pred = model.predict(X)

        # Métricas principales
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        kappa = cohen_kappa_score(y_true, y_pred)

        # ROC / AUC
        try:
            y_prob = model.predict_proba(X)
            auc = roc_auc_score(
                pd.get_dummies(y_true),
                y_prob,
                average="macro",
                multi_class="ovr"
            )
        except Exception:
            auc = np.nan

        gini = (2 * auc - 1) if not np.isnan(auc) else np.nan

        # === Matriz de Confusión ===
        cm = confusion_matrix(y_true, y_pred, labels=range(len(class_labels)))
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels
        )
        plt.title(f"Matriz de Confusión — {name}")
        plt.ylabel("Real")
        plt.xlabel("Predicho")

        cm_path = os.path.join(config.OUTPUT_DIR, f"cm_{name}.png")
        plt.savefig(cm_path, bbox_inches="tight")
        plt.close()

        # === Curva ROC (solo si aplica) ===
        if not np.isnan(auc):
            plt.figure(figsize=(6, 5))
            fpr, tpr, _ = roc_curve(pd.get_dummies(y_true).values.ravel(),
                                    y_prob.ravel())
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("Tasa de Falsos Positivos")
            plt.ylabel("Tasa de Verdaderos Positivos")
            plt.title(f"Curva ROC — {name}")
            plt.legend()
            roc_path = os.path.join(config.OUTPUT_DIR, f"roc_{name}.png")
            plt.savefig(roc_path, bbox_inches="tight")
            plt.close()

        # === Guardar métricas ===
        results[name] = {
            "accuracy": round(acc, 4),
            "f1_macro": round(f1, 4),
            "kappa": round(kappa, 4),
            "auc": round(auc, 4) if not np.isnan(auc) else None,
            "gini": round(gini, 4) if not np.isnan(gini) else None
        }

    # Guardar reporte JSON
    report_path = os.path.join(config.ARTIFACTS_DIR, "metrics_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n✔ Reporte de métricas guardado en: {report_path}")
    return results
