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

    # Paths
    rf_path = os.path.join(config.ARTIFACTS_DIR, "model_rf.joblib")
    xgb_path = os.path.join(config.ARTIFACTS_DIR, "model_xgb.joblib")
    data_path = os.path.join(config.OUTPUT_DIR, "featured_dataset.csv")

    # Verificar existencia de archivos
    if not os.path.exists(rf_path) or not os.path.exists(xgb_path):
        raise FileNotFoundError("No se encontraron los modelos entrenados en artifacts/")
    if not os.path.exists(data_path):
        raise FileNotFoundError("No se encontró el dataset limpio en output/")

    # Cargar dataset
    df = pd.read_csv(data_path)
    X = df.drop(columns=["Anemia"])
    y_true = df["Anemia"]

    # Cargar modelos
    rf_model = joblib.load(rf_path)
    xgb_model = joblib.load(xgb_path)

    results = {}

    for name, model in [("RandomForest", rf_model), ("XGBoost", xgb_model)]:
        print(f"\n📊 Evaluando modelo: {name}")

        y_pred = model.predict(X)

        # Métricas
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        kappa = cohen_kappa_score(y_true, y_pred)

        # ROC / AUC (solo si hay probas)
        try:
            y_prob = model.predict_proba(X)
            # Solo calcula AUC si hay 2 clases (binario)
            auc = roc_auc_score(pd.get_dummies(y_true), y_prob, average="macro", multi_class="ovr")
        except Exception:
            auc = np.nan

        gini = (2 * auc - 1) if not np.isnan(auc) else np.nan

        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=np.unique(y_true),
                    yticklabels=np.unique(y_true))
        plt.title(f"Matriz de confusión — {name}")
        plt.ylabel("Real")
        plt.xlabel("Predicho")

        cm_path = os.path.join(config.OUTPUT_DIR, f"cm_{name}.png")
        plt.savefig(cm_path, bbox_inches="tight")
        plt.close()

        # Curva ROC (solo si tiene predict_proba)
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

        # Guardar resultados
        results[name] = {
            "accuracy": acc,
            "f1_macro": f1,
            "kappa": kappa,
            "auc": auc,
            "gini": gini
        }

    # Exportar reporte JSON
    report_path = os.path.join(config.ARTIFACTS_DIR, "metrics_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n✔ Reporte de métricas guardado en: {report_path}")
    return results
