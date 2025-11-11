# src/metrics_visualization.py
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src import config

def visualize_metrics():
    print("=== BLOQUE 6: COMPARATIVA VISUAL DE MÉTRICAS ===")

    # Cargar métricas desde el archivo JSON
    metrics_path = os.path.join(config.ARTIFACTS_DIR, "metrics_report.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError("No se encontró el archivo de métricas: metrics_report.json")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Convertir a DataFrame
    df_metrics = pd.DataFrame(metrics).T.reset_index()
    df_metrics.rename(columns={"index": "Modelo"}, inplace=True)

    print("\n📊 Resultados comparativos:")
    print(df_metrics)

    # Gráfico de barras comparativo
    plt.figure(figsize=(9, 5))
    melted = df_metrics.melt(id_vars="Modelo", var_name="Métrica", value_name="Valor")

    sns.barplot(
        data=melted,
        x="Valor",
        y="Métrica",
        hue="Modelo",
        palette="viridis"
    )
    plt.title("Comparativa de Métricas entre Modelos")
    plt.xlabel("Valor")
    plt.ylabel("Métrica")
    plt.legend(title="Modelo", loc="lower right")

    output_path = os.path.join(config.OUTPUT_DIR, "metrics_summary.png")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"\n✔ Gráfico comparativo guardado en: {output_path}")
    return df_metrics
