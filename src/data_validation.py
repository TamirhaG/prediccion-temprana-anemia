import pandas as pd
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
from src import config

def validate_dataset(path=config.DATASET_PATH):
    print("=== BLOQUE 1: VALIDACIÓN DE DATOS ===")
    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")

    report = {
        "n_filas": df.shape[0],
        "n_columnas": df.shape[1],
        "columnas": df.columns.tolist(),
        "nulos": df.isnull().sum().to_dict(),
        "tipos": df.dtypes.astype(str).to_dict(),
        "duplicados": int(df.duplicated().sum())
    }

    # Distribución del target
    if "Anemia" in df.columns:
        dist = df["Anemia"].value_counts(normalize=True).round(3) * 100
        report["distribucion_anemia"] = dist.to_dict()
        print("\nDistribución de anemia (%):")
        print(dist)

    # Guardar reporte
    out_path = os.path.join(config.ARTIFACTS_DIR, "validation_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Reporte guardado en {out_path}")
    return df

def plot_eda(df):
    print("=== BLOQUE 1B: ANÁLISIS EXPLORATORIO ===")
    num_cols = ["Edad_meses", "Peso_kg", "Talla_cm", "Hemoglobina_g_dL"]
    eda_dir = os.path.join(config.OUTPUT_DIR, "eda")
    os.makedirs(eda_dir, exist_ok=True)

    for col in num_cols:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], bins=30, kde=True, color="skyblue")
        plt.title(f"Distribución de {col}")
        plt.tight_layout()
        path = os.path.join(eda_dir, f"hist_{col}.png")
        plt.savefig(path)
        plt.close()
    print(f"Gráficos guardados en {eda_dir}")
