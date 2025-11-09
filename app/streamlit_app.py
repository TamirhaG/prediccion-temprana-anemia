# ============================================================
# BLOQUE: CARGA DEL MODELO Y METADATOS (sin reentrenar)
# ============================================================
import os
import joblib
import pandas as pd
import numpy as np

ARTIFACTS_DIR = "artifacts"  # ajusta la ruta si es distinta
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model_xgboost_calibrated.joblib")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "metadata.json")

# Cargar modelo aprobado
model = joblib.load(MODEL_PATH)

# Cargar metadata con nombres de columnas esperadas
import json
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

expected_features = metadata.get("feature_names", [])

# ============================================================
# FUNCIÓN DE PREPROCESAMIENTO PARA INFERENCIA
# ============================================================

def prepare_input(raw_input: dict) -> pd.DataFrame:
    """
    Recibe un diccionario con las variables del formulario Streamlit
    y devuelve un DataFrame transformado con las mismas columnas
    que el modelo espera.
    """
    df = pd.DataFrame([raw_input])

    # Ajustes derivados del entrenamiento original
    # ------------------------------------------------------------
    # 1. Variables binarias
    df["Sexo_F"] = (df["Sexo"].str.upper() == "F").astype(int)
    df["Sexo_M"] = (df["Sexo"].str.upper() == "M").astype(int)
    df["Area_Rural"] = (df["Area"].str.upper() == "RURAL").astype(int)
    df["Area_Urbana"] = (df["Area"].str.upper() == "URBANA").astype(int)

    # 2. Feature derivada (ejemplo: Hemoglobina ajustada)
    if "Hemoglobina_g_dL" in df.columns and "Altitud_m" in df.columns:
        df["Hemoglobina_Ajustada"] = df["Hemoglobina_g_dL"] - (df["Altitud_m"] * 0.00003)
    else:
        df["Hemoglobina_Ajustada"] = np.nan

    # 3. Escalado simple de variables numéricas (sin alterar modelo entrenado)
    for col in ["Edad_meses", "Peso_kg", "Talla_cm", "Altitud_m", "Ingreso_Familiar_Soles"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # 4. Alinear columnas con las esperadas por el modelo
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_features]
    return df

# ============================================================
# FUNCIÓN DE PREDICCIÓN
# ============================================================
def predict_risk(input_dict):
    X_input = prepare_input(input_dict)
    prob = model.predict_proba(X_input)[0, 1]
    
    # Clasificación de riesgo (usando umbrales aprobados)
    if prob < 0.33:
        riesgo = "BAJO"
        recomendacion = "Continuar controles regulares."
    elif prob < 0.66:
        riesgo = "MEDIO"
        recomendacion = "Programar control en 30 días."
    else:
        riesgo = "ALTO"
        recomendacion = "Derivar inmediatamente al centro de salud."

    return {
        "probabilidad": round(prob * 100, 2),
        "riesgo": riesgo,
        "recomendacion": recomendacion
    }

