import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
from supabase import create_client, Client
import os

# -------------------------------
# Configuración inicial
# -------------------------------
st.set_page_config(page_title="Predicción de Riesgo de Anemia - MIDIS", page_icon="🩸", layout="wide")

# Cargar variables de entorno (asegúrate que existan en Streamlit Cloud o tu .env)
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------------
# Cargar modelo y metadata
# -------------------------------
ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model_xgboost_calibrated.joblib")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "metadata.json")

# Cargar modelo
model = joblib.load(MODEL_PATH)

# Cargar metadata
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

expected_features = metadata["feature_names"]

# -------------------------------
# Interfaz de usuario
# -------------------------------
st.title("Sistema Predictivo de Riesgo de Anemia Infantil - MIDIS 2025")
st.markdown("Este sistema utiliza **Machine Learning** para estimar el riesgo de anemia en niños menores de 5 años beneficiarios de programas sociales.")

col1, col2 = st.columns(2)

with col1:
    edad_meses = st.number_input("Edad (meses)", min_value=0, max_value=60, value=24)
    peso = st.number_input("Peso (kg)", min_value=5.0, max_value=30.0, value=12.0)
    talla = st.number_input("Talla (cm)", min_value=50.0, max_value=120.0, value=85.0)

with col2:
    sexo = st.selectbox("Sexo", ["M", "F"])
    altitud = st.number_input("Altitud (m)", min_value=0, max_value=4500, value=2500)
    ingreso = st.number_input("Ingreso familiar (S/)", min_value=0, max_value=5000, value=1200)

# Botón de predicción
if st.button("Evaluar Riesgo de Anemia"):
    # Crear DataFrame de entrada
    input_data = pd.DataFrame([{
        "Edad_meses": edad_meses,
        "Peso_kg": peso,
        "Talla_cm": talla,
        "Sexo_F": 1 if sexo == "F" else 0,
        "Sexo_M": 1 if sexo == "M" else 0,
        "Altitud_m": altitud,
        "Ingreso_Familiar_Soles": ingreso
    }])

    # Alinear con features esperadas
    for col in expected_features:
        if col not in input_data.columns:
            input_data[col] = 0  # columnas faltantes se rellenan con 0

    input_data = input_data[expected_features]

    # Predicción
    try:
        prob = model.predict_proba(input_data)[0][1]
        riesgo = "ALTO" if prob >= 0.7 else "MEDIO" if prob >= 0.5 else "BAJO"

        st.subheader("Resultado del Modelo:")
        st.metric("Probabilidad estimada de anemia", f"{prob*100:.2f}%")
        st.metric("Clasificación de riesgo", riesgo)

        # Recomendación según riesgo
        if riesgo == "ALTO":
            st.warning("Recomendación: Referir a centro de salud para evaluación inmediata.")
        elif riesgo == "MEDIO":
            st.info("Recomendación: Programar control en 30 días.")
        else:
            st.success("Recomendación: Continuar con control anual rutinario.")

        # Guardar en Supabase
        supabase.table("predicciones").insert({
            "edad_meses": edad_meses,
            "peso": peso,
            "talla": talla,
            "sexo": sexo,
            "altitud": altitud,
            "ingreso": ingreso,
            "probabilidad": float(prob),
            "riesgo": riesgo
        }).execute()

    except Exception as e:
        st.error("Error al procesar la predicción.")
        st.exception(e)
