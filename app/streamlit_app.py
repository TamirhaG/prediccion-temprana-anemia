import streamlit as st
import joblib
import pandas as pd
import numpy as np
from supabase import create_client

# --- Configuración inicial ---
st.set_page_config(page_title="MIDIS - Predicción de Anemia Infantil", layout="wide")

# Cargar modelo calibrado
model = joblib.load("artifacts/model_xgboost_calibrated.joblib")

# Conexión Supabase
url = st.secrets["supabase_url"]
key = st.secrets["supabase_key"]
supabase = create_client(url, key)

# --- Interfaz ---
st.title("🩸 Sistema Predictivo de Riesgo de Anemia Infantil")
st.write("Predicción temprana basada en inteligencia artificial - MIDIS 2025")

# Entradas del usuario
edad = st.slider("Edad (meses)", 6, 60, 24)
peso = st.number_input("Peso (kg)", 4.0, 25.0, 12.0)
talla = st.number_input("Talla (cm)", 50.0, 120.0, 85.0)
altitud = st.number_input("Altitud del domicilio (m.s.n.m.)", 0, 4500, 2500)
area = st.selectbox("Área", ["Urbana", "Rural"])
sexo = st.radio("Sexo", ["M", "F"])

# Procesamiento
if st.button("Evaluar riesgo"):
    X = pd.DataFrame([{
        "Edad_meses": edad,
        "Peso_kg": peso,
        "Talla_cm": talla,
        "Altitud_m": altitud,
        "Sexo_M": 1 if sexo == "M" else 0,
        "Area_Rural": 1 if area == "Rural" else 0
    }])
    prob = model.predict_proba(X)[0, 1]
    riesgo = "ALTO" if prob > 0.65 else "MEDIO" if prob > 0.4 else "BAJO"

    st.metric("Probabilidad estimada de anemia", f"{prob*100:.2f}%")
    st.subheader(f"🩺 Riesgo detectado: {riesgo}")

    # Guardar en Supabase
    supabase.table("predicciones_anemia").insert({
        "edad_meses": edad,
        "peso_kg": peso,
        "talla_cm": talla,
        "altitud_m": altitud,
        "sexo": sexo,
        "area": area,
        "prob_anemia": float(prob),
        "riesgo": riesgo
    }).execute()

    st.success("Resultado guardado correctamente en la base de datos.")
