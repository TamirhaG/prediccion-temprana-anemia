import streamlit as st
import pandas as pd
import joblib
import json
from supabase import create_client
import os

st.set_page_config(page_title="Predicción de Riesgo de Anemia - MIDIS", page_icon="🩸", layout="wide")

try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

model = joblib.load("artifacts/model_xgboost_calibrated.joblib")
metadata = json.load(open("artifacts/metadata.json"))
expected = metadata["feature_names"]

st.title("🩸 Predicción de Riesgo de Anemia Infantil - MIDIS 2025")

col1, col2 = st.columns(2)

with col1:
    edad = st.number_input("Edad (meses)", 0, 60, 24)
    peso = st.number_input("Peso (kg)", 5.0, 30.0, 12.0)
    talla = st.number_input("Talla (cm)", 50.0, 120.0, 85.0)

with col2:
    sexo = st.selectbox("Sexo", ["M", "F"])
    altitud = st.number_input("Altitud (m)", 0, 4500, 2500)
    ingreso = st.number_input("Ingreso familiar (S/)", 0, 5000, 1200)

if st.button("Evaluar Riesgo de Anemia"):
    df = pd.DataFrame([{
        "Edad_meses": edad,
        "Peso_kg": peso,
        "Talla_cm": talla,
        "Altitud_m": altitud,
        "Ingreso_Familiar_Soles": ingreso,
        "Sexo_M": 1 if sexo == "M" else 0,
        "Sexo_F": 1 if sexo == "F" else 0,
    }])

    for col in expected:
        if col not in df.columns:
            df[col] = 0

    df = df[expected]

    prob = model.predict_proba(df)[0][1]
    riesgo = "ALTO" if prob >= 0.70 else "MEDIO" if prob >= 0.50 else "BAJO"

    st.metric("Probabilidad estimada de anemia", f"{prob*100:.2f}%")
    st.metric("Clasificación de riesgo", riesgo)

    supabase.table("predicciones").insert({
        "edad_meses": edad,
        "peso": peso,
        "talla": talla,
        "sexo": sexo,
        "altitud": altitud,
        "ingreso": ingreso,
        "probabilidad": float(prob),
        "riesgo": riesgo
    }).execute()

