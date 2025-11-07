# =========================================================
# STREAMLIT APP — Sistema Predictivo de Riesgo de Anemia
# =========================================================
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from supabase import create_client, Client
from dotenv import load_dotenv
import os

# ---------------------------
# Configuración inicial
# ---------------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

MODEL_PATH = "artifacts/model_xgboost_calibrated.joblib"
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Predicción Temprana de Anemia - MIDIS", layout="wide")

st.title("Sistema Predictivo de Riesgo de Anemia (MIDIS 2025)")
st.markdown("### Proyecto: Detección temprana de riesgo de anemia en beneficiarios de programas de alimentación")

# ---------------------------
# Entrada de datos (formulario)
# ---------------------------
st.subheader("Ingreso de Datos del Niño o Gestante")

col1, col2, col3 = st.columns(3)

with col1:
    edad_meses = st.number_input("Edad (meses)", min_value=0, max_value=60, value=24)
    peso_kg = st.number_input("Peso (kg)", min_value=2.0, max_value=30.0, value=12.5)
    talla_cm = st.number_input("Talla (cm)", min_value=40.0, max_value=120.0, value=85.0)

with col2:
    sexo = st.selectbox("Sexo", ["M", "F"])
    altitud_m = st.number_input("Altitud (m)", min_value=0, max_value=5000, value=2500)
    ingreso_familiar_soles = st.number_input("Ingreso familiar (S/.)", min_value=0, max_value=5000, value=850)

with col3:
    nro_hijos = st.number_input("Número de hijos", min_value=1, max_value=10, value=2)
    area_rural = st.selectbox("Área rural", [True, False])
    suplementacion_hierro = st.selectbox("Recibe suplementación de hierro", [True, False])

# ---------------------------
# Predicción
# ---------------------------
if st.button("Calcular Riesgo de Anemia"):
    input_data = pd.DataFrame([{
        "edad_meses": edad_meses,
        "peso_kg": peso_kg,
        "talla_cm": talla_cm,
        "sexo_F": 1 if sexo == "F" else 0,
        "sexo_M": 1 if sexo == "M" else 0,
        "altitud_m": altitud_m,
        "ingreso_familiar_soles": ingreso_familiar_soles,
        "nro_hijos": nro_hijos,
        "area_rural": area_rural,
        "suplementacion_hierro": suplementacion_hierro
    }])

    # Cálculo
    prob = model.predict_proba(input_data)[0][1]
    riesgo = "Alto" if prob >= 0.5 else "Bajo"

    st.metric("Probabilidad de riesgo", f"{prob:.2%}")
    st.success(f"Riesgo Predicho: **{riesgo.upper()}**")

    # Guardar en Supabase
    data_to_insert = {
        "edad_meses": edad_meses,
        "sexo": sexo,
        "peso_kg": peso_kg,
        "talla_cm": talla_cm,
        "altitud_m": altitud_m,
        "area_rural": area_rural,
        "ingreso_familiar_soles": ingreso_familiar_soles,
        "nro_hijos": nro_hijos,
        "suplementacion_hierro": suplementacion_hierro,
        "prob_riesgo_modelo": round(float(prob), 4),
        "riesgo_predicho": riesgo
    }

    try:
        supabase.table("pacientes_anemia").insert(data_to_insert).execute()
        st.info("Registro guardado en la base de datos Supabase")
    except Exception as e:
        st.error(f"Error al guardar en Supabase: {e}")

# ---------------------------
# Panel inferior: datos históricos
# ---------------------------
st.subheader("Historial de registros almacenados")

try:
    registros = supabase.table("pacientes_anemia").select("*").execute()
    df = pd.DataFrame(registros.data)
    st.dataframe(df.sort_values("fecha_registro", ascending=False).head(10))
except Exception as e:
    st.warning(f"No se pudo cargar la data desde Supabase: {e}")

