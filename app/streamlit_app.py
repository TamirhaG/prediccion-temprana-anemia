import streamlit as st
import pandas as pd
import joblib
from supabase import create_client
import datetime

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Predicción Temprana de Anemia - MIDIS 2025", layout="wide")
st.title("🩸 Sistema Predictivo de Riesgo de Anemia")
st.write("Prototipo IA - MIDIS 2025 | Autor: **Ramon Giraldo**")

# --- CARGAR MODELO ---
MODEL_PATH = "artifacts/model_xgboost_calibrated.joblib"
model = joblib.load(MODEL_PATH)

# --- CONEXIÓN SUPABASE ---
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- FORMULARIO DE ENTRADA ---
with st.form("pred_form"):
    st.subheader("Ingrese los datos del beneficiario:")

    edad = st.slider("Edad (meses)", 6, 60, 24)
    peso = st.number_input("Peso (kg)", min_value=4.0, max_value=25.0, value=12.0)
    talla = st.number_input("Talla (cm)", min_value=50.0, max_value=120.0, value=85.0)
    altitud = st.number_input("Altitud del domicilio (m.s.n.m.)", min_value=0, max_value=4500, value=2500)
    sexo = st.radio("Sexo", ["M", "F"])
    area = st.selectbox("Área de residencia", ["Urbana", "Rural"])

    submitted = st.form_submit_button("Predecir riesgo")

if submitted:
    # --- Preparar entrada ---
    X_input = pd.DataFrame([{
        "Edad_meses": edad,
        "Peso_kg": peso,
        "Talla_cm": talla,
        "Altitud_m": altitud,
        "Sexo_M": 1 if sexo == "M" else 0,
        "Area_Rural": 1 if area == "Rural" else 0
    }])

    # --- Predicción ---
    prob = model.predict_proba(X_input)[0, 1]
    riesgo = "ALTO" if prob >= 0.7 else "MEDIO" if prob >= 0.5 else "BAJO"

    # --- Mostrar resultados ---
    st.metric("Probabilidad estimada de anemia", f"{prob*100:.2f}%")
    st.subheader(f"🩺 Riesgo detectado: {riesgo}")
    st.caption(f"Modelo calibrado XGBoost - actualizado al 28/10/2025")

    # --- Guardar en Supabase ---
    record = {
        "fecha": datetime.datetime.now().isoformat(),
        "edad_meses": edad,
        "peso_kg": peso,
        "talla_cm": talla,
        "altitud_m": altitud,
        "sexo": sexo,
        "area": area,
        "prob_anemia": float(prob),
        "riesgo": riesgo
    }

    try:
        supabase.table("predicciones_anemia").insert(record).execute()
        st.success("Predicción registrada correctamente en Supabase ✅")
    except Exception as e:
        st.warning(f"No se pudo guardar en Supabase: {e}")
