import streamlit as st
from supabase import create_client, Client
import joblib
import pandas as pd
import datetime

# --- Configuración inicial ---
url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(url, key)

model = joblib.load("model/pipeline_xgboost_tuned.pkl")

st.title("🩺 Sistema Predictivo MIDIS - Riesgo de Anemia")

# --- Entrada de datos ---
region = st.selectbox("Región", ["Cusco", "Puno", "Loreto", "Lima"])
edad = st.slider("Edad (meses)", 6, 59, 24)
peso = st.number_input("Peso (kg)", 4.0, 20.0, 10.5)
talla = st.number_input("Talla (cm)", 60, 120, 85)
programas = st.multiselect("Programas MIDIS", ["JUNTOS", "Qali Warma", "Cuna Más"])

# --- Preparar datos y predecir ---
if st.button("Evaluar riesgo"):
    df = pd.DataFrame([{
        "Region": region,
        "Edad_meses": edad,
        "Peso_kg": peso,
        "Talla_cm": talla,
        "Programa_JUNTOS": int("JUNTOS" in programas),
        "Programa_Qali_Warma": int("Qali Warma" in programas),
        "Programa_Cuna_Mas": int("Cuna Más" in programas)
    }])
    prob = model.predict_proba(df)[:,1][0]
    riesgo = "Alto" if prob > 0.6 else "Medio" if prob > 0.4 else "Bajo"
    st.metric("Nivel de riesgo", riesgo, f"{prob*100:.1f}%")

    # --- Guardar en Supabase ---
    data = {
        "fecha": str(datetime.date.today()),
        "region": region,
        "edad_meses": edad,
        "probabilidad": float(prob),
        "riesgo": riesgo
    }
    supabase.table("predicciones_anemia").insert(data).execute()
    st.success("✅ Resultado guardado en Supabase")
