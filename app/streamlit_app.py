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
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except Exception:
    # Si estamos en entorno local o Colab, leer desde variables de entorno
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

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
    "edad_meses": edad_meses,
    "peso_kg": peso,
    "talla_cm": talla,
    "altitud_m": altitud,
    "sexo": sexo,   # M / F, sin convertir a 1/0
    "ingreso_familiar_soles": ingreso
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

# =========================================================
# BLOQUE DE DASHBOARD ANALÍTICO — MIDIS 2025
# =========================================================
st.markdown("---")
st.header("Análisis de Predicciones en Tiempo Real")

try:
    # ---------------------------------------------------------
    # 1. Cargar datos históricos y predicciones nuevas
    # ---------------------------------------------------------
    df_hist = pd.DataFrame(supabase.table("anemia_riesgo").select("*").execute().data)
    df_pred = pd.DataFrame(supabase.table("predicciones").select("*").execute().data)

    # Normalizar estructura
    df_hist.rename(columns={
        "edad_meses": "edad_meses",
        "peso_kg": "peso",
        "talla_cm": "talla",
        "altitud_m": "altitud",
        "prob_anemia": "probabilidad",
    }, inplace=True)

    df_hist["fuente"] = df_hist.get("fuente", "dataset_inicial")
    df_pred["fuente"] = "streamlit_app"

    # Combinar datasets
    df_all = pd.concat([df_hist, df_pred], ignore_index=True)
    df_all["created_at"] = pd.to_datetime(df_all["created_at"], errors="coerce")
    df_all = df_all.sort_values("created_at", ascending=False)

    if df_all.empty:
        st.info("No hay registros aún en la base de datos. Realiza una predicción para iniciar el historial.")
    else:
        # ---------------------------------------------------------
        # 2. KPIs principales
        # ---------------------------------------------------------
        total_pred = len(df_all)
        alto = (df_all["riesgo"] == "ALTO").sum()
        medio = (df_all["riesgo"] == "MEDIO").sum()
        bajo = (df_all["riesgo"] == "BAJO").sum()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total de registros", f"{total_pred}")
        col2.metric("Casos ALTO riesgo", f"{alto} ({alto/total_pred*100:.1f}%)")
        col3.metric("Casos MEDIO riesgo", f"{medio} ({medio/total_pred*100:.1f}%)")
        col4.metric("Casos BAJO riesgo", f"{bajo} ({bajo/total_pred*100:.1f}%)")

        # ---------------------------------------------------------
        # 3. Distribución general de riesgo
        # ---------------------------------------------------------
        st.subheader("Distribución general de riesgo")
        dist = df_all["riesgo"].value_counts(normalize=True).mul(100).reset_index()
        dist.columns = ["Riesgo", "Porcentaje"]
        st.bar_chart(dist.set_index("Riesgo"))

        # ---------------------------------------------------------
        # 4. Comparativa: Datos históricos vs Nuevas predicciones
        # ---------------------------------------------------------
        st.subheader("Comparativa por fuente de datos")
        comp = df_all.groupby(["fuente", "riesgo"]).size().unstack(fill_value=0)
        st.bar_chart(comp)

        # ---------------------------------------------------------
        # 5. Tendencia temporal de las predicciones
        # ---------------------------------------------------------
        st.subheader("Tendencia temporal (últimos registros)")
        trend = df_all.groupby(df_all["created_at"].dt.date)["riesgo"].value_counts().unstack(fill_value=0)
        st.line_chart(trend)

        # ---------------------------------------------------------
        # 6. Tabla resumen
        # ---------------------------------------------------------
        st.subheader("Últimas 10 predicciones (todas las fuentes)")
        cols = ["created_at", "edad_meses", "peso", "talla", "altitud", "probabilidad", "riesgo", "fuente"]
        cols = [c for c in cols if c in df_all.columns]
        st.dataframe(df_all[cols].head(10))

except Exception as e:
    st.error("No se pudo conectar con Supabase o generar el dashboard.")
    st.exception(e)

