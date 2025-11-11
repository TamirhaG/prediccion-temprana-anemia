# app.py — Interfaz Streamlit para el sistema predictivo de anemia
import streamlit as st
import pandas as pd
import joblib
import json
import os
from src import config

# ===== CONFIGURACIÓN GENERAL =====
st.set_page_config(
    page_title="Predicción Temprana de Anemia — Perú 2025",
    page_icon="🩸",
    layout="wide"
)

st.title("🩺 Sistema Predictivo de Anemia — Perú 2025")
st.markdown("**Proyecto IDL3 — Universidad Continental / Equipo TamirhaG**")

# ===== CARGA DE MODELOS Y MAPEOS =====

# Detectar entorno: local (Codespaces) o remoto (Streamlit Cloud)
base_dir = os.getcwd()

# Buscar label_mapping.json (prioriza raíz si está desplegado en la nube)
if os.path.exists(os.path.join(base_dir, "label_mapping.json")):
    map_path = os.path.join(base_dir, "label_mapping.json")
else:
    map_path = os.path.join(config.ARTIFACTS_DIR, "label_mapping.json")

# Buscar modelos entrenados (prioriza raíz si está en la nube)
model_paths = {
    "RandomForest": os.path.join(base_dir, "model_RandomForest.joblib")
        if os.path.exists(os.path.join(base_dir, "model_RandomForest.joblib"))
        else os.path.join(config.ARTIFACTS_DIR, "model_RandomForest.joblib"),
    "XGBoost": os.path.join(base_dir, "model_XGBoost.joblib")
        if os.path.exists(os.path.join(base_dir, "model_XGBoost.joblib"))
        else os.path.join(config.ARTIFACTS_DIR, "model_XGBoost.joblib")
}

# Cargar mapeo y modelos
with open(map_path, "r", encoding="utf-8") as f:
    label_map = json.load(f)
inv_label_map = {v: k for k, v in label_map.items()}

models = {}
for name, path in model_paths.items():
    if os.path.exists(path):
        models[name] = joblib.load(path)
    else:
        st.warning(f"⚠️ No se encontró el modelo {name}. Verifica que el archivo esté en el repositorio.")


# Intentar primero ruta local (para Streamlit Cloud)
if os.path.exists("label_mapping.json"):
    map_path = "label_mapping.json"
else:
    map_path = os.path.join(config.ARTIFACTS_DIR, "label_mapping.json")


with open(map_path, "r", encoding="utf-8") as f:
    label_map = json.load(f)
inv_label_map = {v: k for k, v in label_map.items()}

models = {
    name: joblib.load(path) for name, path in model_paths.items() if os.path.exists(path)
}

st.sidebar.header("Selecciona el Modelo")
selected_model_name = st.sidebar.selectbox("Modelo", list(models.keys()))
model = models[selected_model_name]

st.sidebar.markdown("---")
st.sidebar.markdown("**Visualizaciones**")
if st.sidebar.button("Ver métricas y gráficos"):
    metrics_path = os.path.join(config.ARTIFACTS_DIR, "metrics_report.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    st.json(metrics)
    st.image(os.path.join(config.OUTPUT_DIR, "metrics_summary.png"))
    st.image(os.path.join(config.OUTPUT_DIR, f"cm_{selected_model_name}.png"))
    st.image(os.path.join(config.OUTPUT_DIR, f"roc_{selected_model_name}.png"))

# ===== OPCIÓN 1: CARGAR CSV =====
st.header("Cargar datos desde un archivo CSV")
uploaded_file = st.file_uploader("Selecciona un archivo CSV con los datos del paciente", type=["csv"])

if uploaded_file:
    df_input = pd.read_csv(uploaded_file)
    st.dataframe(df_input.head())

    if st.button("Predecir desde CSV"):
        predictions = model.predict(df_input)
        decoded = [inv_label_map[int(p)] for p in predictions]
        df_input["Predicción"] = decoded
        st.success("Predicciones generadas correctamente")
        st.dataframe(df_input)
        st.download_button(
            label="Descargar resultados CSV",
            data=df_input.to_csv(index=False).encode("utf-8"),
            file_name="predicciones_anemia.csv",
            mime="text/csv"
        )

# ===== OPCIÓN 2: FORMULARIO MANUAL =====
st.header("Ingresar datos manualmente")

col1, col2, col3 = st.columns(3)
with col1:
    edad = st.number_input("Edad (meses)", min_value=0, max_value=120, value=24)
    peso = st.number_input("Peso (kg)", min_value=1.0, max_value=30.0, value=10.0)
    talla = st.number_input("Talla (cm)", min_value=30.0, max_value=120.0, value=80.0)
with col2:
    hemoglobina = st.number_input("Hemoglobina (g/dL)", min_value=5.0, max_value=18.0, value=11.0)
    hierro = st.number_input("Hierro sérico (µg/dL)", min_value=10.0, max_value=200.0, value=70.0)
    ferritina = st.number_input("Ferritina (ng/mL)", min_value=1.0, max_value=300.0, value=25.0)
with col3:
    vitA = st.number_input("Vitamina A (µg/dL)", min_value=0.0, max_value=100.0, value=40.0)
    proteina = st.number_input("Proteína total (g/dL)", min_value=3.0, max_value=9.0, value=6.0)
    zinc = st.number_input("Zinc (µg/dL)", min_value=10.0, max_value=200.0, value=90.0)

if st.button("Predecir manualmente"):
    data = pd.DataFrame([{
        "Edad": edad,
        "Peso": peso,
        "Talla": talla,
        "Hemoglobina": hemoglobina,
        "Hierro": hierro,
        "Ferritina": ferritina,
        "Vitamina_A": vitA,
        "Proteina": proteina,
        "Zinc": zinc
    }])
    prediction = model.predict(data)[0]
    decoded = inv_label_map[int(prediction)]

    st.success(f"Predicción del modelo **{selected_model_name}**: **{decoded}**")
    st.balloons()
