import os

# === Configuración general del proyecto ===
PROJECT_NAME = "prediccion-temprana-anemia"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

# Crear carpetas si no existen
for path in [OUTPUT_DIR, ARTIFACTS_DIR, DOCS_DIR]:
    os.makedirs(path, exist_ok=True)

# Rutas de archivos importantes
DATASET_PATH = os.path.join(DATA_DIR, "dataset_anemia_PERU_2025_UTF8SIG.csv")

print(f"✔ Configuración cargada correctamente:")
print(f"  DATASET_PATH: {DATASET_PATH}")
print(f"  OUTPUT_DIR:   {OUTPUT_DIR}")
print(f"  ARTIFACTS_DIR:{ARTIFACTS_DIR}")
