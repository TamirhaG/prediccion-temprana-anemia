import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from src import config

def preprocess_data():
    print("=== BLOQUE 2: PREPROCESAMIENTO ===")
    df = pd.read_csv(config.DATASET_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df.replace({"SÃ­": "Sí", "NÃ³": "No"}, inplace=True)
    df.drop_duplicates(inplace=True)

    num_cols = ["Edad_meses", "Altitud_m", "Ingreso_Familiar_Soles",
                "Nro_Hijos", "Peso_kg", "Talla_cm",
                "Hemoglobina_g_dL", "Hemoglobina_Ajustada"]
    cat_cols = [c for c in df.columns if c not in num_cols + ["Anemia", "ID"]]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    data = preprocessor.fit_transform(df)
    feat_names = (
        list(preprocessor.named_transformers_["num"].get_feature_names_out(num_cols)) +
        list(preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols))
    )

    df_clean = pd.DataFrame(data, columns=feat_names)
    df_clean["Anemia"] = df["Anemia"].values

    clean_path = os.path.join(config.OUTPUT_DIR, "featured_dataset.csv")
    df_clean.to_csv(clean_path, index=False, encoding="utf-8-sig")
    print(f"✔ Dataset limpio guardado en {clean_path}")
    return df_clean

def balance_dataset(df):
    print("=== BLOQUE 2C: BALANCEO DE CLASES ===")
    X = df.drop(columns=["Anemia"])
    y = df["Anemia"]
    smt = SMOTETomek(random_state=42)
    X_bal, y_bal = smt.fit_resample(X, y)
    df_bal = pd.DataFrame(X_bal, columns=X.columns)
    df_bal["Anemia"] = y_bal
    out_path = os.path.join(config.OUTPUT_DIR, "featured_dataset_balanced.csv")
    df_bal.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✔ Dataset balanceado guardado en {out_path}")
    return df_bal
