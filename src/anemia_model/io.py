from pathlib import Path
import pandas as pd
import joblib, yaml

def load_config(path="config/settings.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_csv(path):
    return pd.read_csv(path)

def save_csv(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def save_joblib(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

def load_joblib(path):
    return joblib.load(path)
