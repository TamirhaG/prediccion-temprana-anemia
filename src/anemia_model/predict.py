import pandas as pd
import numpy as np
from .io import load_joblib

# Usa los artefactos que ya generaste en Colab:
PIPELINE_PATH = "data/artifacts/pipeline_xgb_v4_trained.joblib"
ENCODER_PATH  = "data/artifacts/label_encoder_v4.joblib"

def load_pipeline():
    pipeline = load_joblib(PIPELINE_PATH)
    encoder = load_joblib(ENCODER_PATH)
    return pipeline, encoder

def predict_one(payload: dict):
    pipeline, encoder = load_pipeline()
    df = pd.DataFrame([payload])
    proba = pipeline.predict_proba(df)[0]
    pred_idx = int(np.argmax(proba))
    classes = list(range(len(proba)))  # 0..3
    mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    inv_mapping = {v:k for k,v in mapping.items()}
    pred_label = inv_mapping[pred_idx]
    prob_map = {inv_mapping[i]: float(proba[i]) for i in classes}
    return pred_label, prob_map
