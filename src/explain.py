# src/explain.py
import joblib
import shap
import pandas as pd
from src.config import MODEL_DIR

def load_models():
    preproc = joblib.load(MODEL_DIR / "preprocessor.joblib")
    reg = joblib.load(MODEL_DIR / "regressor.joblib")
    clf = joblib.load(MODEL_DIR / "classifier.joblib")
    return preproc, reg, clf

def get_shap_explanation(X_sample):
    preproc, reg, clf = load_models()
    Xt = preproc.transform(X_sample)
    explainer = shap.Explainer(reg)
    shap_values = explainer(Xt)
    return shap_values, Xt
