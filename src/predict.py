import joblib
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import warnings

import numpy as np
import pandas as pd
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
JOBLIB_MODEL_PATH = MODEL_DIR / "rf_investment_model.joblib"  
ENCODERS_PATH = MODEL_DIR / "label_encoders.joblib"          
FEATURES_PATH = MODEL_DIR / "feature_columns.json"           

MLFLOW_RUNS_DIR = PROJECT_ROOT / "mlruns"
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "india_housing_prices.csv" 
def _load_model() -> Any:
   
    if JOBLIB_MODEL_PATH.exists():
        model = joblib.load(JOBLIB_MODEL_PATH)
        return model

    if MLFLOW_RUNS_DIR.exists():
        try:
            for experiment_dir in sorted(MLFLOW_RUNS_DIR.iterdir(), reverse=True):
                if not experiment_dir.is_dir():
                    continue
                for run_dir in sorted(experiment_dir.iterdir(), reverse=True):
                    candidate = run_dir / "artifacts" / "rf_investment_model"
                    if candidate.exists():
                        model_uri = f"runs:/{run_dir.name}/rf_investment_model"
                        try:
                            model = mlflow.sklearn.load_model(model_uri)
                            return model
                        except Exception:
                            try:
                                model = joblib.load(candidate)
                                return model
                            except Exception:
                                continue
        except Exception:
            pass

    raise FileNotFoundError("No model found. Place a joblib model at models/rf_investment_model.joblib "
                            "or ensure MLflow artifacts exist under mlruns/.")

def _load_encoders_and_features() -> Tuple[dict, list]:
  
    encoders = {}
    features = None

    if ENCODERS_PATH.exists():
        try:
            encoders = joblib.load(ENCODERS_PATH)
        except Exception:
            warnings.warn("Failed to load encoders from models/; proceeding without them.")

    if FEATURES_PATH.exists():
        try:
            with open(FEATURES_PATH, "r", encoding="utf-8") as f:
                features = json.load(f)
        except Exception:
            warnings.warn("Failed to load feature column list from models/feature_columns.json")

    return encoders, features

def _safe_build_dataframe(input_dict: Dict[str, Any], feature_columns: list = None, encoders: dict = None) -> pd.DataFrame:
    df = pd.DataFrame([input_dict])
    if ("Price_per_SqFt" not in df.columns or pd.isna(df.loc[0, "Price_per_SqFt"])) and \
       ("Price_in_Lakhs" in df.columns and "Size_in_SqFt" in df.columns):
        try:
            if df.loc[0, "Size_in_SqFt"] and df.loc[0, "Size_in_SqFt"] > 0:
                df["Price_per_SqFt"] = (df["Price_in_Lakhs"] * 100000) / df["Size_in_SqFt"]
            else:
                df["Price_per_SqFt"] = 0.0
        except Exception:
            df["Price_per_SqFt"] = 0.0
    if ("Age_of_Property" not in df.columns or pd.isna(df.loc[0, "Age_of_Property"])) and ("Year_Built" in df.columns):
        try:
            df["Age_of_Property"] = 2025 - df["Year_Built"]
        except Exception:
            df["Age_of_Property"] = 0
    if "City_Median" not in df.columns:
        try:
            df_train = pd.read_csv(TRAIN_DATA_PATH)
            city_median_map = df_train.groupby("City")["Price_per_SqFt"].median().to_dict()
            df["City_Median"] = df["City"].map(city_median_map)
        except Exception:
            df["City_Median"] = 0.0
    if feature_columns:
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0  
        df = df[feature_columns]
    if encoders:
        for col, le in encoders.items():
            if col in df.columns:
                try:
                    vals = df[col].astype(str).fillna("missing")
                    try:
                        df[col] = le.transform(vals)
                    except Exception:
                        existing = list(le.classes_)
                        new_vals = [v for v in vals.unique() if v not in existing]
                        if new_vals:
                            le_classes = list(le.classes_) + new_vals
                            new_le = LabelEncoder()
                            new_le.classes_ = np.array(le_classes, dtype=object)
                            df[col] = new_le.transform(vals)
                        else:
                            df[col] = le.transform(vals)
                except Exception:
                    df[col] = 0
    else:
        for col in df.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            vals = df[col].astype(str).fillna("missing")
            try:
                le.fit(vals)
                df[col] = le.transform(vals)
            except Exception:
                df[col] = 0

    df = df.fillna(0)
    return df

def predict_from_dict(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    model = _load_model()
    encoders, feature_columns = _load_encoders_and_features()
    df = _safe_build_dataframe(input_dict, feature_columns, encoders)

    try:
        X = df
        pred = model.predict(X)
        proba = None
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(X)
                proba = probs.tolist()
            except Exception:
                proba = None
        return {
            "prediction": int(pred[0]) if isinstance(pred, (list, np.ndarray)) else int(pred),
            "probability": proba,
            "input_features": df.to_dict(orient="records")[0]
        }
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")
if __name__ == "__main__":
    example = {
        "ID": 1,
        "State": "Tamil Nadu",
        "City": "Chennai",
        "Locality": "Locality_84",
        "Property_Type": "Apartment",
        "BHK": 1,
        "Size_in_SqFt": 4740,
        "Price_in_Lakhs": 489.76,
        "Price_per_SqFt": 0,
        "Year_Built": 1990,
        "Furnished_Status": "Furnished",
        "Floor_No": 22,
        "Total_Floors": 1,
        "Age_of_Property": 35,
        "Nearby_Schools": 10,
        "Nearby_Hospitals": 3,
        "Public_Transport_Accessibility": "High",
        "Parking_Space": "No",
        "Security": "No",
        "Amenities": "Playground, Gym, Garden, Pool, Clubhouse",
        "Facing": "West",
        "Owner_Type": "Owner",
        "Availability_Status": "Ready_to_Move"
    }
    out = predict_from_dict(example)
    import json
    print(json.dumps(out, indent=2))
