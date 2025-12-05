import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import os
import joblib

DATA_PATH = "data/india_housing_prices.csv"
MLFLOW_TRACKING_URI = "mlruns"  
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("real_estate_investment_model")

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def apply_labeling_option_B(df):
    df = df.copy()

    if "City" not in df.columns or "Price_per_SqFt" not in df.columns:
        raise ValueError("Required columns missing: City, Price_per_SqFt")

    city_medians = df.groupby("City")["Price_per_SqFt"].median()
    df["City_Median"] = df["City"].map(city_medians)

    df["Good_Investment"] = (df["Price_per_SqFt"] < (df["City_Median"] * 0.90)).astype(int)

    print("Label counts:\n", df["Good_Investment"].value_counts())

    if df["Good_Investment"].nunique() < 2:
        print("WARNING: Still only one class after Option B labeling.")
    return df
def encode_data(df):
    df = df.copy()
    label_encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df, label_encoders
def train_model(df):
    if "Good_Investment" not in df.columns:
        raise ValueError("Label Good_Investment missing from DF")

    X = df.drop(["Good_Investment"], axis=1)
    y = df["Good_Investment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)

    with mlflow.start_run():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        print("Accuracy:", acc)
        print(classification_report(y_test, preds))

        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "rf_investment_model")

    return model

def train_all():
    df = load_data()
    df = apply_labeling_option_B(df)

    if df["Good_Investment"].nunique() < 2:
        raise ValueError("Labeling still producing only 1 class. Fix dataset distribution.")

    df_encoded, _ = encode_data(df)
    clf = train_model(df_encoded)
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/rf_investment_model.joblib")
    print("Model saved at models/rf_investment_model.joblib")
if __name__ == "__main__":
    train_all()
