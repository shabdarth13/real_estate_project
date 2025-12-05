# src/tune_optuna.py
import optuna
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBRegressor
from preprocessing import build_preprocessing
from config import DATA_PATH, SEED
from train import create_targets

df = pd.read_csv(DATA_PATH)
df = create_targets(df)
X = df.drop(columns=["Good_Investment", "Future_Price_5yrs"])
y = df["Future_Price_5yrs"]

preproc, *_ = build_preprocessing(df)
preproc.fit(X, df["Good_Investment"])
X_trans = preproc.transform(X)

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "random_state": SEED,
        "verbosity": 0
    }
    model = XGBRegressor(**params)
    scores = -cross_val_score(model, X_trans, y, scoring="neg_root_mean_squared_error", cv=kf, n_jobs=-1)
    return float(np.mean(scores))

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("Best trial:", study.best_trial.params)
joblib.dump(study, "models/optuna_study.pkl")
