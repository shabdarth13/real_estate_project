import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer, SimpleImputer
from category_encoders import TargetEncoder

from src.transformers import OutlierFlagger, PricePerSqftAdder, AgeCalculator
from src.config import SEED

def build_preprocessing(df, target):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    to_exclude = {"ID", target, "Good_Investment", "Future_Price_5yrs"}
    numeric_cols = [c for c in numeric_cols if c not in to_exclude]
    cat_cols = [c for c in cat_cols if c not in to_exclude]

    high_card = [c for c in cat_cols if df[c].nunique() > 20]
    low_card = [c for c in cat_cols if df[c].nunique() <= 20]

    target_encoder = None
    if high_card:
        target_encoder = TargetEncoder(cols=high_card)
        df[high_card] = target_encoder.fit_transform(df[high_card], df[target])

    numeric_pipeline = Pipeline([
        ("imputer", IterativeImputer(random_state=SEED, max_iter=10)),
        ("scaler", StandardScaler())
    ])

    low_card_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("low_cat", low_card_pipeline, low_card)
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    full_pipeline = Pipeline([
        ("outlier_flag", OutlierFlagger(numeric_cols=numeric_cols)),
        ("price_per_sqft", PricePerSqftAdder()),
        ("age", AgeCalculator()),
        ("preprocessor", preprocessor)
    ])

    return full_pipeline, numeric_cols, high_card, low_card, target_encoder, df
