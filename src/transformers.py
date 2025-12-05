# src/transformers.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
class OutlierFlagger(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_cols=None, threshold=3.0):
        self.numeric_cols = numeric_cols
        self.threshold = threshold
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_ = X.copy()
        cols = self.numeric_cols if self.numeric_cols is not None else X_.select_dtypes(include=np.number).columns
        for col in cols:
            if col in X_.columns:
                std = X_[col].std()
                if pd.isna(std) or std == 0:
                    X_[f"{col}_outlier"] = 0
                else:
                    X_[f"{col}_outlier"] = ((X_[col] - X_[col].mean()).abs() > self.threshold * std).astype(int)
        return X_

class PricePerSqftAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        if "Price_in_Lakhs" in X_.columns and "Size_in_SqFt" in X_.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                X_["Price_per_SqFt"] = np.where(
                    (X_["Size_in_SqFt"] > 0),
                    (X_["Price_in_Lakhs"] * 100000) / X_["Size_in_SqFt"],
                    np.nan
                )
        return X_

class AgeCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, current_year=2025):
        self.current_year = current_year

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        if "Year_Built" in X_.columns:
            with np.errstate(invalid="ignore"):
                X_["Age_of_Property"] = np.where(
                    pd.notna(X_["Year_Built"]),
                    self.current_year - X_["Year_Built"],
                    np.nan
                )
        return X_
