# src/feature_engineering.py
import numpy as np
import pandas as pd

def compute_investment_score(df):
    df = df.copy()
    signals = {}
    signals['cheap_pps'] = (df['Price_per_SqFt'].max() - df['Price_per_SqFt']) / (df['Price_per_SqFt'].max())
    signals['age_score'] = 1 - (df['Age_of_Property'] / (1 + df['Age_of_Property'].max()))
    signals['amenities'] = df['Amenities'].apply(lambda x: len(str(x).split(",")) if pd.notna(x) else 0) / (1 + df['Amenities'].map(lambda x: len(str(x).split(","))).max())
    signals['public_transport'] = df['Public_Transport_Accessibility'].map({"Low":0, "Medium":0.5, "High":1}).fillna(0.5)
    df['investment_score'] = 0.4 * signals['cheap_pps'] + 0.2 * signals['age_score'] + 0.2 * signals['amenities'] + 0.2 * signals['public_transport']
    df['investment_score'] = df['investment_score'].clip(0,1)
    return df
