"""
Utility Consumption Prediction Model - Training Script
Trains Electricity, Steam, and Water consumption models from Consumptions.xlsx

FIX APPLIED (2026-07-08):
The previous notebooks replaced "outlier" input values (machine counts) with the
column median using an IQR rule, but did NOT apply the same treatment to the
target column. This meant real high-production days kept their true high
Electricity/Steam/Water values while their machine-count inputs were quietly
swapped for "typical" median counts. The model therefore learned a false
relationship ("typical inputs -> unusually high output"), which shows up as
systematic over-prediction on real inputs. This script removes that step and
trains directly on the real, matched input/output pairs.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
import pickle

FEATURE_COLS = [
    'Knitting - D', 'Knitting - N',
    'Bulk Dye - D', 'Bulk Dye - N',
    'Sample Dye - D', 'Sample Dye - N',
    'Dryers - D', 'Dryers - N',
    'Presetting - D', 'Presetting - N',
    'Chillers - D', 'Chillers - N',
    'AHU - D', 'AHU - N',
    'Compressor - D', 'Compressor - N',
    'Luwa - D', 'Luwa - N'
]

TARGETS = {
    'Electricity (kWh)': 'electricity_pkl.sav',
    'Steam (kg)': 'steam_pkl.sav',
    'Total Water (Cu.m.)': 'water_pkl.sav'
}


def load_clean_data(path='Consumptions.xlsx'):
    df = pd.read_excel(path, sheet_name='Consumptions')

    # Clean placeholder characters and blanks
    df.replace('-', np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Force all model columns to numeric (guards against stray text/formatting)
    for c in FEATURE_COLS + list(TARGETS.keys()):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.fillna(0, inplace=True)

    # Drop exact duplicate rows only (real duplicates, not "unusual" rows)
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    if before != after:
        print(f"Dropped {before - after} exact duplicate rows")

    return df


def train_target(df, target_col, out_file):
    print(f"\n{'='*60}\nTraining model for: {target_col}\n{'='*60}")

    X = df[FEATURE_COLS].copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    dt = DecisionTreeRegressor(max_depth=5, random_state=42)
    etr = ExtraTreesRegressor(n_estimators=100, random_state=42)

    gbr.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    etr.fit(X_train, y_train)

    stacked_model = StackingRegressor(
        estimators=[('et', etr), ('gb', gbr), ('dt', dt)],
        final_estimator=Ridge(alpha=1.0)
    )
    stacked_model.fit(X_train, y_train)

    y_pred = stacked_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test.replace(0, np.nan))) * 100

    cv_scores = cross_val_score(stacked_model, X_train, y_train, cv=5, scoring='r2')

    print(f"Test MAE:  {mae:.2f}")
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test R²:   {r2:.4f}")
    print(f"Test MAPE: {mape:.2f}%")
    print(f"CV R² (mean of 5 folds): {cv_scores.mean():.4f}")

    with open(out_file, 'wb') as f:
        pickle.dump(stacked_model, f)
    print(f"Saved -> {out_file}")

    return {'target': target_col, 'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}


def main():
    df = load_clean_data('Consumptions.xlsx')
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    results = []
    for target_col, out_file in TARGETS.items():
        results.append(train_target(df, target_col, out_file))

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for r in results:
        print(f"{r['target']:<22} R²={r['r2']:.4f}  MAE={r['mae']:.2f}  RMSE={r['rmse']:.2f}  MAPE={r['mape']:.2f}%")


if __name__ == '__main__':
    main()
