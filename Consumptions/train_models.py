"""
Utility Consumption Prediction Model - Training Script
Trains Electricity, Steam, and Water consumption models from Consumptions.xlsx

Usage:
    python train_models.py                # train only (fast)
    python train_models.py --visualize     # train + save all charts to plots/

FIX APPLIED (2026-07-08):
The previous notebooks replaced "outlier" input values (machine counts) with the
column median using an IQR rule, but did NOT apply the same treatment to the
target column. This meant real high-production days kept their true high
Electricity/Steam/Water values while their machine-count inputs were quietly
swapped for "typical" median counts. The model therefore learned a false
relationship ("typical inputs -> unusually high output"), which shows up as
systematic over-prediction on real inputs. This script trains directly on the
real, matched input/output pairs (no outlier replacement).
"""

import argparse
import math
import pickle
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

PLOTS_DIR = 'plots'


def load_clean_data(path='Consumptions.xlsx'):
    df = pd.read_excel(path, sheet_name='Consumptions')

    df.replace('-', np.nan, inplace=True)
    df.fillna(0, inplace=True)

    for c in FEATURE_COLS + list(TARGETS.keys()):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.fillna(0, inplace=True)

    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    if before != after:
        print(f"Dropped {before - after} exact duplicate rows")

    return df


def make_visuals(df, target_col, X_test, y_test, y_pred, feature_importances_df):
    """Save all EDA + evaluation charts for one target as PNG files."""
    import matplotlib
    matplotlib.use('Agg')  # headless-safe: just saves files, no GUI popup needed
    import matplotlib.pyplot as plt
    import seaborn as sns

    name_map = {
        'Electricity (kWh)': 'electricity',
        'Steam (kg)': 'steam',
        'Total Water (Cu.m.)': 'water'
    }
    safe_name = name_map.get(target_col, target_col.split(' ')[0].lower())
    out_dir = os.path.join(PLOTS_DIR, safe_name)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Boxplot of all feature distributions (informational only, no longer used to modify data)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[FEATURE_COLS])
    plt.title(f"Feature Distributions - {target_col}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '01_feature_boxplot.png'))
    plt.close()

    # 2. Correlation heatmap among features
    plt.figure(figsize=(14, 9))
    sns.heatmap(df[FEATURE_COLS].corr(), annot=True, vmin=-1, vmax=1, cmap="coolwarm", fmt='.2f')
    plt.title("Correlation Heatmap - Features")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '02_correlation_heatmap.png'))
    plt.close()

    # 3. Correlation of each feature with the target
    corr_with_target = df[FEATURE_COLS].corrwith(df[target_col]).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    corr_with_target.plot(kind='bar')
    plt.title(f"Feature Correlation with {target_col}")
    plt.ylabel("Correlation Coefficient")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '03_correlation_with_target.png'))
    plt.close()

    # 4. Distribution histograms for each feature
    n_cols = 3
    n_rows = math.ceil(len(FEATURE_COLS) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_COLS):
        sns.histplot(df[col], kde=True, bins=30, ax=axes[i])
        axes[i].set_title(col, fontsize=10)
    for j in range(len(FEATURE_COLS), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '04_feature_distributions.png'))
    plt.close()

    # 5. Scatter plots: each feature vs target
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 5))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_COLS):
        sns.scatterplot(x=df[col], y=df[target_col], ax=axes[i], alpha=0.5)
        axes[i].set_title(f"{col} vs {target_col}", fontsize=9)
    for j in range(len(FEATURE_COLS), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '05_scatter_vs_target.png'))
    plt.close()

    # 6. True vs Predicted (test set)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, label='Predicted vs Actual')
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, color='red', linestyle='--', label='Perfect Prediction')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f"True vs Predicted - {target_col}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '06_true_vs_predicted.png'))
    plt.close()

    # 7. Feature importance (averaged across base models)
    plt.figure(figsize=(10, 6))
    feature_importances_df['Average Importance'].sort_values().plot(kind='barh')
    plt.title(f"Feature Importance - {target_col}")
    plt.xlabel("Average Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '07_feature_importance.png'))
    plt.close()

    print(f"  Saved 7 charts -> {out_dir}/")


def train_target(df, target_col, out_file, visualize=False):
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

    if visualize:
        feature_importances = pd.DataFrame(index=X_train.columns)
        feature_importances['Extra Trees'] = etr.feature_importances_
        feature_importances['Gradient Boosting'] = gbr.feature_importances_
        feature_importances['Decision Tree'] = dt.feature_importances_
        feature_importances['Average Importance'] = feature_importances.mean(axis=1)

        print("  Generating charts...")
        make_visuals(df, target_col, X_test, y_test, y_pred, feature_importances)

    return {'target': target_col, 'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}


def main():
    parser = argparse.ArgumentParser(description='Train utility prediction models')
    parser.add_argument('--visualize', action='store_true',
                         help='Also generate and save EDA/evaluation charts to plots/')
    args = parser.parse_args()

    df = load_clean_data('Consumptions.xlsx')
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    if args.visualize:
        print("Visualization mode ON - charts will be saved under plots/<target>/")

    results = []
    for target_col, out_file in TARGETS.items():
        results.append(train_target(df, target_col, out_file, visualize=args.visualize))

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for r in results:
        print(f"{r['target']:<22} R²={r['r2']:.4f}  MAE={r['mae']:.2f}  RMSE={r['rmse']:.2f}  MAPE={r['mape']:.2f}%")


if __name__ == '__main__':
    main()
