# src/modeling.py

import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys
import numpy as np
import geopandas as gpd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    logging.info("Loading data...")
    yield_df = pd.read_csv("data/raw/usda_yield_2013_2023.csv")
    features_df = pd.read_csv("data/processed/climate_features_2013_2023.csv")

    # Average yield across counties per year (since climate is statewide for now)
    yield_df_avg = yield_df.groupby("year")["yield_bu_per_acre"].mean().reset_index()
    logging.info(f"Available years in yield data: {yield_df_avg['year'].tolist()}")
    logging.info(f"Available years in climate data: {features_df['year'].tolist()}")

    # Merge on year
    df = pd.merge(features_df, yield_df_avg, on="year")
    logging.info(f"Loaded {len(df)} records for modeling.")
    logging.info("\nData summary:")
    logging.info(df.describe())
    return df, yield_df

def train_model(df):
    X = df[["gdd_sum", "precip_sum", "temp_mean", "temp_max", "temp_min"]]
    y = df["yield_bu_per_acre"]
    
    logging.info("\nFeature ranges:")
    for col in X.columns:
        logging.info(f"{col}: {X[col].min():.2f} to {X[col].max():.2f}")
    
    logging.info(f"\nTarget range: {y.min():.2f} to {y.max():.2f}")

    # Since we have very few samples, use all data for training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Get feature importances
    importances = model.feature_importances_
    
    # Calculate predictions and metrics on full dataset
    predictions = model.predict(X)
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)

    logging.info(f"\nModel performance on full dataset:")
    logging.info(f"MAE: {mae:.2f}")
    logging.info(f"R²: {r2:.2f}")

    return model, X, y, predictions, X.columns.tolist(), importances, mae, r2

def plot_feature_importance(columns, importances, output_path):
    plt.figure(figsize=(10, 6))
    sorted_idx = np.argsort(importances)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.barh(pos, importances[sorted_idx])
    plt.yticks(pos, np.array(columns)[sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close('all')
    logging.info(f"Saved feature importance plot to {output_path}")

def plot_actual_vs_predicted(y, predictions, output_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(y, predictions, color='blue')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual Yield')
    plt.ylabel('Predicted Yield')
    plt.title('Actual vs. Predicted Yield')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close('all')
    logging.info(f"Saved actual vs. predicted plot to {output_path}")

def plot_residuals(y, predictions, output_path):
    residuals = y - predictions
    plt.figure(figsize=(8, 6))
    plt.scatter(y, residuals, color='purple')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Actual Yield')
    plt.ylabel('Residual (Actual - Predicted)')
    plt.title('Residuals Plot')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close('all')
    logging.info(f"Saved residuals plot to {output_path}")

def plot_iowa_yield_map(yield_df, year, county_geojson_path, output_path):
    # Load county boundaries
    counties_gdf = gpd.read_file(county_geojson_path)
    # Merge yield data for the selected year
    year_yield = yield_df[yield_df['year'] == year].copy()
    year_yield['county_fips'] = year_yield['county_fips'].astype(str).str.zfill(3)
    counties_gdf['COUNTYFP'] = counties_gdf['COUNTYFP'].astype(str).str.zfill(3)
    merged = counties_gdf.merge(year_yield, left_on='COUNTYFP', right_on='county_fips', how='left')
    # Plot
    plt.figure(figsize=(12, 8))
    merged.plot(column='yield_bu_per_acre', cmap='YlGn', legend=True, edgecolor='black', linewidth=0.5)
    plt.title(f'Iowa Corn Yield by County, {year}')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close('all')
    logging.info(f"Saved Iowa yield map to {output_path}")

def save_model(model, path):
    joblib.dump(model, path)
    logging.info(f"Saved trained model to {path}")

def write_metrics(mae, r2, path):
    with open(path, "w") as f:
        f.write(f"MAE: {mae:.2f}\n")
        f.write(f"R² Score: {r2:.2f}\n")
    logging.info(f"Saved model metrics to {path}")

if __name__ == "__main__":
    try:
        os.makedirs("models", exist_ok=True)
        os.makedirs("results", exist_ok=True)

        df, yield_df = load_data()
        model, X, y, predictions, columns, importances, mae, r2 = train_model(df)

        save_model(model, "models/trained_rf_model.pkl")
        plot_feature_importance(columns, importances, "results/feature_importance.png")
        plot_actual_vs_predicted(y, predictions, "results/actual_vs_predicted.png")
        plot_residuals(y, predictions, "results/residuals.png")
        write_metrics(mae, r2, "results/model_metrics.txt")

        # Map for latest year
        latest_year = yield_df['year'].max()
        plot_iowa_yield_map(yield_df, latest_year, "data/raw/iowa_counties_2020.geojson", "results/iowa_yield_map.png")

        logging.info("\nFeature importances:")
        for col, imp in zip(columns, importances):
            logging.info(f"  {col}: {imp:.4f}")
        logging.info("\nModeling pipeline completed successfully.")
        
        plt.close('all')
        logging.shutdown()
        sys.exit(0)
    except Exception as e:
        logging.error(f"Modeling pipeline failed: {str(e)}")
        plt.close('all')
        logging.shutdown()
        sys.exit(1)