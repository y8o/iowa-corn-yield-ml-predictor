# src/visualization.py

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import os
import numpy as np
import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    features = pd.read_csv("data/processed/climate_features_2013_2023.csv")
    yield_df = pd.read_csv("data/raw/usda_yield_2013_2023.csv")
    yield_avg = yield_df.groupby("year")["yield_bu_per_acre"].mean().reset_index()
    df = pd.merge(features, yield_avg, on="year")
    return df, yield_df

def plot_actual_vs_predicted(model, df, output_path):
    X = df[["gdd_sum", "precip_sum", "temp_mean", "temp_max", "temp_min"]]
    y_true = df["yield_bu_per_acre"]
    y_pred = model.predict(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, color='blue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
    plt.xlabel("Actual Yield")
    plt.ylabel("Predicted Yield")
    plt.title("Actual vs Predicted Corn Yield")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close('all')
    logging.info(f"Saved actual vs predicted plot to {output_path}")

def plot_residuals(model, df, output_path):
    X = df[["gdd_sum", "precip_sum", "temp_mean", "temp_max", "temp_min"]]
    y_true = df["yield_bu_per_acre"]
    y_pred = model.predict(X)
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, residuals, color='purple')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Actual Yield')
    plt.ylabel('Residual (Actual - Predicted)')
    plt.title('Residuals Plot')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close('all')
    logging.info(f"Saved residuals plot to {output_path}")

def plot_feature_importance(model, output_path):
    importances = model.feature_importances_
    columns = ["gdd_sum", "precip_sum", "temp_mean", "temp_max", "temp_min"]
    sorted_idx = np.argsort(importances)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.figure(figsize=(10, 6))
    plt.barh(pos, importances[sorted_idx])
    plt.yticks(pos, np.array(columns)[sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close('all')
    logging.info(f"Saved feature importance plot to {output_path}")

def plot_iowa_yield_map(yield_df, year, county_geojson_path, output_path):
    counties = gpd.read_file(county_geojson_path)
    latest = yield_df[yield_df['year'] == year].copy()
    latest['county_fips'] = latest['county_fips'].astype(str).str.zfill(3)
    counties['COUNTYFP'] = counties['COUNTYFP'].astype(str).str.zfill(3)
    merged = counties.merge(latest, left_on='COUNTYFP', right_on='county_fips', how='left')
    plt.figure(figsize=(12, 8))
    merged.plot(column='yield_bu_per_acre', cmap='YlGn', legend=True, edgecolor='black', linewidth=0.5)
    plt.title(f'Iowa Corn Yield by County, {year}')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close('all')
    logging.info(f"Saved Iowa yield map to {output_path}")
    time.sleep(1)  # Small delay to ensure resources are released

def plot_average_yield_map(yield_df, county_geojson_path, output_path):
    counties = gpd.read_file(county_geojson_path)
    avg_yield = yield_df.groupby('county_fips')['yield_bu_per_acre'].mean().reset_index()
    avg_yield['county_fips'] = avg_yield['county_fips'].astype(str).str.zfill(3)
    counties['COUNTYFP'] = counties['COUNTYFP'].astype(str).str.zfill(3)
    merged = counties.merge(avg_yield, left_on='COUNTYFP', right_on='county_fips', how='left')
    plt.figure(figsize=(12, 8))
    merged.plot(column='yield_bu_per_acre', cmap='YlGn', legend=True, edgecolor='black', linewidth=0.5)
    plt.title('Average Iowa Corn Yield by County')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close('all')
    logging.info(f"Saved average yield map to {output_path}")

def main():
    try:
        os.makedirs("results", exist_ok=True)
        model = joblib.load("models/trained_rf_model.pkl")
        df, yield_df = load_data()
        plot_actual_vs_predicted(model, df, "results/actual_vs_predicted.png")
        plot_residuals(model, df, "results/residuals.png")
        plot_feature_importance(model, "results/feature_importance.png")
        latest_year = yield_df['year'].max()
        plot_iowa_yield_map(yield_df, latest_year, "data/raw/iowa_counties_2020.geojson", "results/iowa_yield_map.png")
        plot_average_yield_map(yield_df, "data/raw/iowa_counties_2020.geojson", "results/average_yield_map.png")
        logging.info("All visualizations generated successfully.")
        plt.close('all')
        logging.shutdown()
        sys.exit(0)
    except Exception as e:
        logging.error(f"Visualization pipeline failed: {str(e)}")
        plt.close('all')
        logging.shutdown()
        sys.exit(1)

if __name__ == "__main__":
    main()