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

def plot_iowa_yield_map(yield_df, county_geojson_path, output_path):
    counties = gpd.read_file(county_geojson_path)
    # Ensure counties_gdf only contains Iowa counties before merging (STATEFP == '19')
    iowa_counties_gdf = counties[counties['STATEFP'] == '19'].copy()
    
    # Explicitly plot data for the year 2023
    year_to_plot = 2023
    # Filter for Iowa counties in the specified year and exclude aggregate data (FIPS '000')
    latest_year_iowa_yield = yield_df[(yield_df['year'] == year_to_plot) & (yield_df['county_fips'] != '000')].copy()
    
    if latest_year_iowa_yield.empty:
        logging.warning(f"No county-level yield data found for the year {year_to_plot}. Skipping latest year map.")
        # Attempt to find the maximum year with *any* data in yield_df for logging purposes
        max_year_in_data = yield_df['year'].max() if not yield_df.empty else 'N/A'
        logging.warning(f"Maximum year found in yield_df is {max_year_in_data}.")
        return
        
    # Clean and format fips codes for merging
    latest_year_iowa_yield['county_fips'] = latest_year_iowa_yield['county_fips'].astype(str).str.zfill(3)
    iowa_counties_gdf['COUNTYFP'] = iowa_counties_gdf['COUNTYFP'].astype(str).str.zfill(3)

    # Merge yield data with county boundaries
    merged = iowa_counties_gdf.merge(latest_year_iowa_yield, left_on='COUNTYFP', right_on='county_fips', how='left')
    
    # Handle potential missing data after merge (counties with no data for the year)
    # These will appear with the 'nan_fill_color' if using a library like Folium, 
    # but with matplotlib they might just be blank. Ensure we have a column to plot.
    if 'yield_bu_per_acre' not in merged.columns:
         logging.error("Yield column not found after merging for latest year map.")
         return # Or handle appropriately

    plt.figure(figsize=(12, 8))
    # Use the merged GeoDataFrame for plotting
    merged.plot(column='yield_bu_per_acre', cmap='YlGn', legend=True, edgecolor='black', linewidth=0.5)
    plt.title(f'Iowa Corn Yield by County, {year_to_plot}')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close('all')
    logging.info(f"Saved Iowa yield map for {year_to_plot} to {output_path}")
    time.sleep(1)  # Small delay to ensure resources are released

def plot_average_yield_map(yield_df, county_geojson_path, output_path):
    counties = gpd.read_file(county_geojson_path)
    # Ensure counties_gdf only contains Iowa counties before merging (STATEFP == '19')
    iowa_counties_gdf = counties[counties['STATEFP'] == '19'].copy()
    
    # Group by county_fips and calculate mean yield, ensuring only Iowa counties from yield_df are considered (implicitly done by geojson merge)
    # Also exclude aggregate data (FIPS '000') before grouping
    avg_yield = yield_df[yield_df['county_fips'] != '000'].groupby('county_fips')['yield_bu_per_acre'].mean().reset_index()
    avg_yield['county_fips'] = avg_yield['county_fips'].astype(str).str.zfill(3)
    
    # Merge average yield data with Iowa county boundaries
    merged = iowa_counties_gdf.merge(avg_yield, left_on='COUNTYFP', right_on='county_fips', how='left')
    
    plt.figure(figsize=(12, 8))
    # Use the merged GeoDataFrame for plotting
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
        # Pass yield_df to the map functions
        plot_iowa_yield_map(yield_df, "data/raw/iowa_counties_2020.geojson", "results/iowa_yield_map.png")
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