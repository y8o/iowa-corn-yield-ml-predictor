# src/feature_engineering.py

import pandas as pd
import numpy as np
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_gdd(temp_max, temp_min, base_temp=50, max_temp=86):
    """
    Calculate daily GDD from max and min temp.
    Uses the formula: GDD = ((Tmax + Tmin)/2) - base_temp
    """
    avg_temp = (temp_max + temp_min) / 2
    gdd = np.clip(avg_temp - base_temp, 0, max_temp - base_temp)
    return gdd

def process_climate_file(file_path, year):
    """
    Process a single climate CSV and return summarized features.
    """
    logging.info(f"Processing climate data for year {year}")
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['temp_f'] = pd.to_numeric(df['temp_f'], errors='coerce')
        df['precip_in'] = pd.to_numeric(df['precip_in'], errors='coerce')

        df = df[(df['month'] >= 4) & (df['month'] <= 9)]  # April to September
        df = df.dropna(subset=['temp_f'])

        # Assume temp_max and temp_min are estimated from rolling windows
        df['temp_max'] = df['temp_f'].rolling(3, min_periods=1).max()
        df['temp_min'] = df['temp_f'].rolling(3, min_periods=1).min()
        df['gdd'] = calculate_gdd(df['temp_max'], df['temp_min'])

        summary = {
            'year': year,
            'gdd_sum': df['gdd'].sum(),
            'precip_sum': df['precip_in'].sum(),
            'temp_mean': df['temp_f'].mean(),
            'temp_max': df['temp_f'].max(),
            'temp_min': df['temp_f'].min()
        }
        logging.info(f"Successfully processed {len(df)} records for year {year}")
        return pd.DataFrame([summary])
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {str(e)}")
        raise

def generate_features():
    """
    Loop through all climate CSVs and compile yearly feature summaries.
    """
    climate_dir = "data/raw/"
    output_dir = "data/processed/"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Verify input directory exists
    if not os.path.exists(climate_dir):
        raise FileNotFoundError(f"Input directory {climate_dir} does not exist")

    feature_rows = []
    for year in range(2013, 2024):  # Process years 2013-2023
        file_path = os.path.join(climate_dir, f"climate_dsm_{year}.csv")
        if os.path.exists(file_path):
            logging.info(f"Processing {file_path}")
            yearly_features = process_climate_file(file_path, year)
            feature_rows.append(yearly_features)
        else:
            logging.warning(f"Climate data file not found for year {year}: {file_path}")

    if not feature_rows:
        raise Exception("No climate data files were processed successfully")

    result = pd.concat(feature_rows, ignore_index=True)
    output_path = os.path.join(output_dir, "climate_features_2013_2023.csv")
    result.to_csv(output_path, index=False)
    logging.info(f"Saved processed features to {output_path}")
    return result

if __name__ == "__main__":
    try:
        features_df = generate_features()
        logging.info("Feature engineering completed successfully")
        logging.info("\nFeature summary:")
        logging.info(features_df.describe())
    except Exception as e:
        logging.error(f"Feature engineering failed: {str(e)}")
        raise