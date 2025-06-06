# src/data_collection.py

import pandas as pd
import numpy as np
import requests
import geopandas as gpd
from io import StringIO
import os
import logging
from dotenv import load_dotenv
import tempfile
import shutil
import zipfile
import urllib.request
import ssl
import us

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# ----------------------------------------
# 1. Get USDA Corn Yield Data by County
# ----------------------------------------
def get_usda_data(year_start=2013):
    """Fetch corn yield data from USDA NASS for Iowa counties (2013 onwards)."""
    logging.info("Downloading USDA NASS corn yield data...")
    base_url = "https://quickstats.nass.usda.gov/api/api_GET/"
    params = {
        'key': os.getenv('USDA_API_KEY'),
        'commodity_desc': 'CORN',
        'year__GE': str(year_start),
        'state_alpha': 'IA',
        'format': 'JSON'
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        logging.info(f"USDA API Response Status: {response.status_code}")
        
        data = response.json()
        if 'data' not in data:
            logging.error(f"Unexpected API response format: {data}")
            raise Exception("API response missing 'data' field")
            
        df = pd.DataFrame(data['data'])
        if df.empty:
            logging.error("No data returned from USDA API")
            raise Exception("Empty dataframe returned from USDA API")
            
        df = df[['year', 'county_name', 'county_ansi', 'Value']]
        df.columns = ['year', 'county', 'county_fips', 'yield_bu_per_acre']
        df['year'] = df['year'].astype(int)
        df['yield_bu_per_acre'] = pd.to_numeric(df['yield_bu_per_acre'].str.replace(",", ""), errors='coerce')
        df['county_fips'] = df['county_fips'].astype(str).str.zfill(3)
        df = df.dropna()
        logging.info(f"Successfully processed {len(df)} records from USDA API")
        return df
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch USDA data: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error processing USDA data: {str(e)}")
        raise

# ----------------------------------------
# 2. Get Climate Data from IEM
# ----------------------------------------
def get_iem_climate_data(station='DSM', year=2022):
    """Fetch daily temp and precipitation data from IEM for a given station and year."""
    logging.info(f"Downloading IEM climate data for {station}, {year}...")
    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    params = {
        "station": station,
        "data": "tmpf,p01i",
        "year1": year,
        "month1": 4,
        "day1": 1,
        "year2": year,
        "month2": 9,
        "day2": 30,
        "tz": "Etc/UTC",
        "format": "comma",
        "latlon": "yes",
        "missing": "M",
        "trace": "T",
        "direct": "yes"
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        logging.info(f"IEM API Response Status: {response.status_code}")
        
        content = response.content.decode('utf-8')
        df = pd.read_csv(StringIO(content), skiprows=5)
        if df.empty:
            logging.error("No data returned from IEM API")
            raise Exception("Empty dataframe returned from IEM API")
            
        df.columns = [c.strip().lower() for c in df.columns]
        df['date'] = pd.to_datetime(df['valid'])
        df = df[['date', 'tmpf', 'p01i']].rename(columns={'tmpf': 'temp_f', 'p01i': 'precip_in'})
        df = df.dropna()
        logging.info(f"Successfully processed {len(df)} records from IEM API")
        return df
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch climate data: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error processing climate data: {str(e)}")
        raise

# ----------------------------------------
# 3. Get Iowa County Boundaries
# ----------------------------------------
def load_iowa_counties():
    """
    Download and load Iowa county boundaries from the Census Bureau TIGER/Line shapefile.
    Returns a GeoDataFrame with county boundaries for Iowa only.
    """
    import zipfile
    import io
    import requests
    TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER2022/COUNTY/tl_2022_us_county.zip"
    out_zip = "data/raw/tl_2022_us_county.zip"
    out_shp = "data/raw/tl_2022_us_county.shp"
    logging.info("Downloading US county boundaries shapefile from Census Bureau...")
    try:
        if not os.path.exists(out_zip):
            r = requests.get(TIGER_URL, stream=True)
            r.raise_for_status()
            with open(out_zip, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            logging.info(f"Downloaded shapefile zip to {out_zip}")
        else:
            logging.info(f"Shapefile zip already exists at {out_zip}")
        # Extract if not already extracted
        with zipfile.ZipFile(out_zip, 'r') as zip_ref:
            zip_ref.extractall("data/raw/")
        logging.info("Extracted shapefile contents.")
        # Read with geopandas
        gdf = gpd.read_file(out_shp)
        # Filter for Iowa (STATEFP == '19')
        iowa_gdf = gdf[gdf['STATEFP'] == '19'].copy()
        if iowa_gdf.empty:
            raise Exception("No Iowa counties found in shapefile.")
        logging.info(f"Loaded {len(iowa_gdf)} Iowa counties.")
        return iowa_gdf
    except Exception as e:
        logging.error(f"Error loading county boundaries: {str(e)}")
        raise

# ----------------------------------------
# 4. Main Execution
# ----------------------------------------
if __name__ == "__main__":
    try:
        # Create data directory if it doesn't exist
        os.makedirs('data/raw', exist_ok=True)
        
        # 1. Download and save corn yield data
        usda_df = get_usda_data()
        usda_path = 'data/raw/usda_yield_2013_2023.csv'
        usda_df.to_csv(usda_path, index=False)
        logging.info(f"Saved USDA yield data to {usda_path}")

        # 2. Download and save climate data for 2013-2023
        for year in range(2013, 2024):
            climate_df = get_iem_climate_data('DSM', year)
            climate_path = f'data/raw/climate_dsm_{year}.csv'
            climate_df.to_csv(climate_path, index=False)
            logging.info(f"Saved climate data to {climate_path}")

        # 3. Download and save Iowa county shapefile
        counties_gdf = load_iowa_counties()
        counties_path = "data/raw/iowa_counties_2020.geojson"
        counties_gdf.to_file(counties_path, driver='GeoJSON')
        logging.info(f"Saved county boundaries to {counties_path}")
        
    except Exception as e:
        logging.error(f"Script failed: {str(e)}")
        raise