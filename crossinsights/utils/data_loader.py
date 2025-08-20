# crossinsights/utils/data_loader.py
import pandas as pd
import yaml
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='C:/crossinsights_project/crossinsights/logs/data_loader.log',
    filemode='a'
)
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path("C:/crossinsights_project/crossinsights/config/config.yaml")
    logger.info(f"Loading configuration from {config_path}")
    if not config_path.exists():
        logger.error(f"Config file not found at: {config_path}")
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    logger.info("Configuration loaded successfully")
    return config

def load_data(file_path):
    """Load CSV data from a given path."""
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            logger.error(f"Data file at {file_path} is empty")
            raise ValueError(f"Data file at {file_path} is empty")
        logger.info(f"Data loaded successfully from {file_path}")
        return df
    except FileNotFoundError:
        logger.error(f"Data file not found at: {file_path}")
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    except pd.errors.EmptyDataError:
        logger.error(f"Data file at {file_path} is empty")
        raise ValueError(f"Data file at {file_path} is empty")

def load_raw_data():
    """Load raw users, movies, and ratings data."""
    logger.info("Loading raw data...")
    config = load_config()
    users = load_data(config['data']['raw']['users'])
    movies = load_data(config['data']['raw']['movies'])
    ratings = load_data(config['data']['raw']['ratings'])
    logger.info("Raw data loaded successfully")
    return users, movies, ratings

def save_processed_data(df, output_path):
    """Save processed data to CSV."""
    logger.info(f"Saving processed data to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to {output_path}")
