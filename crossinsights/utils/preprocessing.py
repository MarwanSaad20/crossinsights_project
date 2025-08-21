# crossinsights/utils/preprocessing.py
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from crossinsights.utils.data_loader import load_config, save_processed_data, load_raw_data
from typing import Tuple
import numpy as np
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='C:/crossinsights_project/crossinsights/logs/preprocessing.log',
    filemode='a'
)
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

def report_missing(df: pd.DataFrame, name: str):
    """Log missing values report for a DataFrame."""
    missing_report = df.isnull().sum()
    total_rows = len(df)
    missing_summary = {col: f"{count} ({count/total_rows:.2%})"
                       for col, count in missing_report.items() if count > 0}
    if missing_summary:
        logger.info(f"Missing values in {name}: {missing_summary}")
    else:
        logger.info(f"No missing values in {name}")

def clean_users(df: pd.DataFrame) -> pd.DataFrame:
    """Clean users data: handle missing values and validate data types."""
    logger.info("Cleaning users data...")
    df = df.copy()
    report_missing(df, "users")

    # Ensure user_id is int64
    df['user_id'] = pd.to_numeric(df['user_id'], errors='coerce').astype('Int64')
    invalid_users = df['user_id'].isna().sum()
    if invalid_users > 0:
        logger.warning(f"Found {invalid_users} invalid user_id values. Dropping them.")
        df = df.dropna(subset=['user_id'])

    # Fill missing values intelligently
    df['age'] = df['age'].fillna(df['age'].median()).astype(int)
    df['gender'] = df['gender'].fillna('Unknown').astype(str)
    df['occupation'] = df['occupation'].fillna('Unknown').astype(str)
    df['zip_code'] = df['zip_code'].fillna('Unknown').astype(str)

    logger.info("Users cleaned: age median filled, other missing values replaced with 'Unknown'")
    return df

def clean_movies(df: pd.DataFrame) -> pd.DataFrame:
    """Clean movies data: handle missing titles and encode genres as one-hot."""
    logger.info("Cleaning movies data...")
    df = df.copy()
    report_missing(df, "movies")

    # Ensure movie_id is int64
    df['movie_id'] = pd.to_numeric(df['movie_id'], errors='coerce').astype('Int64')
    invalid_movies = df['movie_id'].isna().sum()
    if invalid_movies > 0:
        logger.warning(f"Found {invalid_movies} invalid movie_id values. Dropping them.")
        df = df.dropna(subset=['movie_id'])

    # Fill missing titles
    df['title'] = df['title'].fillna('Unknown').astype(str)

    # Convert genres to list
    df['genres_list'] = df['genres'].apply(lambda x: x.split('|') if isinstance(x, str) and x else ['Unknown'])

    # MultiLabel encoding
    mlb = MultiLabelBinarizer()
    encoded = mlb.fit_transform(df['genres_list'])

    # Convert encoded matrix to ndarray safely
    encoded_array = np.array(encoded)

    # Create DataFrame for encoded genres
    genres_encoded = pd.DataFrame(
        data=encoded_array,
        columns=[str(c) for c in mlb.classes_],
        index=df.index
    )

    # Concatenate original columns with encoded genres
    df_clean = pd.concat([df[['movie_id', 'title', 'genres']], genres_encoded], axis=1)

    # Log genre columns
    logger.info(f"Movies cleaned: {len(mlb.classes_)} genres encoded")
    logger.info(f"Genre columns: {genres_encoded.columns.tolist()}")

    return df_clean

def clean_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Clean ratings data: ensure valid ratings (1-5), handle duplicates, convert timestamps."""
    logger.info("Cleaning ratings data...")
    df = df.copy()
    report_missing(df, "ratings")

    # Ensure user_id and movie_id are int64
    df['user_id'] = pd.to_numeric(df['user_id'], errors='coerce').astype('Int64')
    df['movie_id'] = pd.to_numeric(df['movie_id'], errors='coerce').astype('Int64')
    invalid_ids = df[['user_id', 'movie_id']].isna().sum()
    if invalid_ids.sum() > 0:
        logger.warning(f"Found invalid IDs: {invalid_ids.to_dict()}. Dropping them.")
        df = df.dropna(subset=['user_id', 'movie_id'])

    # Convert rating column to numeric, coercing invalid values to NaN
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

    # Log and drop rows with invalid (NaN) ratings
    invalid_count = df['rating'].isna().sum()
    if invalid_count > 0:
        logger.warning(f"Found {invalid_count} invalid (non-numeric) ratings. Dropping them.")
        df = df.dropna(subset=['rating'])

    # Ensure ratings are within valid range (1-5)
    invalid_ratings = df[~df['rating'].between(1, 5)].shape[0]
    if invalid_ratings > 0:
        logger.warning(f"Found {invalid_ratings} ratings outside valid range (1-5). Dropping them.")
        df = df[df['rating'].between(1, 5)]

    # Convert timestamp safely
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce').astype('Int64')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')

    # Aggregate duplicate user-movie ratings
    duplicates_count = df.duplicated(subset=['user_id', 'movie_id']).sum()
    if duplicates_count > 0:
        logger.warning(f"Found {duplicates_count} duplicate user-movie pairs. Aggregating by mean rating.")
        df = pd.DataFrame(df.groupby(['user_id', 'movie_id'])['rating'].mean()).reset_index()

    logger.info(f"Ratings cleaned: {len(df)} valid ratings remaining")
    return df

def preprocess_all() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Preprocess all datasets and save to processed directory."""
    logger.info("Preprocessing all datasets...")
    config = load_config()
    users, movies, ratings = load_raw_data()

    users_clean = clean_users(users)
    movies_clean = clean_movies(movies)
    ratings_clean = clean_ratings(ratings)

    save_processed_data(users_clean, config['data']['processed']['users'])
    save_processed_data(movies_clean, config['data']['processed']['movies'])
    save_processed_data(ratings_clean, config['data']['processed']['ratings'])

    logger.info("All datasets preprocessed and saved")
    return users_clean, movies_clean, ratings_clean
