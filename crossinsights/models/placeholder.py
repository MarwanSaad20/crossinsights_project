import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from crossinsights.utils.data_loader import load_config
import joblib
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='C:/crossinsights_project/crossinsights/logs/models.log',
    filemode='a'
)
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

def train_model(ratings: pd.DataFrame):
    """
    Train an SVD-based recommendation model.
    """
    logger.info("Training SVD model...")
    config = load_config()

    if not {'user_id', 'movie_id', 'rating'}.issubset(ratings.columns):
        logger.error("Required columns not found in ratings DataFrame")
        raise ValueError("Required columns not found in ratings DataFrame")

    # Aggregate duplicate ratings if any
    duplicates_count = ratings.duplicated(subset=['user_id', 'movie_id']).sum()
    if duplicates_count > 0:
        logger.warning(f"Found {duplicates_count} duplicate user-movie ratings. Aggregating by mean.")
        ratings = ratings.groupby(['user_id', 'movie_id'])['rating'].mean().reset_index()


    # Create user-item matrix
    user_item_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

    # Initialize and train SVD
    svd = TruncatedSVD(
        n_components=config['models']['svd']['n_factors'],
        n_iter=config['models']['svd']['n_epochs'],
        random_state=42
    )
    svd.fit(user_item_matrix)

    # Save model
    output_path = os.path.join(config['models']['output_dir'], 'svd_model.pkl')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(svd, output_path)
    logger.info(f"SVD model trained and saved to {output_path}")
    return svd

def predict(svd_model, ratings: pd.DataFrame, user_id: int, n_recommendations: int = 10):
    """
    Predict top-N movie recommendations for a given user.
    """
    logger.info(f"Generating recommendations for user {user_id}")
    if not {'user_id', 'movie_id', 'rating'}.issubset(ratings.columns):
        logger.error("Required columns not found in ratings DataFrame")
        raise ValueError("Required columns not found in ratings DataFrame")

    # Aggregate duplicate ratings if any
    duplicates_count = ratings.duplicated(subset=['user_id', 'movie_id']).sum()
    if duplicates_count > 0:
        logger.warning(f"Found {duplicates_count} duplicate user-movie ratings. Aggregating by mean.")
        ratings = ratings.groupby(['user_id', 'movie_id'])['rating'].mean().reset_index()


    # Create user-item matrix
    user_item_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

    # Get user vector
    if user_id not in user_item_matrix.index:
        logger.error(f"User {user_id} not found in ratings data")
        raise ValueError(f"User {user_id} not found in ratings data")

    user_vector = np.array(user_item_matrix.loc[user_id].values).reshape(1, -1)

    # Predict scores
    scores = svd_model.transform(user_vector) @ svd_model.components_
    movie_ids = user_item_matrix.columns
    recommendations = pd.DataFrame({
        'movie_id': movie_ids,
        'score': scores.flatten()
    })

    # Sort and get top-N recommendations
    recommendations = recommendations.sort_values(by='score', ascending=False).head(n_recommendations)
    logger.info(f"Top {n_recommendations} recommendations generated for user {user_id}")
    return recommendations
