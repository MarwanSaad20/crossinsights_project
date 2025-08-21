# crossinsights/models/placeholder.py
"""
Professional SVD-based recommendation model for CrossInsights.
Handles both existing and dummy users with robust error handling.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from crossinsights.utils.data_loader import load_config
import joblib
import os
import logging
from typing import Optional

# إعداد السجلات
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

def train_model(ratings: pd.DataFrame) -> TruncatedSVD:
    """
    Train an SVD-based recommendation model with professional error handling.

    Args:
        ratings (pd.DataFrame): DataFrame with user_id, movie_id, and rating columns.

    Returns:
        TruncatedSVD: Trained SVD model.
    """
    logger.info("Training SVD model...")
    config = load_config()

    # التحقق من الأعمدة
    required_columns = {'user_id', 'movie_id', 'rating'}
    if not required_columns.issubset(ratings.columns):
        logger.error(f"Required columns {required_columns} not found in ratings DataFrame")
        raise ValueError(f"Required columns {required_columns} not found in ratings DataFrame")

    # التعامل مع التكرارات
    duplicates_count = ratings.duplicated(subset=['user_id', 'movie_id']).sum()
    if duplicates_count > 0:
        logger.warning(f"Found {duplicates_count} duplicate user-movie ratings. Aggregating by mean.")
        ratings = ratings.groupby(['user_id', 'movie_id'])['rating'].mean().reset_index()

    # إنشاء مصفوفة المستخدم-الفيلم
    user_item_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    logger.info(f"User-item matrix created with shape: {user_item_matrix.shape}")

    # تحديد عدد المكونات
    n_factors = config['models']['svd']['n_factors']
    n_epochs = config['models']['svd']['n_epochs']
    n_features = user_item_matrix.shape[1]
    n_components = min(n_factors, max(1, n_features - 1))
    logger.info(f"Using n_components={n_components} (n_features={n_features})")

    # تدريب نموذج SVD
    try:
        svd = TruncatedSVD(n_components=n_components, n_iter=n_epochs, random_state=42)
        svd.fit(user_item_matrix)
    except Exception as e:
        logger.error(f"Failed to train SVD model: {str(e)}")
        raise

    # حفظ النموذج
    output_path = os.path.join(config['models']['output_dir'], 'svd_model.pkl')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(svd, output_path)
    logger.info(f"SVD model trained and saved to {output_path}")
    return svd

def predict(svd_model: TruncatedSVD, ratings: pd.DataFrame, user_id: int, n_recommendations: int = 10) -> pd.DataFrame:
    """
    Predict top-N movie recommendations for a user, supporting dummy users.

    Args:
        svd_model: Trained SVD model.
        ratings (pd.DataFrame): DataFrame with user_id, movie_id, and rating columns.
        user_id (int): ID of the user (existing or dummy).
        n_recommendations (int): Number of recommendations to return.

    Returns:
        pd.DataFrame: DataFrame with movie_id and score columns.
    """
    logger.info(f"Generating recommendations for user {user_id}")

    # التحقق من الأعمدة
    required_columns = {'user_id', 'movie_id', 'rating'}
    if not required_columns.issubset(ratings.columns):
        logger.error(f"Required columns {required_columns} not found in ratings DataFrame")
        raise ValueError(f"Required columns {required_columns} not found in ratings DataFrame")

    # إنشاء مصفوفة المستخدم-الفيلم
    try:
        user_item_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    except Exception as e:
        logger.error(f"Failed to create user-item matrix: {str(e)}")
        return pd.DataFrame(columns=['movie_id', 'score'])

    # التحقق من وجود المستخدم
    if user_id not in user_item_matrix.index:
        logger.warning(f"User {user_id} not found in user-item matrix. Assuming dummy user.")
        # إضافة المستخدم الوهمي إذا لم يكن موجودًا
        user_item_matrix.loc[user_id] = 0
        if ratings[ratings['user_id'] == user_id].empty:
            logger.warning(f"No ratings provided for user {user_id}. Returning empty recommendations.")
            return pd.DataFrame(columns=['movie_id', 'score'])

    # الحصول على متجه المستخدم
    try:
        user_vector = np.array(user_item_matrix.loc[user_id].values).reshape(1, -1)
        scores = svd_model.transform(user_vector) @ svd_model.components_
        movie_ids = user_item_matrix.columns
        recommendations = pd.DataFrame({
            'movie_id': movie_ids,
            'score': scores.flatten()
        })
    except Exception as e:
        logger.error(f"Failed to generate scores for user {user_id}: {str(e)}")
        return pd.DataFrame(columns=['movie_id', 'score'])

    # تصفية الأفلام التي قيّمها المستخدم بالفعل
    rated_movies = ratings[ratings['user_id'] == user_id]['movie_id']
    recommendations = recommendations[~recommendations['movie_id'].isin(rated_movies)]

    # ترتيب التوصيات واختيار الأعلى تقييمًا
    recommendations = recommendations.sort_values(by='score', ascending=False).head(n_recommendations)
    logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
    return recommendations.reset_index(drop=True)
