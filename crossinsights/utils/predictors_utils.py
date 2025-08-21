import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
from crossinsights.utils.data_loader import load_config, load_data

# إعداد السجلات
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='C:/crossinsights_project/crossinsights/logs/predictors_utils.log',
    filemode='a'
)
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

def prepare_features_table(users: pd.DataFrame, movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Prepare a features table by merging user, movie, and rating data."""
    logger.info("Preparing features table for prediction...")

    # التأكد من الأعمدة المطلوبة
    required_user_cols = ['user_id', 'age', 'gender', 'occupation']
    required_movie_cols = ['movie_id', 'genres']
    required_rating_cols = ['user_id', 'movie_id', 'rating']

    if not all(col in users.columns for col in required_user_cols):
        logger.error("Missing required columns in users DataFrame")
        raise ValueError("Missing required columns in users DataFrame")
    if not all(col in movies.columns for col in required_movie_cols):
        logger.error("Missing required columns in movies DataFrame")
        raise ValueError("Missing required columns in movies DataFrame")
    if not all(col in ratings.columns for col in required_rating_cols):
        logger.error("Missing required columns in ratings DataFrame")
        raise ValueError("Missing required columns in ratings DataFrame")

    # دمج البيانات
    features = ratings.merge(users, on='user_id', how='left')
    features = features.merge(movies, on='movie_id', how='left')

    # تحديد الأعمدة المطلوبة
    feature_cols = ['user_id', 'movie_id', 'age', 'gender', 'occupation', 'genres', 'rating']
    features = features[feature_cols]

    # معالجة المتغيرات الفئوية
    le_gender = LabelEncoder()
    le_occupation = LabelEncoder()
    features['gender'] = le_gender.fit_transform(features['gender'].fillna('Unknown'))
    features['occupation'] = le_occupation.fit_transform(features['occupation'].fillna('Unknown'))

    # تحويل الأنواع (genres) إلى أعمدة مشفرة
    features['genres_list'] = features['genres'].apply(
        lambda x: x.split('|') if isinstance(x, str) and x else ['Unknown']
    )

    # إنشاء أعمدة مشفرة للأنواع
    genres_encoded = pd.get_dummies(features['genres_list'].explode()).groupby(level=0).sum()
    features = pd.concat([features.drop(columns=['genres', 'genres_list']), genres_encoded], axis=1)

    # تسجيل عدد الأعمدة الناتجة
    logger.info(f"Features table prepared with {features.shape[1]} columns")
    return features
