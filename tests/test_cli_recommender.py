"""
Unit tests for CrossInsights CLI recommender.
Uses real data samples and edge cases for robust testing.
"""

import pytest
import pandas as pd
from crossinsights.utils.knn_recommender import KNNRecommender
from crossinsights.cli_recommender import load_resources, find_movie_id, get_recommendations_knn, get_recommendations_svd
from crossinsights.utils.data_loader import load_config
import os
import logging

# إعداد السجلات للاختبارات
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='C:/crossinsights_project/crossinsights/logs/test_cli_recommender.log',
    filemode='a'
)
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

@pytest.fixture
def setup_data():
    """Load a sample of real data from processed files."""
    logger.info("Setting up test data...")
    config = load_config()
    movies_path = config['data']['processed']['movies']
    ratings_path = config['data']['processed']['ratings']

    # تحميل عينة من البيانات الحقيقية
    try:
        movies = pd.read_csv(movies_path).head(100)  # تقليل الحجم لتسريع الاختبار
        ratings = pd.read_csv(ratings_path).head(1000)
        logger.info("Test data loaded successfully.")
        return movies, ratings
    except Exception as e:
        logger.error(f"Failed to load test data: {str(e)}")
        pytest.skip(f"Failed to load test data: {str(e)}")

def test_load_resources():
    """Test loading resources."""
    logger.info("Testing load_resources...")
    try:
        movies, ratings, knn_model, svd_model = load_resources()
        assert isinstance(movies, pd.DataFrame), "Movies should be a DataFrame"
        assert isinstance(ratings, pd.DataFrame), "Ratings should be a DataFrame"
        assert 'movie_id' in movies.columns, "Movies DataFrame missing movie_id"
        assert 'rating' in ratings.columns, "Ratings DataFrame missing rating"
        assert isinstance(knn_model, KNNRecommender), "KNN model should be KNNRecommender"
        assert hasattr(svd_model, 'transform'), "SVD model should have transform method"
        logger.info("load_resources test passed.")
    except FileNotFoundError:
        logger.warning("Model files not found, skipping test.")
        pytest.skip("Model files not found, skipping test.")

def test_find_movie_id(setup_data):
    """Test finding movie ID with real data."""
    logger.info("Testing find_movie_id...")
    movies, _ = setup_data
    movie_id = find_movie_id(movies, "The Godfather")
    assert movie_id == 2, f"Expected movie_id 2 for 'The Godfather', got {movie_id}"

    movie_id = find_movie_id(movies, "Nonexistent Movie")
    assert movie_id is None, "Expected None for nonexistent movie"
    logger.info("find_movie_id test passed.")

def test_get_recommendations_knn(setup_data):
    """Test KNN recommendations with real data."""
    logger.info("Testing get_recommendations_knn...")
    movies, ratings = setup_data
    config = load_config()
    knn_model = KNNRecommender(k_neighbors=config['models']['knn']['k_neighbors'])
    knn_model.fit(ratings, movies)

    recommendations = get_recommendations_knn(knn_model, ratings, movie_id=2, n_recommendations=5)
    assert isinstance(recommendations, pd.DataFrame), "Recommendations should be a DataFrame"
    assert 'movie_id' in recommendations.columns, "Recommendations missing movie_id column"
    assert 'score' in recommendations.columns, "Recommendations missing score column"
    assert len(recommendations) <= 5, f"Expected up to 5 recommendations, got {len(recommendations)}"
    assert 2 not in recommendations['movie_id'].values, "Recommended movies should not include the input movie"
    logger.info("get_recommendations_knn test passed.")

def test_get_recommendations_svd(setup_data):
    """Test SVD recommendations with a dummy user."""
    logger.info("Testing get_recommendations_svd...")
    movies, ratings = setup_data
    from crossinsights.models.placeholder import train_model

    # تدريب نموذج SVD
    svd_model = train_model(ratings)

    # إنشاء مستخدم وهمي
    max_user_id = ratings['user_id'].max()
    dummy_user_id = max_user_id + 1
    dummy_rating = pd.DataFrame({
        'user_id': [dummy_user_id],
        'movie_id': [2],  # The Godfather
        'rating': [5.0]
    })
    extended_ratings = pd.concat([ratings, dummy_rating], ignore_index=True)

    recommendations = get_recommendations_svd(svd_model, extended_ratings, movies, movie_id=2, n_recommendations=5)
    assert isinstance(recommendations, pd.DataFrame), "Recommendations should be a DataFrame"
    assert 'movie_id' in recommendations.columns, "Recommendations missing movie_id column"
    assert 'score' in recommendations.columns, "Recommendations missing score column"
    assert len(recommendations) <= 5, f"Expected up to 5 recommendations, got {len(recommendations)}"
    assert 2 not in recommendations['movie_id'].values, "Recommended movies should not include the input movie"
    logger.info("get_recommendations_svd test passed.")

def test_get_recommendations_knn_no_genre_filter(setup_data):
    """Test KNN recommendations without genre filtering."""
    logger.info("Testing get_recommendations_knn with no genre filtering...")
    movies, ratings = setup_data
    config = load_config()
    knn_model = KNNRecommender(k_neighbors=config['models']['knn']['k_neighbors'])
    knn_model.fit(ratings, movies)

    recommendations = get_recommendations_knn(knn_model, ratings, movie_id=2, n_recommendations=5, use_genre_filtering=False)
    assert isinstance(recommendations, pd.DataFrame), "Recommendations should be a DataFrame"
    assert 'movie_id' in recommendations.columns, "Recommendations missing movie_id column"
    assert 'score' in recommendations.columns, "Recommendations missing score column"
    assert len(recommendations) <= 5, f"Expected up to 5 recommendations, got {len(recommendations)}"
    assert 2 not in recommendations['movie_id'].values, "Recommended movies should not include the input movie"
    logger.info("get_recommendations_knn_no_genre_filter test passed.")

def test_get_recommendations_svd_no_genre_filter(setup_data):
    """Test SVD recommendations without genre filtering."""
    logger.info("Testing get_recommendations_svd with no genre filtering...")
    movies, ratings = setup_data
    from crossinsights.models.placeholder import train_model

    # تدريب نموذج SVD
    svd_model = train_model(ratings)

    # إنشاء مستخدم وهمي
    max_user_id = ratings['user_id'].max()
    dummy_user_id = max_user_id + 1
    dummy_rating = pd.DataFrame({
        'user_id': [dummy_user_id],
        'movie_id': [2],  # The Godfather
        'rating': [5.0]
    })
    extended_ratings = pd.concat([ratings, dummy_rating], ignore_index=True)

    recommendations = get_recommendations_svd(svd_model, extended_ratings, movies, movie_id=2, n_recommendations=5, use_genre_filtering=False)
    assert isinstance(recommendations, pd.DataFrame), "Recommendations should be a DataFrame"
    assert 'movie_id' in recommendations.columns, "Recommendations missing movie_id column"
    assert 'score' in recommendations.columns, "Recommendations missing score column"
    assert len(recommendations) <= 5, f"Expected up to 5 recommendations, got {len(recommendations)}"
    assert 2 not in recommendations['movie_id'].values, "Recommended movies should not include the input movie"
    logger.info("get_recommendations_svd_no_genre_filter test passed.")

def test_edge_cases(setup_data):
    """Test edge cases like invalid movie IDs."""
    logger.info("Testing edge cases...")
    movies, ratings = setup_data
    config = load_config()
    knn_model = KNNRecommender(k_neighbors=config['models']['knn']['k_neighbors'])
    knn_model.fit(ratings, movies)

    # اختبار معرف فيلم غير موجود
    recommendations = get_recommendations_knn(knn_model, ratings, movie_id=99999, n_recommendations=5)
    assert recommendations.empty, "Expected empty recommendations for invalid movie_id"
    logger.info("edge_cases test passed.")
