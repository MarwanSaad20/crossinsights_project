# tests/test_knn_recommender.py
import unittest
import pandas as pd
import sys
from pathlib import Path
import logging
import numpy as np

# إعداد السجلات
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='C:/crossinsights_project/crossinsights/logs/test_knn_recommender.log',
    filemode='a'
)
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

# إضافة جذر المشروع إلى sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from crossinsights.utils.knn_recommender import KNNRecommender

class TestKNNRecommender(unittest.TestCase):
    def setUp(self):
        self.logger = logger
        self.ratings = pd.DataFrame({
            'user_id': [1, 1, 2, 2],
            'movie_id': [101, 102, 101, 103],
            'rating': [4, 5, 3, 4]
        })
        self.knn = KNNRecommender(k_neighbors=2)

    def test_fit(self):
        self.logger.info("Testing KNNRecommender.fit...")
        try:
            self.knn.fit(self.ratings)
            self.assertIsNotNone(self.knn.user_item_matrix)
            self.assertIsNotNone(self.knn.user_similarity)
            self.assertIsNotNone(self.knn.item_similarity)
            if self.knn.user_item_matrix is not None:
                self.assertEqual(self.knn.user_item_matrix.shape, (2, 3))
            self.logger.info("KNNRecommender.fit test passed")
        except Exception as e:
            self.logger.error(f"KNNRecommender.fit test failed: {str(e)}")
            raise

    def test_predict_user_user(self):
        self.logger.info("Testing KNNRecommender.predict_user_user...")
        try:
            self.knn.fit(self.ratings)
            recommendations = self.knn.predict_user_user(user_id=1, n_recommendations=2)
            self.assertIsInstance(recommendations, pd.DataFrame)
            self.assertTrue('movie_id' in recommendations.columns)
            self.assertTrue('predicted_rating' in recommendations.columns)
            self.logger.info("KNNRecommender.predict_user_user test passed")
        except Exception as e:
            self.logger.error(f"KNNRecommender.predict_user_user test failed: {str(e)}")
            raise

    def test_predict_item_item(self):
        self.logger.info("Testing KNNRecommender.predict_item_item...")
        try:
            self.knn.fit(self.ratings)
            # استخدام movie_id بدلاً من user_id (مثلاً 101 موجود بالبيانات)
            recommendations = self.knn.predict_item_item(movie_id=101, n_recommendations=2)
            self.assertIsInstance(recommendations, pd.DataFrame)
            self.assertTrue('movie_id' in recommendations.columns)
            self.assertTrue('predicted_rating' in recommendations.columns)
            self.logger.info("KNNRecommender.predict_item_item test passed")
        except Exception as e:
            self.logger.error(f"KNNRecommender.predict_item_item test failed: {str(e)}")
            raise

    def test_cold_start(self):
        self.logger.info("Testing KNNRecommender cold-start handling...")
        try:
            self.knn.fit(self.ratings)
            recommendations = self.knn.predict_user_user(user_id=999, n_recommendations=2)
            self.assertTrue(recommendations.empty)
            self.logger.info("KNNRecommender cold-start test passed")
        except Exception as e:
            self.logger.error(f"KNNRecommender cold-start test failed: {str(e)}")
            raise

if __name__ == '__main__':
    unittest.main()
