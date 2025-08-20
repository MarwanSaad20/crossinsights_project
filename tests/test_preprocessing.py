# tests/test_preprocessing.py
import unittest
import pandas as pd
import sys
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='C:/crossinsights_project/crossinsights/logs/test_preprocessing.log',
    filemode='a'
)
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from crossinsights.utils.preprocessing import clean_users, clean_movies, clean_ratings, preprocess_all

class TestPreprocessing(unittest.TestCase):
    def test_clean_users(self):
        logger.info("Testing clean_users...")
        df = pd.DataFrame({
            'user_id': [1, 2],
            'age': [25, None],
            'gender': ['M', None],
            'occupation': ['student', None],
            'zip_code': ['12345', None]
        })
        try:
            cleaned = clean_users(df)
            self.assertFalse(cleaned['age'].isnull().any())
            self.assertFalse(cleaned['gender'].isnull().any())
            self.assertFalse(cleaned['occupation'].isnull().any())
            self.assertFalse(cleaned['zip_code'].isnull().any())
            logger.info("clean_users test passed")
        except Exception as e:
            logger.error(f"clean_users test failed: {str(e)}")
            raise

    def test_clean_movies(self):
        logger.info("Testing clean_movies...")
        df = pd.DataFrame({
            'movie_id': [1, 2],
            'title': ['Movie A', None],
            'genres': ['Action|Comedy', 'Drama']
        })
        try:
            cleaned = clean_movies(df)
            self.assertFalse(cleaned['title'].isnull().any())
            self.assertTrue('Action' in cleaned.columns)
            self.assertTrue('Comedy' in cleaned.columns)
            self.assertTrue('Drama' in cleaned.columns)
            logger.info("clean_movies test passed")
        except Exception as e:
            logger.error(f"clean_movies test failed: {str(e)}")
            raise

    def test_clean_ratings(self):
        logger.info("Testing clean_ratings...")
        df = pd.DataFrame({
            'user_id': [1, 2],
            'movie_id': [1, 2],
            'rating': [6, 0],
            'timestamp': [1234567890, 1234567891]
        })
        try:
            cleaned = clean_ratings(df)
            self.assertTrue(cleaned['rating'].between(1, 5).all())
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned['timestamp']))
            logger.info("clean_ratings test passed")
        except Exception as e:
            logger.error(f"clean_ratings test failed: {str(e)}")
            raise

    def test_preprocess_all(self):
        logger.info("Testing preprocess_all...")
        try:
            users, movies, ratings = preprocess_all()
            self.assertIsInstance(users, pd.DataFrame)
            self.assertIsInstance(movies, pd.DataFrame)
            self.assertIsInstance(ratings, pd.DataFrame)
            self.assertFalse(users['age'].isnull().any())
            self.assertTrue('Action' in movies.columns or 'Drama' in movies.columns)
            self.assertTrue(ratings['rating'].between(1, 5).all())
            logger.info("preprocess_all test passed")
        except Exception as e:
            logger.error(f"preprocess_all test failed: {str(e)}")
            raise

if __name__ == '__main__':
    unittest.main()
