# tests/test_data_loader.py
import unittest
import pandas as pd
import sys
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='C:/crossinsights_project/crossinsights/logs/test_data_loader.log',
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

from crossinsights.utils.data_loader import load_raw_data, load_config, load_data

class TestDataLoader(unittest.TestCase):
    def test_load_raw_data(self):
        logger.info("Testing load_raw_data...")
        try:
            users, movies, ratings = load_raw_data()
            self.assertIsInstance(users, pd.DataFrame)
            self.assertIsInstance(movies, pd.DataFrame)
            self.assertIsInstance(ratings, pd.DataFrame)
            self.assertTrue(len(users) > 0)
            self.assertTrue(len(movies) > 0)
            self.assertTrue(len(ratings) > 0)
            logger.info("load_raw_data test passed")
        except Exception as e:
            logger.error(f"load_raw_data test failed: {str(e)}")
            raise

    def test_config_loading(self):
        logger.info("Testing load_config...")
        try:
            config = load_config()
            self.assertIn('data', config)
            self.assertIn('raw', config['data'])
            self.assertIn('processed', config['data'])
            logger.info("load_config test passed")
        except Exception as e:
            logger.error(f"load_config test failed: {str(e)}")
            raise

    def test_load_data_file_not_found(self):
        logger.info("Testing load_data with nonexistent file...")
        with self.assertRaises(FileNotFoundError):
            load_data("C:/crossinsights_project/crossinsights/data/raw/nonexistent_file.csv")
        logger.info("load_data_file_not_found test passed")

if __name__ == '__main__':
    unittest.main()
