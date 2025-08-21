import sys
from pathlib import Path
sys.path.append(str(Path('C:/crossinsights_project')))

import unittest
import pandas as pd
import os
from crossinsights.utils.data_loader import load_config, load_data
from crossinsights.models.predictors import train_predictors

class TestPredictors(unittest.TestCase):
    def setUp(self):
        """Set up test environment and train predictors once."""
        self.config = load_config()
        self.users = load_data(self.config['data']['processed']['users'])
        self.movies = load_data(self.config['data']['processed']['movies'])
        self.ratings = load_data(self.config['data']['processed']['ratings'])
        self.predictions_path = self.config['data']['processed']['predicted_ratings']
        self.lr_model_path = os.path.join(self.config['models']['output_dir'], 'linear_regression_model.pkl')
        self.rf_model_path = os.path.join(self.config['models']['output_dir'], 'random_forest_model.pkl')
        self.plot_path = self.config['analysis']['predictors']

        # تشغيل train_predictors مرة واحدة وحفظ النتائج
        self.lr_model, self.rf_model, self.predictions_df = train_predictors(
            self.users, self.movies, self.ratings
        )

    def test_train_predictors_no_errors(self):
        """Test that predictors train without errors."""
        self.assertIsNotNone(self.lr_model, "Linear Regression model is None")
        self.assertIsNotNone(self.rf_model, "Random Forest model is None")
        self.assertIsNotNone(self.predictions_df, "Predictions DataFrame is None")

    def test_predictions_file_created(self):
        """Test that predictions CSV file is created."""
        self.assertTrue(os.path.exists(self.predictions_path),
                        f"Predictions file not found at {self.predictions_path}")

    def test_predictions_columns(self):
        """Test that predictions CSV contains required columns."""
        predictions = load_data(self.predictions_path)
        required_cols = ['user_id', 'movie_id', 'predicted_rating_linear', 'predicted_rating_forest']
        self.assertTrue(all(col in predictions.columns for col in required_cols),
                        f"Missing columns in predictions: {required_cols}")

    def test_model_files_created(self):
        """Test that model files are created."""
        self.assertTrue(os.path.exists(self.lr_model_path),
                        f"Linear Regression model file not found at {self.lr_model_path}")
        self.assertTrue(os.path.exists(self.rf_model_path),
                        f"Random Forest model file not found at {self.rf_model_path}")

    def test_analysis_plot_created(self):
        """Test that analysis plot is created."""
        self.assertTrue(os.path.exists(self.plot_path),
                        f"Analysis plot not found at {self.plot_path}")

if __name__ == '__main__':
    unittest.main()
