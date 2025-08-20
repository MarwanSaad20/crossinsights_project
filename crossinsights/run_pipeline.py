# crossinsights/run_pipeline.py
import logging
import sys
import argparse
from pathlib import Path
from crossinsights.utils.data_loader import load_raw_data, load_config
from crossinsights.utils.preprocessing import preprocess_all
from crossinsights.utils.eda_tools import plot_rating_distribution, plot_genre_popularity, plot_age_vs_ratings
from crossinsights.utils.knn_recommender import run_knn_recommendations  # جديد
from crossinsights.models.placeholder import train_model, predict

def setup_logging():
    """Set up logging Patient logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='C:/crossinsights_project/crossinsights/logs/pipeline.log',
        filemode='a'
    )
    logger = logging.getLogger(__name__)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger

def run_pipeline(stages=None):
    """Run the complete CrossInsights pipeline."""
    logger = setup_logging()
    logger.info("Starting CrossInsights pipeline...")

    summary = {
        'data_loaded': False,
        'data_preprocessed': False,
        'eda_completed': False,
        'knn_recommendations': False,  # جديد
        'model_trained': False,
        'recommendations_generated': False
    }

    try:
        # Step 1: Load configuration
        if not stages or 'load_config' in stages:
            logger.info("Loading configuration...")
            config = load_config()
            summary['data_loaded'] = True

        # newly added Step 2: Load raw data
        if not stages or 'load_data' in stages:
            logger.info("Loading raw data...")
            users, movies, ratings = load_raw_data()
            summary['data_loaded'] = True

        # Step 3: Preprocess data
        if not stages or 'preprocess' in stages:
            logger.info("Preprocessing data...")
            users_clean, movies_clean, ratings_clean = preprocess_all()
            summary['data_preprocessed'] = True

        # Step 4: Perform EDA
        if not stages or 'eda' in stages:
            logger.info("Performing exploratory data analysis...")
            plot_rating_distribution(ratings_clean)
            plot_genre_popularity(movies_clean)
            plot_age_vs_ratings(users_clean, ratings_clean)
            summary['eda_completed'] = True

        # Step 5: Run KNN Recommendations (جديد)
        if not stages or 'knn_recommendations' in stages:
            logger.info("Running KNN recommendations...")
            knn_recommendations = run_knn_recommendations(ratings_clean, user_id=1, n_recommendations=10, mode='user-user')
            logger.info(f"KNN recommendations for user 1:\n{knn_recommendations}")
            summary['knn_recommendations'] = True

        # Step 6: Train model
        if not stages or 'train_model' in stages:
            logger.info("Training model...")
            svd_model = train_model(ratings_clean)
            summary['model_trained'] = True

        # Step 7: Generate SVD recommendations
        if not stages or 'predict' in stages:
            logger.info("Generating SVD recommendations...")
            recommendations = predict(svd_model, ratings_clean, user_id=1, n_recommendations=10)
            logger.info(f"Top 10 SVD recommendations for user 1:\n{recommendations}")
            summary['recommendations_generated'] = True

        # Print summary
        logger.info("Pipeline Summary:")
        for key, value in summary.items():
            logger.info(f"{key.replace('_', ' ').title()}: {'Success' if value else 'Skipped'}")

        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == '__main__':
    # Ensure the project roots is in sys.path
    sys.path.append(str(Path('C:/crossinsights_project')))

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the CrossInsights pipeline")
    parser.add_argument('--stages', nargs='*', choices=['load_config', 'load_data', 'preprocess', 'eda', 'knn_recommendations', 'train_model', 'predict'],  # تحديث
                        help="Specific stages to run (default: all)")
    args = parser.parse_args()

    run_pipeline(stages=args.stages)
