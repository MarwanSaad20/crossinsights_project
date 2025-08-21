import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from crossinsights.utils.data_loader import load_config, save_processed_data
from crossinsights.utils.predictors_utils import prepare_features_table
import joblib
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# إعداد السجلات
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='C:/crossinsights_project/crossinsights/logs/predictors.log',
    filemode='a'
)
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

def train_predictors(users: pd.DataFrame, movies: pd.DataFrame, ratings: pd.DataFrame) -> tuple:
    """Train Linear Regression and Random Forest models for rating prediction."""
    logger.info("Training prediction models...")
    config = load_config()

    # إعداد جدول الميزات
    features_table = prepare_features_table(users, movies, ratings)

    # تحديد المتغيرات التفسيرية (X) والهدف (y)
    X = features_table.drop(columns=['user_id', 'movie_id', 'rating'])
    y = features_table['rating']

    # تقسيم البيانات إلى تدريب واختبار
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # تدريب نموذج الانحدار الخطي
    lr_model = LinearRegression(fit_intercept=config['models']['linear_regression']['fit_intercept'])
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)

    # تدريب نموذج الغابة العشوائية
    rf_model = RandomForestRegressor(
        n_estimators=config['models']['random_forest']['n_estimators'],
        max_depth=config['models']['random_forest']['max_depth'],
        random_state=config['models']['random_forest']['random_state']
    )
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)

    # حفظ النماذج
    lr_model_path = os.path.join(config['models']['output_dir'], 'linear_regression_model.pkl')
    rf_model_path = os.path.join(config['models']['output_dir'], 'random_forest_model.pkl')
    os.makedirs(os.path.dirname(lr_model_path), exist_ok=True)
    joblib.dump(lr_model, lr_model_path)
    joblib.dump(rf_model, rf_model_path)
    logger.info(f"Linear Regression model saved to {lr_model_path}")
    logger.info(f"Random Forest model saved to {rf_model_path}")

    # إنشاء ملف التنبؤات
    predictions_df = pd.DataFrame({
        'user_id': features_table.loc[X_test.index, 'user_id'],
        'movie_id': features_table.loc[X_test.index, 'movie_id'],
        'predicted_rating_linear': lr_predictions,
        'predicted_rating_forest': rf_predictions,
        'actual_rating': y_test
    })

    # حفظ التنبؤات
    predictions_path = config['data']['processed']['predicted_ratings']
    save_processed_data(predictions_df, predictions_path)
    logger.info(f"Predictions saved to {predictions_path}")

    # إنشاء رسم بياني للتحليل
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=predictions_df, x='actual_rating', label='Actual Ratings', color='blue')
    sns.kdeplot(data=predictions_df, x='predicted_rating_linear', label='Linear Regression', color='orange')
    sns.kdeplot(data=predictions_df, x='predicted_rating_forest', label='Random Forest', color='green')
    plt.title('Distribution of Actual vs Predicted Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Density')
    plt.legend()
    plot_path = config['analysis']['predictors']
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Prediction analysis plot saved to {plot_path}")

    return lr_model, rf_model, predictions_df
