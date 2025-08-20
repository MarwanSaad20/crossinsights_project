# crossinsights/utils/knn_recommender.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from crossinsights.utils.data_loader import load_config, save_processed_data
import logging
from typing import Optional, cast

# إعداد السجلات
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='C:/crossinsights_project/crossinsights/logs/knn_recommender.log',
    filemode='a'
)
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

class KNNRecommender:
    """KNN-based recommender system for user-user and item-item recommendations."""

    def __init__(self, k_neighbors: int = 5, similarity_metric: str = 'cosine'):
        self.k_neighbors = k_neighbors
        self.similarity_metric = similarity_metric
        self.user_item_matrix: Optional[pd.DataFrame] = None
        self.user_similarity: Optional[pd.DataFrame] = None
        self.item_similarity: Optional[pd.DataFrame] = None
        self.logger = logger

    def fit(self, ratings: pd.DataFrame) -> None:
        """Build user-item matrix and compute similarities."""
        self.logger.info("Building user-item matrix...")
        if not {'user_id', 'movie_id', 'rating'}.issubset(ratings.columns):
            self.logger.error("Required columns not found in ratings DataFrame")
            raise ValueError("Required columns not found in ratings DataFrame")

        # إنشاء مصفوفة المستخدم-العنصر
        self.user_item_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
        self.logger.info(f"User-item matrix created with shape: {self.user_item_matrix.shape}")

        # حساب التشابه بين المستخدمين (user-user)
        self.logger.info(f"Computing {self.similarity_metric} similarity for users...")
        user_similarity_array = cosine_similarity(self.user_item_matrix)
        self.user_similarity = pd.DataFrame(
            user_similarity_array,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        self.logger.info("User-user similarity computed")

        # حساب التشابه بين الأفلام (item-item)
        self.logger.info(f"Computing {self.similarity_metric} similarity for items...")
        item_similarity_array = cosine_similarity(self.user_item_matrix.T)
        self.item_similarity = pd.DataFrame(
            item_similarity_array,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )
        self.logger.info("Item-item similarity computed")

    def predict_user_user(self, user_id: int, n_recommendations: int = 10) -> pd.DataFrame:
        """Generate recommendations using user-user collaborative filtering."""
        self.logger.info(f"Generating user-user recommendations for user {user_id}")
        if self.user_item_matrix is None or self.user_similarity is None:
            self.logger.error("Model not fitted. Call fit() first.")
            raise ValueError("Model not fitted. Call fit() first.")

        if user_id not in self.user_item_matrix.index:
            self.logger.warning(f"User {user_id} not found. Cold-start problem detected.")
            return pd.DataFrame(columns=['movie_id', 'predicted_rating'])

        # الحصول على أقرب الجيران
        user_similarities = cast(pd.Series, self.user_similarity.loc[user_id]).sort_values(ascending=False)[1:self.k_neighbors + 1]
        neighbor_ids = user_similarities.index
        self.logger.info(f"Top {self.k_neighbors} neighbors for user {user_id}: {neighbor_ids.tolist()}")

        # الأفلام التي لم يقم المستخدم بتقييمها
        user_ratings = cast(pd.Series, self.user_item_matrix.loc[user_id])
        unrated_movies_mask = user_ratings == 0
        unrated_movies = cast(pd.Series, user_ratings[unrated_movies_mask]).index

        # توقع التقييمات بناءً على الجيران
        predictions = []
        for movie_id in unrated_movies:
            neighbor_ratings = cast(pd.Series, self.user_item_matrix.loc[neighbor_ids, movie_id])
            valid_ratings = neighbor_ratings[neighbor_ratings > 0]
            valid_similarities = cast(pd.Series, user_similarities[neighbor_ratings > 0])
            if valid_ratings.empty:
                continue
            weighted_sum = np.sum(valid_ratings * valid_similarities)
            similarity_sum = np.sum(np.abs(valid_similarities))
            predicted_rating = weighted_sum / similarity_sum if similarity_sum > 0 else 0
            predictions.append({'movie_id': movie_id, 'predicted_rating': predicted_rating})

        recommendations = pd.DataFrame(predictions)
        if not recommendations.empty:
            recommendations = recommendations.sort_values(by='predicted_rating', ascending=False).head(n_recommendations)
        self.logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
        return recommendations

    def predict_item_item(self, user_id: int, n_recommendations: int = 10) -> pd.DataFrame:
        """Generate recommendations using item-item collaborative filtering."""
        self.logger.info(f"Generating item-item recommendations for user {user_id}")
        if self.user_item_matrix is None or self.item_similarity is None:
            self.logger.error("Model not fitted. Call fit() first.")
            raise ValueError("Model not fitted. Call fit() first.")

        if user_id not in self.user_item_matrix.index:
            self.logger.warning(f"User {user_id} not found. Cold-start problem detected.")
            return pd.DataFrame(columns=['movie_id', 'predicted_rating'])

        # الأفلام التي قيّمها المستخدم
        user_ratings = cast(pd.Series, self.user_item_matrix.loc[user_id])
        rated_movies_mask = user_ratings > 0
        rated_movies = cast(pd.Series, user_ratings[rated_movies_mask]).index

        # الأفلام التي لم يقم المستخدم بتقييمها
        unrated_movies_mask = user_ratings == 0
        unrated_movies = cast(pd.Series, user_ratings[unrated_movies_mask]).index

        # توقع التقييمات بناءً على تشابه الأفلام
        predictions = []
        for movie_id in unrated_movies:
            similarities = cast(pd.Series, self.item_similarity.loc[movie_id])[rated_movies]
            valid_ratings = user_ratings[rated_movies]
            valid_similarities = similarities[similarities > 0]
            valid_ratings = valid_ratings[similarities > 0]
            if valid_ratings.empty:
                continue
            weighted_sum = np.sum(valid_ratings * valid_similarities)
            similarity_sum = np.sum(np.abs(valid_similarities))
            predicted_rating = weighted_sum / similarity_sum if similarity_sum > 0 else 0
            predictions.append({'movie_id': movie_id, 'predicted_rating': predicted_rating})

        recommendations = pd.DataFrame(predictions)
        if not recommendations.empty:
            recommendations = recommendations.sort_values(by='predicted_rating', ascending=False).head(n_recommendations)
        self.logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
        return recommendations

    def save_model(self, output_path: str) -> None:
        """Save the KNN model."""
        self.logger.info(f"Saving KNN model to {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(self, output_path)
        self.logger.info(f"KNN model saved to {output_path}")

    def plot_similarity_distribution(self, output_path: str) -> None:
        """Plot the distribution of similarity scores."""
        self.logger.info("Generating similarity distribution plot")
        if self.user_similarity is None:
            self.logger.error("User similarity matrix not available. Call fit() first.")
            raise ValueError("User similarity matrix not available. Call fit() first.")

        plt.figure(figsize=(10, 6))
        sns.histplot(self.user_similarity.to_numpy().flatten(), bins=50, kde=True)
        plt.title('Distribution of User-User Similarity Scores')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Count')
        plt.savefig(output_path)
        plt.close()
        self.logger.info(f"Similarity distribution plot saved to {output_path}")

def run_knn_recommendations(ratings: pd.DataFrame, user_id: int = 1, n_recommendations: int = 10, mode: str = 'user-user') -> pd.DataFrame:
    """Run KNN recommendation pipeline and save results."""
    logger.info(f"Running {mode} KNN recommendations...")
    config = load_config()

    # إنشاء نموذج KNN
    knn = KNNRecommender(
        k_neighbors=config.get('models', {}).get('knn', {}).get('k_neighbors', 5),
        similarity_metric=config.get('models', {}).get('knn', {}).get('similarity_metric', 'cosine')
    )

    # تدريب النموذج
    knn.fit(ratings)

    # حفظ النموذج
    knn_model_path = os.path.join(config['models']['output_dir'], 'knn_model.pkl')
    knn.save_model(knn_model_path)

    # توليد التوصيات
    if mode == 'user-user':
        recommendations = knn.predict_user_user(user_id, n_recommendations)
    else:
        recommendations = knn.predict_item_item(user_id, n_recommendations)

    # حفظ التوصيات
    output_path = config['data']['processed']['knn_recommendations']
    save_processed_data(recommendations, output_path)

    # رسم توزيع التشابه
    plot_path = 'C:/crossinsights_project/crossinsights/analysis/knn_analysis.png'
    knn.plot_similarity_distribution(plot_path)

    logger.info(f"{mode} KNN recommendations completed for user {user_id}")
    return recommendations
