# crossinsights/utils/knn_recommender.py
"""
Professional KNN-based recommender system for CrossInsights.
Supports user-user and item-item collaborative filtering with improved genre-based filtering.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from crossinsights.utils.data_loader import load_config, save_processed_data, load_data
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
        self.movies: Optional[pd.DataFrame] = None
        self.logger = logger

    def fit(self, ratings: pd.DataFrame, movies: Optional[pd.DataFrame] = None) -> None:
        """Build user-item matrix and compute similarities."""
        self.logger.info("Building user-item matrix...")
        if not {'user_id', 'movie_id', 'rating'}.issubset(ratings.columns):
            self.logger.error("Required columns not found in ratings DataFrame")
            raise ValueError("Required columns not found in ratings DataFrame")

        # إنشاء مصفوفة المستخدم-العنصر
        try:
            self.user_item_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
            self.logger.info(f"User-item matrix created with shape: {self.user_item_matrix.shape}")
        except Exception as e:
            self.logger.error(f"Failed to create user-item matrix: {str(e)}")
            raise

        # تحميل بيانات الأفلام لتصفية الأنواع
        if movies is None:
            config = load_config()
            movies_path = config['data']['processed']['movies']
            self.movies = load_data(movies_path)
        else:
            self.movies = movies

        # التحقق من وجود عمود الأنواع
        if self.movies is not None and 'genres' not in self.movies.columns:
            self.logger.warning("Genres column not found in movies DataFrame. Adding empty genres.")
            self.movies['genres'] = 'Unknown'

        # حساب التشابه بين المستخدمين
        self.logger.info(f"Computing {self.similarity_metric} similarity for users...")
        try:
            user_similarity_array = cosine_similarity(self.user_item_matrix)
            self.user_similarity = pd.DataFrame(
                user_similarity_array,
                index=self.user_item_matrix.index,
                columns=self.user_item_matrix.index
            )
            self.logger.info("User-user similarity computed")
        except Exception as e:
            self.logger.error(f"Failed to compute user-user similarity: {str(e)}")
            raise

        # حساب التشابه بين الأفلام
        self.logger.info(f"Computing {self.similarity_metric} similarity for items...")
        try:
            item_similarity_array = cosine_similarity(self.user_item_matrix.T)
            self.item_similarity = pd.DataFrame(
                item_similarity_array,
                index=self.user_item_matrix.columns,
                columns=self.user_item_matrix.columns
            )
            self.logger.info("Item-item similarity computed")
        except Exception as e:
            self.logger.error(f"Failed to compute item-item similarity: {str(e)}")
            raise

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

        # توقع التقييمات
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

    def predict_item_item(self, movie_id: int, n_recommendations: int = 10, use_genre_filtering: bool = True) -> pd.DataFrame:
        """Generate recommendations using item-item collaborative filtering with optional smart genre filtering."""
        self.logger.info(f"Generating item-item recommendations for movie_id {movie_id}")
        if self.user_item_matrix is None or self.item_similarity is None or self.movies is None:
            self.logger.error("Model not fitted or movies data not loaded. Call fit() first.")
            raise ValueError("Model not fitted or movies data not loaded.")

        if movie_id not in self.item_similarity.index:
            self.logger.warning(f"Movie {movie_id} not found in item similarity matrix.")
            return pd.DataFrame(columns=['movie_id', 'predicted_rating'])

        # الحصول على تشابه الأفلام
        similarities = cast(pd.Series, self.item_similarity.loc[movie_id]).sort_values(ascending=False)[1:]

        # إنشاء التوصيات الأولية مع عدد أكبر لضمان التصفية اللاحقة
        initial_count = max(n_recommendations * 3, 50)  # أخذ عدد أكبر لضمان وجود توصيات بعد التصفية
        initial_recommendations = pd.DataFrame({
            'movie_id': similarities.head(initial_count).index.astype(int),
            'predicted_rating': similarities.head(initial_count).values
        })

        if not use_genre_filtering or initial_recommendations.empty:
            final_recommendations = initial_recommendations.head(n_recommendations)
            self.logger.info(f"Generated {len(final_recommendations)} recommendations for movie_id {movie_id} (no genre filtering)")
            return final_recommendations

        # تطبيق تصفية الأنواع الذكية
        movie_info = self.movies[self.movies['movie_id'] == movie_id]
        if movie_info.empty or 'genres' not in movie_info.columns or pd.isna(movie_info['genres'].iloc[0]):
            self.logger.warning(f"No valid genres found for movie_id {movie_id}. Returning recommendations without genre filtering.")
            final_recommendations = initial_recommendations.head(n_recommendations)
            self.logger.info(f"Generated {len(final_recommendations)} recommendations for movie_id {movie_id}")
            return final_recommendations

        # الحصول على أنواع الفيلم المدخل
        movie_genres = movie_info['genres'].iloc[0].split('|')
        self.logger.info(f"Source movie genres: {movie_genres}")

        # تصفية التوصيات بناءً على الأنواع
        filtered_movies = self.movies[self.movies['movie_id'].isin(initial_recommendations['movie_id'])].copy()
        filtered_movies['genre_match'] = filtered_movies['genres'].apply(
            lambda x: any(g in x.split('|') for g in movie_genres) if pd.notna(x) else False
        )

        # ترتيب التوصيات: أولاً التي تتطابق أنواعها، ثم الباقي
        genre_matched = filtered_movies[filtered_movies['genre_match']]['movie_id']
        genre_unmatched = filtered_movies[~filtered_movies['genre_match']]['movie_id']

        # دمج التوصيات مع إعطاء أولوية للأفلام المتطابقة الأنواع
        matched_recommendations = initial_recommendations[initial_recommendations['movie_id'].isin(genre_matched)]
        unmatched_recommendations = initial_recommendations[initial_recommendations['movie_id'].isin(genre_unmatched)]

        # إذا لم نجد توصيات متطابقة الأنواع كافية، نأخذ من غير المتطابقة
        if len(matched_recommendations) >= n_recommendations:
            final_recommendations = matched_recommendations.head(n_recommendations)
            self.logger.info(f"Generated {len(final_recommendations)} genre-matched recommendations")
        else:
            # أخذ كل المتطابقة + بعض غير المتطابقة
            remaining_count = n_recommendations - len(matched_recommendations)
            combined_recommendations = pd.concat([
                matched_recommendations,
                unmatched_recommendations.head(remaining_count)
            ], ignore_index=True)
            final_recommendations = combined_recommendations.head(n_recommendations)
            self.logger.info(f"Generated {len(matched_recommendations)} genre-matched + {len(final_recommendations) - len(matched_recommendations)} other recommendations")

        self.logger.info(f"Generated {len(final_recommendations)} total recommendations for movie_id {movie_id}")
        return final_recommendations

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

    # تحميل بيانات الأفلام
    movies_path = config['data']['processed']['movies']
    movies = load_data(movies_path)

    # إنشاء نموذج KNN
    knn = KNNRecommender(
        k_neighbors=config.get('models', {}).get('knn', {}).get('k_neighbors', 5),
        similarity_metric=config.get('models', {}).get('knn', {}).get('similarity_metric', 'cosine')
    )

    # تدريب النموذج
    knn.fit(ratings, movies)

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
