# crossinsights/cli_recommender.py
"""
Professional CLI recommender for CrossInsights.
Supports KNN and SVD recommendations with genre-based filtering and robust error handling.
Integrates with updated KNNRecommender for smart genre filtering.
"""

import pandas as pd
import argparse
import logging
import os
import joblib
import numpy as np
from fuzzywuzzy import process, fuzz
from crossinsights.utils.data_loader import load_config, load_data
from crossinsights.utils.knn_recommender import KNNRecommender
from crossinsights.models.placeholder import predict
from typing import Optional, Tuple
import re

# إعداد السجلات
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='C:/crossinsights_project/crossinsights/logs/cli_recommender.log',
    filemode='a'
)
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

def load_resources() -> Tuple[pd.DataFrame, pd.DataFrame, KNNRecommender, object]:
    """Load necessary data and models.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, KNNRecommender, object]:
        movies, ratings, knn_model, svd_model
    """
    logger.info("Loading resources for CLI recommender...")
    config = load_config()

    # تحميل البيانات
    movies_path = config['data']['processed']['movies']
    ratings_path = config['data']['processed']['ratings']
    movies = load_data(movies_path)
    ratings = load_data(ratings_path)

    # التحقق من وجود عمود الأنواع
    if 'genres' not in movies.columns:
        logger.warning("Genres column not found in movies DataFrame. Adding empty genres.")
        movies['genres'] = 'Unknown'

    # التحقق من جودة بيانات الأنواع
    godfather = movies[movies['title'].str.contains('The Godfather \\(1972\\)', na=False)]
    if not godfather.empty and pd.notna(godfather['genres'].iloc[0]):
        genres = godfather['genres'].iloc[0]
        if 'Crime' not in genres:
            logger.warning(f"Unexpected genres for 'The Godfather (1972)': {genres}. Expected 'Drama|Crime'.")

    # تحميل النماذج
    knn_model_path = os.path.join(config['models']['output_dir'], 'knn_model.pkl')
    svd_model_path = os.path.join(config['models']['output_dir'], 'svd_model.pkl')

    if not os.path.exists(knn_model_path) or not os.path.exists(svd_model_path):
        logger.error("One or both model files not found.")
        raise FileNotFoundError("Model files (knn_model.pkl or svd_model.pkl) not found.")

    knn_model = joblib.load(knn_model_path)
    svd_model = joblib.load(svd_model_path)

    # تهيئة بيانات الأفلام لنموذج KNN
    if not hasattr(knn_model, 'movies') or knn_model.movies is None:
        logger.info("Initializing KNN model with movies data.")
        knn_model.movies = movies
        try:
            knn_model.fit(ratings, movies)
        except Exception as e:
            logger.error(f"Failed to re-fit KNN model: {str(e)}")
            raise

    logger.info("Resources loaded successfully.")
    return movies, ratings, knn_model, svd_model

def clean_title(title: str) -> str:
    """Clean movie title by removing special characters, extra spaces, and converting to lowercase."""
    title = re.sub(r'[\'"]', '', title)  # إزالة الاقتباسات
    title = re.sub(r'\s+', ' ', title.strip())  # إزالة المسافات الزائدة
    title = re.sub(r'[^\w\s]', '', title)  # إزالة الأحرف الخاصة
    return title.lower()

def find_movie_id(movies: pd.DataFrame, movie_title: str) -> Optional[int]:
    """Find movie ID by partial or full title match using fuzzy matching.

    Args:
        movies (pd.DataFrame): DataFrame containing movie information
        movie_title (str): Movie title to search for

    Returns:
        Optional[int]: Movie ID if found, None otherwise
    """
    logger.info(f"Searching for movie: {movie_title}")
    # تنظيف عنوان الفيلم المدخل
    movie_title_cleaned = clean_title(movie_title)

    # إنشاء قائمة بالعناوين المنظفة مع الاحتفاظ بالفهرس الأصلي
    titles_with_index = [(clean_title(title), idx) for idx, title in enumerate(movies['title'])]
    cleaned_titles = [t[0] for t in titles_with_index]

    # استخدام fuzzywuzzy للبحث الذكي
    match_result = process.extractOne(movie_title_cleaned, cleaned_titles, score_cutoff=80, scorer=fuzz.ratio)

    if match_result is None:
        logger.warning(f"No match found for movie: {movie_title}")
        print(f"❌ لم يتم العثور على الفيلم: {movie_title}")
        # عرض أقرب التطابقات المحتملة
        closest_matches = process.extract(movie_title_cleaned, cleaned_titles, limit=5, scorer=fuzz.ratio)
        if closest_matches:
            print("💡 أقرب التطابقات:")
            for match in closest_matches:
                # فك النتيجة بناءً على طولها
                if len(match) == 3:
                    match_title, score, _ = match
                else:
                    match_title, score = match
                # العثور على الفهرس الأصلي
                original_index = next((idx for title, idx in titles_with_index if title == match_title), None)
                if original_index is not None:
                    original_title = movies['title'].iloc[original_index]
                    print(f" - {original_title} (تشابه: {score}%)")
        print("💡 تأكد من كتابة اسم الفيلم بشكل صحيح أو جرب اسماً مختلفاً")
        return None

    # فك النتيجة بناءً على طولها
    if len(match_result) == 3:
        matched_title, score, _ = match_result
    else:
        matched_title, score = match_result
    # العثور على الفهرس الأصلي
    original_index = next((idx for title, idx in titles_with_index if title == matched_title), None)
    if original_index is None:
        logger.error(f"Could not find original index for matched title: {matched_title}")
        return None

    movie_id = int(movies.iloc[original_index]['movie_id'])
    original_title = movies['title'].iloc[original_index]

    logger.info(f"Found movie: {original_title} (movie_id: {movie_id}, score: {score})")
    print(f"✅ تم العثور على الفيلم: {original_title}")
    return movie_id

def get_recommendations_knn(knn_model: KNNRecommender, ratings: pd.DataFrame,
                            movie_id: int, n_recommendations: int = 5,
                            use_genre_filtering: bool = True) -> pd.DataFrame:
    """Generate item-item recommendations using KNN model with smart genre filtering.

    Args:
        knn_model (KNNRecommender): Trained KNN model
        ratings (pd.DataFrame): Ratings data
        movie_id (int): Movie ID to get recommendations for
        n_recommendations (int): Number of recommendations to return
        use_genre_filtering (bool): Whether to apply genre filtering

    Returns:
        pd.DataFrame: Recommendations with movie_id and score columns
    """
    logger.info(f"Generating KNN recommendations for movie_id: {movie_id}")

    if knn_model.item_similarity is None or knn_model.user_item_matrix is None:
        logger.error("KNN model not properly initialized.")
        raise ValueError("KNN model not properly initialized.")

    # تحميل بيانات الأفلام إذا لم تكن موجودة
    if knn_model.movies is None:
        config = load_config()
        movies_path = config['data']['processed']['movies']
        knn_model.movies = load_data(movies_path)
        logger.info("Loaded movies data into KNN model.")

    if movie_id not in knn_model.item_similarity.index:
        logger.warning(f"Movie {movie_id} not found in item similarity matrix.")
        return pd.DataFrame(columns=['movie_id', 'score'])

    # توليد التوصيات مع أو بدون تصفية الأنواع
    try:
        recommendations = knn_model.predict_item_item(movie_id, n_recommendations, use_genre_filtering)
        # إعادة تسمية العمود لتوافق الاختبارات
        if 'predicted_rating' in recommendations.columns:
            recommendations = recommendations.rename(columns={'predicted_rating': 'score'})
        logger.info(f"Generated {len(recommendations)} KNN recommendations.")
        return recommendations
    except Exception as e:
        logger.error(f"Error generating KNN recommendations: {str(e)}")
        return pd.DataFrame(columns=['movie_id', 'score'])

def get_recommendations_svd(svd_model, ratings: pd.DataFrame, movies: pd.DataFrame, movie_id: int,
                            n_recommendations: int = 5, use_genre_filtering: bool = True) -> pd.DataFrame:
    """Generate recommendations using SVD model for a dummy user with optional genre filtering.

    Args:
        svd_model: Trained SVD model
        ratings (pd.DataFrame): Ratings data
        movies (pd.DataFrame): Movies data for genre filtering
        movie_id (int): Movie ID that the dummy user likes
        n_recommendations (int): Number of recommendations to return
        use_genre_filtering (bool): Whether to apply genre filtering

    Returns:
        pd.DataFrame: Recommendations with movie_id and score columns
    """
    logger.info(f"Generating SVD recommendations for movie_id: {movie_id}")

    # التحقق من صحة بيانات التقييمات
    required_columns = {'user_id', 'movie_id', 'rating'}
    if not required_columns.issubset(ratings.columns):
        logger.error(f"Required columns {required_columns} not found in ratings DataFrame")
        return pd.DataFrame(columns=['movie_id', 'score'])

    # إنشاء مصفوفة المستخدم-الفيلم
    try:
        user_item_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    except Exception as e:
        logger.error(f"Failed to create user-item matrix: {str(e)}")
        return pd.DataFrame(columns=['movie_id', 'score'])

    # إنشاء معرف مستخدم وهمي
    max_user_id = int(user_item_matrix.index.max())
    dummy_user_id = max_user_id + 1

    # إضافة المستخدم الوهمي
    user_item_matrix.loc[dummy_user_id] = 0
    if movie_id in user_item_matrix.columns:
        user_item_matrix.loc[dummy_user_id, movie_id] = 5  # افتراض تقييم عالي للفيلم
    else:
        logger.warning(f"Movie {movie_id} not found in ratings data.")
        return pd.DataFrame(columns=['movie_id', 'score'])

    # تحويل إلى تنسيق ratings
    dummy_ratings = user_item_matrix.loc[dummy_user_id].reset_index()
    dummy_ratings.columns = ['movie_id', 'rating']
    dummy_ratings = dummy_ratings[dummy_ratings['rating'] > 0][['movie_id', 'rating']]
    dummy_ratings['user_id'] = dummy_user_id
    dummy_ratings = dummy_ratings[['user_id', 'movie_id', 'rating']]
    extended_ratings = pd.concat([ratings, dummy_ratings], ignore_index=True)

    # توليد التوصيات
    try:
        initial_count = max(n_recommendations * 3, 50)  # طلب عدد أكبر لضمان وجود توصيات بعد التصفية
        recommendations = predict(svd_model, extended_ratings, dummy_user_id, initial_count)
        logger.info(f"Generated {len(recommendations)} initial SVD recommendations.")
        if not use_genre_filtering:
            recommendations = recommendations.head(n_recommendations)
            logger.info(f"Generated {len(recommendations)} SVD recommendations without genre filtering.")
            return recommendations

        # تصفية بناءً على الأنواع
        movie_info = movies[movies['movie_id'] == movie_id]
        if not movie_info.empty and 'genres' in movie_info.columns and pd.notna(movie_info['genres'].iloc[0]):
            movie_genres = movie_info['genres'].iloc[0].split('|')
            filtered_movies = movies[movies['movie_id'].isin(recommendations['movie_id'])].copy()
            filtered_movies['genre_match'] = filtered_movies['genres'].apply(
                lambda x: any(g in x.split('|') for g in movie_genres) if pd.notna(x) else False
            )
            valid_movie_ids = filtered_movies[filtered_movies['genre_match']]['movie_id']
            filtered_recommendations = recommendations[recommendations['movie_id'].isin(valid_movie_ids)]
            if len(filtered_recommendations) >= n_recommendations:
                recommendations = filtered_recommendations.head(n_recommendations)
            else:
                # إذا لم يكن هناك توصيات كافية، أضف توصيات بدون تصفية
                logger.warning(f"Only {len(filtered_recommendations)} genre-matched recommendations found. Adding non-matched to reach {n_recommendations}.")
                remaining_count = n_recommendations - len(filtered_recommendations)
                non_matched = recommendations[~recommendations['movie_id'].isin(valid_movie_ids)].head(remaining_count)
                recommendations = pd.concat([filtered_recommendations, non_matched], ignore_index=True).head(n_recommendations)
            logger.info(f"Generated {len(recommendations)} SVD recommendations after genre filtering.")
        else:
            logger.warning(f"No valid genres for movie_id {movie_id}. Returning unfiltered recommendations.")
            recommendations = recommendations.head(n_recommendations)
        return recommendations
    except Exception as e:
        logger.error(f"Error generating SVD recommendations: {str(e)}")
        return pd.DataFrame(columns=['movie_id', 'score'])

def display_recommendations(recommendations: pd.DataFrame, movies: pd.DataFrame,
                           model_name: str, n: int = 5) -> None:
    """Display recommendations with movie titles, genres, and year (if available).

    Args:
        recommendations (pd.DataFrame): Recommendations with movie_id column
        movies (pd.DataFrame): Movies data with titles, genres, and optionally year
        model_name (str): Name of the recommendation model
        n (int): Number of recommendations to display
    """
    logger.info(f"Displaying top {n} {model_name} recommendations...")

    if recommendations.empty:
        print(f"❌ لا توجد توصيات متاحة من نموذج {model_name}")
        return

    print(f"\n📌 أفضل {n} توصيات ({model_name}):")
    recommendations_display = recommendations.head(n)

    if len(recommendations_display) < n:
        print(f"⚠️ تحذير: تم العثور على {len(recommendations_display)} توصيات فقط بدلاً من {n} بسبب تصفية الأنواع")

    for i, (idx, row) in enumerate(recommendations_display.iterrows()):
        try:
            movie_id = int(row['movie_id'])
            score = row.get('score', row.get('predicted_rating', 'N/A'))
            movie_info = movies[movies['movie_id'] == movie_id]
            if not movie_info.empty:
                title = movie_info['title'].iloc[0]
                genres = movie_info['genres'].iloc[0] if 'genres' in movie_info.columns and pd.notna(movie_info['genres'].iloc[0]) else 'Unknown'
                # استخراج السنة من العنوان إذا كانت موجودة
                year = re.search(r'\((\d{4})\)', title)
                year = year.group(1) if year else 'Unknown'
                display_num = i + 1
                if isinstance(score, (int, float)) and not pd.isna(score):
                    print(f"{display_num}. {title} (Year: {year}, Genres: {genres}, Score: {score:.3f})")
                else:
                    print(f"{display_num}. {title} (Year: {year}, Genres: {genres})")
            else:
                display_num = i + 1
                print(f"{display_num}. Unknown Movie (ID: {movie_id})")
        except (ValueError, KeyError) as e:
            logger.warning(f"Error displaying recommendation {i}: {str(e)}")
            continue

def validate_inputs(args) -> bool:
    """Validate command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        bool: True if inputs are valid, False otherwise
    """
    if args.n <= 0:
        print("❌ خطأ: عدد التوصيات يجب أن يكون أكبر من صفر")
        return False

    if args.n > 50:
        print("⚠️ تحذير: عدد التوصيات كبير جدًا، سيتم تقليله إلى 20")
        args.n = 20

    return True

def main():
    """Main function for CrossInsights CLI recommender."""
    parser = argparse.ArgumentParser(
        description="CrossInsights Movie Recommender CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
أمثلة على الاستخدام:
  python -m crossinsights.cli_recommender --movie "Titanic" --n 10 --model knn
  python -m crossinsights.cli_recommender --movie "Star Wars" --model svd --no-genre-filter
  python -m crossinsights.cli_recommender --movie "Toy Story"
        """
    )

    parser.add_argument('--movie', type=str,
                        help="Movie title to get recommendations for")
    parser.add_argument('--n', type=int, default=5,
                        help="Number of recommendations to display (default: 5)")
    parser.add_argument('--model', type=str,
                        choices=['knn', 'svd', 'both'], default='both',
                        help="Model to use for recommendations (knn, svd, or both)")
    parser.add_argument('--interactive', action='store_true',
                        help="Run in interactive mode")
    parser.add_argument('--no-genre-filter', action='store_true',
                        help="Disable genre filtering for recommendations")

    args = parser.parse_args()

    # التحقق من صحة المدخلات
    if not validate_inputs(args):
        logger.error("Invalid input arguments provided.")
        return

    print("🎬 أهلاً بك في CrossInsights Recommender!")
    print("=" * 50)

    # تحميل الموارد
    try:
        print("📂 جاري تحميل البيانات والنماذج...")
        movies, ratings, knn_model, svd_model = load_resources()
        print("✅ تم تحميل الموارد بنجاح")
        logger.info("Resources loaded: movies, ratings, KNN model, SVD model")
    except Exception as e:
        logger.error(f"Failed to load resources: {str(e)}")
        print(f"❌ خطأ: تعذر تحميل الموارد - {str(e)}")
        print("تأكد من وجود الملفات والنماذج وتشغيل pipeline أولاً")
        return

    # الحصول على إدخال المستخدم
    if args.interactive or not args.movie:
        movie_title = input("🎥 أدخل اسم فيلم تحبه: ").strip()
    else:
        movie_title = args.movie.strip()

    if not movie_title:
        logger.error("No movie title provided.")
        print("❌ يجب إدخال اسم فيلم صحيح")
        return

    # البحث عن الفيلم
    print(f"🔎 جاري البحث عن الفيلم: {movie_title}...")
    movie_id = find_movie_id(movies, movie_title)

    if movie_id is None:
        logger.warning(f"Movie {movie_title} not found.")
        print(f"❌ لم يتم العثور على الفيلم: {movie_title}")
        print("💡 تأكد من كتابة اسم الفيلم بشكل صحيح أو جرب اسماً مختلفاً")
        return

    matched_movie = movies[movies['movie_id'] == movie_id]['title'].iloc[0]
    print(f"✅ تم العثور على الفيلم: {matched_movie}")
    print("-" * 50)

    # توليد وعرض التوصيات
    try:
        if args.model in ['knn', 'both']:
            print("🤖 جاري توليد توصيات KNN...")
            knn_recommendations = get_recommendations_knn(knn_model, ratings, movie_id, args.n, not args.no_genre_filter)
            display_recommendations(knn_recommendations, movies, "KNN", args.n)

        if args.model in ['svd', 'both']:
            print("🎯 جاري توليد توصيات SVD...")
            svd_recommendations = get_recommendations_svd(svd_model, ratings, movies, movie_id, args.n, not args.no_genre_filter)
            display_recommendations(svd_recommendations, movies, "SVD", args.n)

    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        print(f"❌ خطأ: تعذر توليد التوصيات - {str(e)}")

    print("\n" + "=" * 50)
    print("🎬 شكراً لاستخدام CrossInsights Recommender!")

if __name__ == "__main__":
    main()
